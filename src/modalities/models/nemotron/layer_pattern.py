# The layer-pattern symbols are adopted from NVIDIA's Megatron-LM
# (megatron/core/models/hybrid/hybrid_layer_allocation.py::Symbols) so that pattern strings are
# portable between the two frameworks.
# Copyright (c) 2024-2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.

"""Layer pattern parsing for hybrid Mamba-Transformer models.

A hybrid model such as Nemotron-3 Nano is described by a pattern string in which every
character denotes one residual sublayer, e.g. ``"MEM*E"``. Each sublayer is a self-contained
pre-norm residual block (``x = x + f(norm(x))``); there is no combined "attention + MLP"
block as in a classical transformer.

The symbols follow the reference implementation in Megatron-LM
(``megatron/core/models/hybrid/hybrid_layer_allocation.py``) so that pattern strings can be
copied verbatim between the two frameworks.

A pattern may additionally contain **loop groups** written ``[<symbols>]^<K>``, which execute the
bracketed layers ``K`` times while reusing one set of weights (depth-wise weight sharing, as in a
Universal Transformer). ``"[ME]^3"`` builds one Mamba and one MoE layer and runs the pair three
times.

The repeat operator is ``^`` rather than ``*`` because ``*`` is the attention symbol: ``"M[ME]^3*E"``
reads as though the ``3`` were multiplied by an attention layer, whereas ``"M[ME]^3*E"`` is
unambiguous both to the parser and to a reader.

Bracket-free patterns parse exactly as before, so existing configs are unaffected.
"""

import re
from dataclasses import dataclass
from enum import Enum


class LayerSymbol(str, Enum):
    """
    Enum of the layer types that a hybrid layer pattern can contain.

    The values match the symbols used by Megatron-LM so that pattern strings are portable.

    Attributes:
        MAMBA (str): A Mamba-2 mixer layer.
        ATTENTION (str): A (grouped-query) self-attention layer.
        MOE (str): A mixture-of-experts feed-forward layer.
        MLP (str): A dense feed-forward layer.
    """

    MAMBA = "M"
    ATTENTION = "*"
    MOE = "E"
    MLP = "-"


def parse_layer_pattern(pattern: str) -> list[LayerSymbol]:
    """
    Parses a hybrid layer pattern string into a list of layer symbols.

    Args:
        pattern (str): The pattern string, e.g. ``"MEM*E"``.

    Raises:
        ValueError: If the pattern is empty or contains an unknown symbol.

    Returns:
        list[LayerSymbol]: One symbol per layer, in model order.
    """
    if len(pattern) == 0:
        raise ValueError("The layer pattern must not be empty.")

    valid_symbols = {symbol.value for symbol in LayerSymbol}
    layer_symbols: list[LayerSymbol] = []
    for position, character in enumerate(pattern):
        if character not in valid_symbols:
            hint = (
                " This pattern contains a loop group, which this function does not handle; "
                "use parse_layer_schedule instead."
                if character in "[]"
                else ""
            )
            raise ValueError(
                f"Invalid layer symbol '{character}' at position {position} of layer pattern '{pattern}'. "
                f"Valid symbols are {sorted(valid_symbols)}.{hint}"
            )
        layer_symbols.append(LayerSymbol(character))
    return layer_symbols


@dataclass(frozen=True)
class LoopGroup:
    """
    One execution unit of a layer schedule: a run of layers, executed ``num_loops`` times.

    A plain (non-looped) layer is represented as a group of one key with ``num_loops == 1``, so a
    schedule is uniformly a list of groups.

    Attributes:
        layer_keys (tuple[str, ...]): Keys into the model's layer ``ModuleDict``, in execution
            order. All iterations reuse these same layers, which is what makes the loop
            weight-shared.
        num_loops (int): How many times the run of layers is executed.
    """

    layer_keys: tuple[str, ...]
    num_loops: int

    @property
    def num_executed_layers(self) -> int:
        """
        Returns the number of layer applications this group performs.

        Returns:
            int: ``len(layer_keys) * num_loops``.
        """
        return len(self.layer_keys) * self.num_loops


# "[MEM]^3" -> group "MEM", repeat "3".
_LOOP_GROUP_RE = re.compile(r"\[([^\[\]]*)\]\^(\d+)")


def parse_layer_schedule(pattern: str) -> tuple[list[LayerSymbol], list[LoopGroup]]:
    """
    Parses a layer pattern into the layers to build and the order in which to execute them.

    Loop groups (``[<symbols>]^<K>``) build their layers **once** and execute them ``K`` times, so
    the returned symbol list contains one entry per set of weights, not per execution.

    Args:
        pattern (str): The pattern string, e.g. ``"M[ME]^3*E"``.

    Raises:
        ValueError: If the pattern is empty, contains an unknown symbol, has a malformed or nested
            loop group, or specifies a loop count below one.

    Returns:
        tuple[list[LayerSymbol], list[LoopGroup]]: The symbol of each layer to build (indexable by
            the integer form of the schedule's layer keys), and the execution schedule.
    """
    if len(pattern) == 0:
        raise ValueError("The layer pattern must not be empty.")

    valid_symbols = {symbol.value for symbol in LayerSymbol}
    layer_symbols: list[LayerSymbol] = []
    schedule: list[LoopGroup] = []

    def _append_layer(symbol_character: str) -> str:
        layer_symbols.append(LayerSymbol(symbol_character))
        return str(len(layer_symbols) - 1)

    position = 0
    while position < len(pattern):
        character = pattern[position]

        if character == "]":
            raise ValueError(
                f"Unmatched ']' at position {position} of layer pattern '{pattern}'. "
                f"A loop group is written as '[<symbols>]^<count>', e.g. '[ME]^3'."
            )

        if character == "[":
            match = _LOOP_GROUP_RE.match(pattern, position)
            if match is None:
                raise ValueError(
                    f"Malformed loop group starting at position {position} of layer pattern "
                    f"'{pattern}'. A loop group is written as '[<symbols>]^<count>', e.g. '[ME]^3'. "
                    f"Nested groups are not supported."
                )
            group_body, loop_count_text = match.group(1), match.group(2)
            if len(group_body) == 0:
                raise ValueError(f"Empty loop group at position {position} of layer pattern '{pattern}'.")
            num_loops = int(loop_count_text)
            if num_loops < 1:
                raise ValueError(
                    f"Loop count must be at least 1, got {num_loops} in loop group "
                    f"'{match.group(0)}' of layer pattern '{pattern}'."
                )
            for offset, group_character in enumerate(group_body):
                if group_character not in valid_symbols:
                    raise ValueError(
                        f"Invalid layer symbol '{group_character}' at position {position + 1 + offset} "
                        f"of layer pattern '{pattern}'. Valid symbols are {sorted(valid_symbols)}."
                    )
            schedule.append(LoopGroup(layer_keys=tuple(_append_layer(c) for c in group_body), num_loops=num_loops))
            position = match.end()
            continue

        if character not in valid_symbols:
            raise ValueError(
                f"Invalid layer symbol '{character}' at position {position} of layer pattern '{pattern}'. "
                f"Valid symbols are {sorted(valid_symbols)}."
            )
        schedule.append(LoopGroup(layer_keys=(_append_layer(character),), num_loops=1))
        position += 1

    return layer_symbols, schedule


def count_layers_by_type(pattern: str) -> dict[LayerSymbol, int]:
    """
    Counts how many layer *applications* of each type a pattern performs.

    A looped layer is counted once per iteration, because this count feeds cost models (FLOPs,
    pipeline balancing) that care about executed work rather than about how many weights exist.
    For bracket-free patterns this is the same as counting characters.

    Args:
        pattern (str): The pattern string, e.g. ``"M[ME]^3E"``.

    Returns:
        dict[LayerSymbol, int]: Counts for every layer type, including types with a count of zero.
    """
    layer_symbols, schedule = parse_layer_schedule(pattern)
    counts = {symbol: 0 for symbol in LayerSymbol}
    for group in schedule:
        for layer_key in group.layer_keys:
            counts[layer_symbols[int(layer_key)]] += group.num_loops
    return counts


def get_num_layers(pattern: str) -> int:
    """
    Returns the number of layer applications a pattern performs.

    Looped layers count once per iteration. Use :func:`get_num_built_layers` for the number of
    layers that are actually instantiated.

    Args:
        pattern (str): The pattern string, e.g. ``"M[ME]^3E"``.

    Returns:
        int: The number of executed layers.
    """
    _, schedule = parse_layer_schedule(pattern)
    return sum(group.num_executed_layers for group in schedule)


def get_num_built_layers(pattern: str) -> int:
    """
    Returns the number of layers a pattern instantiates, i.e. the number of distinct weight sets.

    A loop group contributes its bracketed layers once, however often it is executed.

    Args:
        pattern (str): The pattern string, e.g. ``"M[ME]^3E"``.

    Returns:
        int: The number of built layers.
    """
    layer_symbols, _ = parse_layer_schedule(pattern)
    return len(layer_symbols)
