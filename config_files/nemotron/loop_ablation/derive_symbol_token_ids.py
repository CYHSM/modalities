#!/usr/bin/env python
"""Derives the token ids that the synthetic reasoning evaluations use as their symbol alphabet.

The evaluation configs carry plain token ids rather than a tokenizer component, so that evaluation
has no runtime tokenizer dependency and the exact question sequences are reproducible from the
config alone. This script is what produces those ids, and re-running it is how you check that the
ids in the config still match the tokenizer the training data was built with.

Every symbol must encode to **exactly one** token. If it did not, the answer would not sit at a
single position and the masked target could not identify it; the script therefore refuses to emit
an alphabet with any multi-token symbol.

Run from the repository root::

    python config_files/nemotron/loop_ablation/derive_symbol_token_ids.py \\
        --tokenizer_path /raid/s3/opengptx/max_lue/repositories/training_datasets/llama3_tokenizer/\\
models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2

The default path is the Meta-Llama-3-8B-Instruct tokenizer recorded in the training data's
``tokenization_config.yaml``, i.e. the tokenizer the fineweb pbin files were produced with.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_TOKENIZER_PATH = Path(
    "/raid/s3/opengptx/max_lue/repositories/training_datasets/llama3_tokenizer/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
)

# Space-prefixed capitals. The leading space matters: it is what makes each letter a single token
# in a byte-level BPE vocabulary, and it also makes the rendered prompt (" A B C ...") look like
# text the model has actually seen rather than like a run of subword fragments.
SYMBOLS = [f" {chr(ord('A') + offset)}" for offset in range(26)]

# The statement separator and the assignment operator of the variable_binding task, in the order
# the dataset expects them in `delimiter_token_ids`.
DELIMITERS = [" ;", " ="]


def _encode_single_tokens(tokenizer, strings: list[str], description: str) -> list[int]:
    """
    Encodes each string, requiring every one of them to be a single token.

    Args:
        tokenizer: The tokenizer to encode with.
        strings (list[str]): The strings to encode.
        description (str): What this group is, used in the error message.

    Raises:
        ValueError: If any string does not encode to exactly one token.

    Returns:
        list[int]: One token id per string, in order.
    """
    encoded = {string: tokenizer.encode(string, add_special_tokens=False) for string in strings}
    multi_token = {string: ids for string, ids in encoded.items() if len(ids) != 1}
    if multi_token:
        raise ValueError(
            f"These {description} do not encode to a single token with this tokenizer: {multi_token}. "
            f"The synthetic evaluations need single-token symbols so that the answer occupies exactly "
            f"one position. Pick a different alphabet."
        )
    return [encoded[string][0] for string in strings]


def main() -> None:
    """Prints the symbol and delimiter token ids as YAML ready to paste into a config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.tokenizer_path))
    symbol_token_ids = _encode_single_tokens(tokenizer, SYMBOLS, "symbols")
    delimiter_token_ids = _encode_single_tokens(tokenizer, DELIMITERS, "delimiters")

    overlap = set(symbol_token_ids) & set(delimiter_token_ids)
    if overlap:
        raise ValueError(
            f"The delimiters share token ids {sorted(overlap)} with the symbols, which would make the "
            f"statement structure of the variable_binding task ambiguous."
        )

    print(f"# Derived from {arguments.tokenizer_path}")
    print(f"# symbols:    {' '.join(repr(symbol) for symbol in SYMBOLS)}")
    print(f"symbol_token_ids: {symbol_token_ids}")
    print(f"# delimiters: {' '.join(repr(delimiter) for delimiter in DELIMITERS)}")
    print(f"delimiter_token_ids: {delimiter_token_ids}")


if __name__ == "__main__":
    main()
