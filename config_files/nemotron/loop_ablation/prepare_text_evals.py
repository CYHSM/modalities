#!/usr/bin/env python
"""Tokenizes the Minerva MATH and TriviaQA evaluations once into prepared files with masked targets.

These two exist to be read **against each other**. The claim about looped models is that depth-wise
weight sharing buys reasoning without buying parameters, so the two should move in opposite
directions:

``minerva_math``
    Reasoning. Where looping is expected to help.

``triviaqa``
    Closed-book factual recall -- parametric knowledge, which lives in weights a loop does not add.
    This is the **control**: looping is expected to be neutral or *worse* here, and showing that
    dissociation is what makes a win on MATH credible rather than an artifact of an arm simply
    training better.

Why negative log-likelihood and not accuracy. At 1.1B parameters and a few billion tokens a model
scores ~zero on MATH and near-zero on closed-book TriviaQA; solve rate is not a measurement at this
scale, it is a constant. The likelihood assigned to the *correct* answer is graded and moves long
before accuracy leaves the floor.

Why offline. The synthetic evaluations generate their own token ids and need no tokenizer. These
are real text, so they must be tokenized -- and doing that at training time would make every run
depend on a tokenizer being present and would let a tokenizer change silently alter what the arms
are compared on. Tokenizing once and shipping token ids makes every arm's evaluation
byte-identical, the same reason the training data lives in pbin files.

WHAT "MINERVA MATH" MEANS HERE. It is not plain Hendrycks MATH. Following lm-evaluation-harness's
``minerva_math`` task (https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/minerva_math):
the same ``EleutherAI/hendrycks_math`` test split, but with Minerva's prompt
``Problem:\\n{problem}\\n\\nSolution:`` and its **4-shot** prefix, reproduced verbatim below from
``utils.py::list_fewshot_samples``. Minerva's other half -- SymPy answer normalization and
equivalence checking -- only exists to score a *generated* answer for exact match, and has no
counterpart in a likelihood evaluation, so it is not used. What is scored is the negative
log-likelihood of the reference solution given the 4-shot prefix and the problem.

Run from the repository root::

    python config_files/nemotron/loop_ablation/prepare_text_evals.py

Note what MATH measures. Most solution tokens are prose and LaTeX rather than reasoning, so a gain
confined to the reasoning steps arrives diluted. That is a sensitivity cost, not a validity
problem: the problems and tokens are identical across arms, so there is no evaluation sampling
noise and the limiting noise is the training seed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TOKENIZER_PATH = Path(
    "/raid/s3/opengptx/max_lue/repositories/training_datasets/llama3_tokenizer/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data" / "prepared_evals"

# The target value torch.nn.CrossEntropyLoss skips. Duplicated from
# modalities.dataloader.synthetic_reasoning so this script needs no import of the package.
IGNORE_INDEX = -100

# All seven MATH subjects, so the benchmark is not silently reduced to one topic.
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# The full test split is ~5000 problems, and each carries the ~500-token 4-shot prefix. Evaluating
# all of them at every interval would cost several times the language-modelling split for no gain:
# at ~200 scored tokens per problem, 1024 problems already pin the mean far tighter than the
# training-seed noise the comparison is limited by. Taken as a deterministic per-subject prefix so
# the selection is reproducible and balanced across subjects.
NUM_MATH_PROBLEMS = 1024

# TriviaQA answers are a few tokens each, so more questions are needed to accumulate a comparable
# number of scored tokens. A deterministic prefix of the validation split, for the same reason.
NUM_TRIVIAQA_QUESTIONS = 8192

# Verbatim from lm-evaluation-harness lm_eval/tasks/minerva_math/utils.py::list_fewshot_samples,
# kept as data rather than as literals so that no reformatting can ever alter them.
MINERVA_FEWSHOT_PATH = Path(__file__).with_name("minerva_fewshot_samples.json")
MINERVA_FEWSHOT_SAMPLES = json.loads(MINERVA_FEWSHOT_PATH.read_text())

# lm-evaluation-harness joins examples with "\n\n" and separates a prompt from its target with a
# single space, so the prefix below is byte-identical to what minerva_math builds at 4-shot.
MINERVA_FEWSHOT_PREFIX = "\n\n".join(
    f"Problem:\n{sample['problem']}\n\nSolution: {sample['solution']}" for sample in MINERVA_FEWSHOT_SAMPLES
)


def _minerva_prompt(problem: str) -> str:
    """
    Builds the 4-shot Minerva prompt for one problem.

    Args:
        problem (str): The problem statement.

    Returns:
        str: The prompt, ending in "Solution:" so that the reference solution continues it.
    """
    return f"{MINERVA_FEWSHOT_PREFIX}\n\nProblem:\n{problem}\n\nSolution:"


def _build_sample(tokenizer, prompt: str, answer: str, max_length: int) -> tuple[list[int], list[int]] | None:
    """
    Tokenizes one problem and masks everything except the reference answer.

    Args:
        tokenizer: A fast tokenizer, required for the character-to-token offset mapping.
        prompt (str): The text the model conditions on, which is never scored.
        answer (str): The reference answer, every token of which is scored.
        max_length (int): Drop problems longer than this.

    Returns:
        tuple[list[int], list[int]] | None: Input ids and targets, or None if the problem is too
            long or ends up with nothing to score.
    """
    text = prompt + answer
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    if len(token_ids) > max_length:
        return None

    answer_start = len(prompt)
    # targets[i] is what the model should predict *after* seeing token i, hence the shift by one.
    targets = [IGNORE_INDEX] * len(token_ids)
    for token_index, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
        if token_index > 0 and token_end > token_start and token_start >= answer_start:
            targets[token_index - 1] = token_ids[token_index]

    if all(target == IGNORE_INDEX for target in targets):
        return None
    return token_ids, targets


def _write(output_path: Path, samples: list, pad_token_id: int, metadata: dict, num_dropped: int) -> None:
    """Right-pads the samples and writes them, with metadata, to a compressed npz."""
    length = max(len(token_ids) for token_ids, _ in samples)
    # Right-padding is safe for a causal model: no scored position can attend to it, and the pad
    # positions are masked out of the targets so they contribute nothing to any metric.
    inputs = np.full((len(samples), length), pad_token_id, dtype=np.int64)
    targets = np.full((len(samples), length), IGNORE_INDEX, dtype=np.int64)
    for row, (token_ids, sample_targets) in enumerate(samples):
        inputs[row, : len(token_ids)] = token_ids
        targets[row, : len(sample_targets)] = sample_targets

    metadata = {
        **metadata,
        "num_problems": len(samples),
        "num_dropped_too_long": num_dropped,
        "sequence_length": int(length),
        "num_scored_tokens": int((targets != IGNORE_INDEX).sum()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, input_ids=inputs, target_ids=targets, metadata=json.dumps(metadata))
    print(
        f"{output_path.name:18s} problems={metadata['num_problems']:5d} (dropped {num_dropped:4d}) "
        f"len={length:5d} scored_tokens={metadata['num_scored_tokens']:7d} "
        f"({metadata['num_scored_tokens'] / len(samples):5.1f}/problem)"
    )


def prepare_minerva_math(tokenizer, output_directory: Path, max_length: int, tokenizer_path: str) -> None:
    """Writes the Minerva MATH evaluation, balanced over all seven subjects."""
    per_subject = -(-NUM_MATH_PROBLEMS // len(MATH_SUBJECTS))
    samples: list[tuple[list[int], list[int]]] = []
    num_dropped = 0
    for subject in MATH_SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for problem in dataset.select(range(min(per_subject, len(dataset)))):
            sample = _build_sample(
                tokenizer,
                prompt=_minerva_prompt(problem["problem"]),
                answer=" " + problem["solution"].strip(),
                max_length=max_length,
            )
            num_dropped += sample is None
            if sample is not None:
                samples.append(sample)

    _write(
        output_directory / "minerva_math.npz",
        samples[:NUM_MATH_PROBLEMS],
        tokenizer.eos_token_id,
        {
            "benchmark": "minerva_math",
            "tokenizer": tokenizer_path,
            "subjects": MATH_SUBJECTS,
            "num_fewshot": len(MINERVA_FEWSHOT_SAMPLES),
            "source": "lm-evaluation-harness lm_eval/tasks/minerva_math",
        },
        num_dropped,
    )


def prepare_triviaqa(tokenizer, output_directory: Path, max_length: int, tokenizer_path: str) -> None:
    """Writes the closed-book TriviaQA evaluation, the parametric-knowledge control."""
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    samples: list[tuple[list[int], list[int]]] = []
    num_dropped = 0
    for question in dataset.select(range(min(NUM_TRIVIAQA_QUESTIONS, len(dataset)))):
        # No context is supplied: the model must answer from its weights, which is the whole point
        # of using this as the parametric-knowledge control. Zero-shot, because the format is
        # trivial and is in any case constant across arms.
        sample = _build_sample(
            tokenizer,
            prompt=f"Question: {question['question']}\nAnswer:",
            answer=" " + question["answer"]["value"].strip(),
            max_length=max_length,
        )
        num_dropped += sample is None
        if sample is not None:
            samples.append(sample)

    _write(
        output_directory / "triviaqa.npz",
        samples,
        tokenizer.eos_token_id,
        {"benchmark": "triviaqa", "tokenizer": tokenizer_path, "split": "validation", "closed_book": True},
        num_dropped,
    )


def main() -> None:
    """Downloads, tokenizes and writes both prepared evaluations."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output_directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    # The model's sequence length. The 4-shot prefix alone is ~500 tokens, so a shorter cap would
    # drop a large fraction of MATH rather than a long tail; the count dropped is reported.
    parser.add_argument("--max_length", type=int, default=2048)
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.tokenizer_path))
    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required: the masking is built from the character-to-token offset "
            "mapping, which only a fast tokenizer provides."
        )

    prepare_minerva_math(tokenizer, arguments.output_directory, arguments.max_length, str(arguments.tokenizer_path))
    prepare_triviaqa(tokenizer, arguments.output_directory, arguments.max_length, str(arguments.tokenizer_path))


if __name__ == "__main__":
    main()
