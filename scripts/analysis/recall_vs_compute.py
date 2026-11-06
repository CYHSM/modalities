#!/usr/bin/env python
"""Deep share of numeric tokens that are computed vs. recalled (reviewer YXfZ, Q3a).

All groups are numeric tokens (digit strings), so token type and format are matched; they differ in
where the number comes from:
  gsm8k_computed     result of a calculation: after "=" (inside or outside <<...>>), after ">>",
                     or after "####"
  gsm8k_operand      operand inside <<...>> before "=": copied from earlier in the context
  gsm8k_question     numbers in the question, given in the context
  triviaqa_answer    numbers in a TriviaQA answer: recalled from parametric memory
  wikitext           numbers in WikiText-103 (mostly years and quantities; mixed recall)

Deep share rho_d (Eq. 6) per token and layer comes from the stored paloma_diagnostics parquets.
Reports per checkpoint: tokens per group, mean rho_d over all layers and over the late layers
(L10-15), and the per-layer profile.

Usage: recall_vs_compute.py --out results.json  (runs on the 18 Table 4 checkpoints)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import paloma_diagnostics as pdg

BASE = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp")
TOKENIZER = "/leonardo_work/AIFAC_S07_154/mfrey/tokenizer/local_gpt2_tokenizer/tokenizer.json"
CKPTS = {
    "80M_alpha25_K2": "2026-05-18__13-49-25_46661c3a0490b08d", "80M_alpha50_K2": "2026-05-13__02-05-13_fa51525556e2cd3b",
    "80M_alpha75_K2": "2026-05-18__19-28-55_0d6659323c9236d8", "80M_alpha25_K3": "2026-05-18__12-12-23_207cde6cd1273411",
    "80M_alpha50_K3": "2026-05-13__02-21-25_03070170b2aa73fa", "80M_alpha75_K3": "2026-05-18__19-28-58_635662eb51947f62",
    "80M_alpha25_K4": "2026-05-18__23-27-56_3af89f56dea89107", "80M_alpha50_K4": "2026-05-13__05-25-03_e3f7d3d2ad461681",
    "80M_alpha75_K4": "2026-05-18__19-28-56_159b57f1f3f20442", "160M_alpha25_K2": "2026-05-18__12-12-23_59f7fd9274a1eb2c",
    "160M_alpha50_K2": "2026-05-12__06-09-41_7254f5ea8908ea68", "160M_alpha75_K2": "2026-05-18__19-28-56_7d6f149ca64eea51",
    "160M_alpha25_K3": "2026-05-18__12-12-23_2b5348bcf89b949f", "160M_alpha50_K3": "2026-05-14__08-07-59_50122419e575e16d",
    "160M_alpha75_K3": "2026-05-18__19-28-59_b5263d4c7baf76af", "160M_alpha25_K4": "2026-05-18__12-12-23_355c95b231e4474c",
    "160M_alpha50_K4": "2026-05-12__06-22-54_3719e6b84f11c9c1", "160M_alpha75_K4": "2026-05-18__19-28-56_55916ccdd76d9adc",
}
GROUPS = ["gsm8k_computed", "gsm8k_operand", "gsm8k_question", "triviaqa_answer", "wikitext"]
DIGITS = re.compile(r"^\d[\d,.]*$")


def token_texts(tok: Tokenizer, ids) -> list[str]:
    return [tok.id_to_token(int(i)).replace("Ġ", " ").replace("Ċ", "\n") for i in ids]


def label_gsm8k(texts: list[str]) -> list[str | None]:
    """Group label per token of one GSM8K chunk (None for non-numeric or unclassified tokens).

    A number is "computed" when it follows "=", ">>" or "####" (allowing "$" and whitespace in
    between), an "operand" when it sits inside <<...>> before "=", and "question" before "Answer".
    A digit token that continues the previous digit token (GPT-2 splits long numbers) inherits its label.
    """
    out: list[str | None] = [None] * len(texts)
    in_answer = in_calc = pending = False
    for i, raw in enumerate(texts):
        t = raw.strip()
        if t.startswith("Answer"):
            in_answer = True
        if DIGITS.match(t):
            if i > 0 and not raw.startswith((" ", "\n")) and DIGITS.match(texts[i - 1].strip()):
                out[i] = out[i - 1]
            elif not in_answer:
                out[i] = "gsm8k_question"
            elif pending:
                out[i] = "gsm8k_computed"
            elif in_calc:
                out[i] = "gsm8k_operand"
            pending = False
            continue
        if "<<" in t:
            in_calc, pending = True, False
        if t.endswith("=") or t.endswith(">>") or t.startswith("####"):
            pending = True
            if t.endswith(">>"):
                in_calc = False
        elif t not in ("", "$"):
            pending = False
    return out


def label_simple(texts: list[str], group: str, answer_only: bool) -> list[str | None]:
    out: list[str | None] = [None] * len(texts)
    in_answer = not answer_only
    for i, raw in enumerate(texts):
        t = raw.strip()
        if t.startswith("Answer"):
            in_answer = True
        if in_answer and DIGITS.match(t):
            out[i] = group
    return out


def analyse(ckpt_dir: Path, tok: Tokenizer) -> dict:
    diag = ckpt_dir / "paloma_diagnostics"
    acc = {g: [] for g in GROUPS}   # list of (L,) deep-share vectors
    for source in ("gsm8k", "triviaqa", "wikitext_103"):
        rec = pdg.load_source(diag, source)
        rho = pdg.update_pref(rec)                      # (L, N)
        start = 0
        for ids, n in zip(rec["chunk_ids"], rec["chunk_lens"]):
            texts = token_texts(tok, ids)
            if source == "gsm8k":
                labels = label_gsm8k(texts)
            elif source == "triviaqa":
                labels = label_simple(texts, "triviaqa_answer", answer_only=True)
            else:
                labels = label_simple(texts, "wikitext", answer_only=False)
            for j, g in enumerate(labels):
                if g is not None:
                    acc[g].append(rho[:, start + j])
            start += int(n)
    out = {}
    for g, vecs in acc.items():
        if not vecs:
            out[g] = {"n": 0}
            continue
        m = np.nanmean(np.stack(vecs), axis=0)          # (L,)
        out[g] = {"n": len(vecs), "mean_all_layers": float(np.nanmean(m)),
                  "mean_late_L10_15": float(np.nanmean(m[10:])), "per_layer": [float(x) for x in m]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = Tokenizer.from_file(TOKENIZER)
    results = {}
    for name, d in CKPTS.items():
        ck = next(iter(sorted((BASE / d).glob("eid_*-seen_steps_18360-*/hf_checkpoint"))), None)
        if ck is None or not (ck / "paloma_diagnostics").exists():
            print(name, "missing diagnostics", flush=True)
            continue
        results[name] = analyse(ck, tok)
        r = results[name]
        print(name, " ".join(f"{g}={r[g].get('mean_late_L10_15', float('nan')):.3f}(n={r[g]['n']})" for g in GROUPS), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
