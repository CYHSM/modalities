#!/usr/bin/env python
"""
POS-tag the tokens in existing paloma_diagnostics parquets and write sidecar files.

Usage:
    python paloma_pos_tag.py \
        --in-dir /path/to/paloma_diagnostics \
        --ckpt /path/to/hf_checkpoint   # needed for the tokenizer

For each paloma_<source>.parquet in <in-dir>, writes a sidecar:

    paloma_<source>_pos.parquet

containing one row per (doc, chunk) with a `pos` field — a list[str] of UPOS
tags per subword token (or "SPECIAL" / "UNKNOWN").

----------------------------------------------------------------------------
Why this script was rewritten
----------------------------------------------------------------------------
The previous version did this:

    text = tokenizer.decode(input_ids)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    if enc["input_ids"] != input_ids:
        return ["UNKNOWN"] * len(input_ids)

This fallback was triggered for *every* chunk in practice. Reasons:
  - The recorded input_ids may include a BOS that re-tokenizing without
    special tokens doesn't reproduce.
  - BPE/SentencePiece tokenizers rarely round-trip via decode->encode
    because of whitespace normalization, byte-fallback, etc.

The new approach is alignment-by-construction: we decode each token *one
at a time*, concatenate the surface strings to build the text, and record
the (char_start, char_end) of each token as we go. Then we POS-tag the
concatenated text and assign each token the tag of whichever word contains
(or maximally overlaps) its char span.

This is robust to any tokenizer because we never re-tokenize anything.
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Universal POS tagset (UD v2). Anything outside this set should never end
# up in the sidecar — it'd mean the spaCy pipeline is misconfigured.
VALID_UPOS = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
})


def decode_with_offsets(input_ids, tokenizer):
    """Decode each token individually and accumulate (text, offsets).

    Returns:
        text     : str       - concatenation of all token surface strings
        offsets  : list[(int, int)] - per-token (start_char, end_char) into `text`

    Special tokens (BOS/EOS/PAD/etc.) decode to "" with this approach (since
    `skip_special_tokens=True` removes them). We mark those positions with
    (cur, cur) — a zero-width span — and they get tagged "SPECIAL" downstream.

    Note on leading spaces: BPE tokenizers typically emit tokens like " the"
    (with a leading space) for mid-sentence words. We keep the space — it's
    part of the surface text — and the word's char span naturally falls
    inside the (start, end) we record. spaCy ignores leading whitespace
    when assigning POS, so alignment still works cleanly.
    """
    text_parts = []
    offsets = []
    cur = 0
    special_ids = set(tokenizer.all_special_ids or [])
    for tid in input_ids:
        if tid in special_ids:
            offsets.append((cur, cur))
            continue
        # Decode just this one token. clean_up_tokenization_spaces=False
        # keeps the raw surface form so offsets remain exact.
        piece = tokenizer.decode([tid], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        # If the tokenizer still returns a special-token marker (e.g. for
        # tokens not in all_special_ids but rendered as <|...|>), treat as
        # zero-width too. Detect heuristically: decoded piece is empty.
        if piece == "":
            offsets.append((cur, cur))
            continue
        start = cur
        end = cur + len(piece)
        text_parts.append(piece)
        offsets.append((start, end))
        cur = end
    return "".join(text_parts), offsets


def align_tokens_to_pos(input_ids, tokenizer, nlp):
    """Return a UPOS tag for every token in input_ids.

    Uses per-token decoding (no re-tokenization) for robust alignment.
    """
    text, offsets = decode_with_offsets(input_ids, tokenizer)

    if not text.strip():
        # All-special or empty chunk. Mark accordingly.
        return ["SPECIAL" if s == e else "UNKNOWN" for (s, e) in offsets]

    # spaCy gives us word-level (char_start, char_end, upos). Skip tokens
    # whose pos_ is empty — that should never happen now that we keep
    # attribute_ruler, but defense in depth so a misconfigured pipeline
    # can't silently produce empty-string tags in the sidecar again.
    doc = nlp(text)
    word_spans = [(t.idx, t.idx + len(t), t.pos_ or "UNKNOWN") for t in doc]

    out = []
    sp_idx = 0  # cursor through word_spans (monotonic with token order)
    for start, end in offsets:
        if start == end:
            # Zero-width => was a special token.
            out.append("SPECIAL")
            continue

        # Advance cursor past spans that end at or before our start.
        while sp_idx < len(word_spans) and word_spans[sp_idx][1] <= start:
            sp_idx += 1

        # Scan for best overlap with a word span. With BPE leading-space tokens,
        # the token span "[space]the" may extend slightly before the word's
        # char_start, so we accept any overlap, not just containment.
        tag = "UNKNOWN"
        best_overlap = 0
        scan = sp_idx
        while scan < len(word_spans) and word_spans[scan][0] < end:
            w_start, w_end, pos = word_spans[scan]
            overlap = max(0, min(end, w_end) - max(start, w_start))
            if overlap > best_overlap:
                best_overlap = overlap
                tag = pos
            scan += 1
        out.append(tag)
    return out


def process_source(parquet_path: Path, out_path: Path, tokenizer, nlp,
                   batch_size: int = 32, max_chunks: int = 0):
    """Process one source parquet → sidecar."""
    print(f"\n[{parquet_path.name}] reading...", flush=True)
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    if max_chunks > 0 and n_rows > max_chunks:
        print(f"[{parquet_path.name}] capping {n_rows} chunks at {max_chunks}", flush=True)
        n_rows = max_chunks
    print(f"[{parquet_path.name}] {n_rows} chunks", flush=True)

    rows = table.to_pylist()[:n_rows]
    writer = None
    pending = []
    total_unknown = 0
    total_special = 0
    total_real = 0   # tokens with a real UPOS tag
    total_tokens = 0

    def flush_pending():
        nonlocal pending, writer
        if not pending:
            return
        t = pa.Table.from_pylist(pending)
        if writer is None:
            writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
        writer.write_table(t)
        pending = []

    try:
        for i, row in enumerate(rows):
            ids = row["tokens"]
            try:
                pos = align_tokens_to_pos(ids, tokenizer, nlp)
            except Exception as e:
                # If anything goes wrong on a single chunk, mark and move on
                # rather than killing the whole source.
                print(f"  chunk {i} (doc={row['doc']}): error during alignment: {e}",
                      flush=True)
                pos = ["UNKNOWN"] * len(ids)

            # Coverage counts only tokens with a real UPOS tag. The old
            # check (p == "UNKNOWN" or p == "SPECIAL") allowed empty strings
            # to slip through as "covered" — that's how the bug hid before.
            # Here we explicitly require membership in the known UPOS set.
            real_pos = sum(1 for p in pos if p in VALID_UPOS)
            total_real += real_pos
            total_unknown += sum(1 for p in pos if p == "UNKNOWN")
            total_special += sum(1 for p in pos if p == "SPECIAL")
            total_tokens += len(pos)

            pending.append({
                "doc": int(row["doc"]),
                "chunk": int(row["chunk"]),
                "pos": pos,
            })
            if len(pending) >= batch_size:
                flush_pending()
            if (i + 1) % 20 == 0:
                # Coverage = fraction of tokens that got a real UPOS tag.
                cov = 100 * total_real / max(total_tokens, 1)
                other = total_tokens - total_real - total_unknown - total_special
                extra = f", other={other}" if other else ""
                print(f"  {i+1}/{n_rows} chunks, coverage {cov:.1f}% "
                      f"(real={total_real}, unknown={total_unknown}, "
                      f"special={total_special}{extra})", flush=True)
        flush_pending()
    finally:
        if writer is not None:
            writer.close()

    cov = 100 * total_real / max(total_tokens, 1)
    print(f"[{parquet_path.name}] DONE — coverage {cov:.1f}% "
          f"({total_real}/{total_tokens} tokens with real POS tags; "
          f"{total_unknown} UNKNOWN, {total_special} SPECIAL)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Directory containing paloma_<source>.parquet files")
    ap.add_argument("--ckpt", required=True,
                    help="HF checkpoint for the tokenizer")
    ap.add_argument("--spacy-model", default="en_core_web_sm",
                    help="spaCy model name")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Optional list of sources to process (default: all)")
    ap.add_argument("--max-chunks-per-source", type=int, default=200,
                    help="Cap chunks tagged per source (0 = no cap)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True, use_fast=True)
    print(f"  special_ids: {sorted(set(tokenizer.all_special_ids or []))}")

    print(f"Loading spaCy model {args.spacy_model}...")
    import spacy
    # IMPORTANT: do NOT disable attribute_ruler. In modern spaCy pipelines
    # (v3+), attribute_ruler is what maps fine-grained tag_ -> coarse pos_
    # (UPOS). Disabling it leaves token.tag_ populated but token.pos_ as
    # the empty string, which propagates into the sidecar as "" everywhere.
    # That's the bug that gave us "99.6% coverage" yet all-UNKNOWN bars.
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = 5_000_000
    # Sanity check: run on a known sentence and verify pos_ is populated.
    test_doc = nlp("The cat sat on the mat.")
    test_tags = [t.pos_ for t in test_doc]
    print(f"  spaCy POS sanity check: 'The cat sat on the mat.' -> {test_tags}")
    if not any(test_tags) or all(t == "" for t in test_tags):
        raise SystemExit(
            "spaCy POS tagger returned only empty strings. The pipeline is "
            "misconfigured. Make sure 'tagger' and 'attribute_ruler' "
            "components are enabled."
        )

    parquets = sorted(in_dir.glob("paloma_*.parquet"))
    parquets = [p for p in parquets if not p.stem.endswith("_pos")]

    if args.sources:
        wanted = set(args.sources)
        parquets = [p for p in parquets if p.stem.replace("paloma_", "") in wanted]

    if not parquets:
        raise SystemExit(f"No paloma_*.parquet files found in {in_dir}")

    for p in parquets:
        out_path = p.with_name(p.stem + "_pos.parquet")
        if out_path.exists():
            print(f"[{p.name}] sidecar exists, skipping")
            continue
        process_source(p, out_path, tokenizer, nlp,
                       max_chunks=args.max_chunks_per_source)

    print("\nAll done.")


if __name__ == "__main__":
    main()