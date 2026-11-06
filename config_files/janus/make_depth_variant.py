#!/usr/bin/env python3
"""Derive an L<N> janus config from the corresponding L32 config.

F80M is a per-layer FLOP budget, so the FFN dimensions do not change with depth
-- only `n_layer`, the `layer_types` list, and the experiment id do. Generating
the deeper variants mechanically (rather than by hand) keeps the dual arm and
its pure-wide control identical in every respect except the model.

The deeper models do not fit at micro=2, so batch settings are overridden here;
both arms get the SAME override, since the comparison requires identical batch
and gradient-accumulation.

Usage: make_depth_variant.py <in_L32.yaml> <out.yaml> <n_layer> <micro> <grad_acc>
"""
import re, sys
from pathlib import Path

src, dst, n_layer, micro, grad_acc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
s = Path(src).read_text()

# n_layer
s, n = re.subn(r"^    n_layer: 32$", f"    n_layer: {n_layer}", s, flags=re.M)
assert n == 1, f"n_layer: expected 1 substitution, got {n}"

# layer_types -- preserve whichever type the template uses, just change the count
def repl_types(m):
    kind = m.group(1)
    return '      layer_types: [' + ", ".join([f'"{kind}"'] * n_layer) + ']'
s, n = re.subn(r'^      layer_types: \["(\w+)"(?:, "\w+")*\]$', repl_types, s, flags=re.M)
assert n == 1, f"layer_types: expected 1 substitution, got {n}"

# batch settings (identical for both arms)
s, n = re.subn(r"^    gradient_accumulation_steps: \d+$", f"    gradient_accumulation_steps: {grad_acc}", s, flags=re.M)
assert n == 1, f"grad_acc: expected 1 substitution, got {n}"
s, n = re.subn(r"^    local_train_micro_batch_size: \d+$", f"    local_train_micro_batch_size: {micro}", s, flags=re.M)
assert n == 1, f"micro: expected 1 substitution, got {n}"

# experiment id
s, n = re.subn(r"^    experiment_id: dm768_L32_", f"    experiment_id: dm768_L{n_layer}_", s, flags=re.M)
assert n == 1, f"experiment_id: expected 1 substitution, got {n}"

Path(dst).write_text(s)
print(f"wrote {dst}  (n_layer={n_layer}, micro={micro}, grad_acc={grad_acc})")
