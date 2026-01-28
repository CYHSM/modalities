import wandb
from collections import Counter

ENTITY = "cyhsm"
PROJECT = "loop"
RUN_ID = "2u49bm6x"  # Original run

api = wandb.Api(timeout=60)
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

print("=== Analyzing scan_history() step distribution ===\n")

# Get all rows
print("Fetching scan_history()...")
rows = list(run.scan_history())
print(f"Total rows returned: {len(rows)}")

# Count steps
steps = [r.get('_step') for r in rows if r.get('_step') is not None]
unique_steps = sorted(set(steps))
print(f"Unique _step values: {len(unique_steps)}")
print(f"Step range: {min(unique_steps)} to {max(unique_steps)}")

# Check step gaps
step_gaps = []
for i in range(1, len(unique_steps)):
    gap = unique_steps[i] - unique_steps[i-1]
    if gap > 1:
        step_gaps.append((unique_steps[i-1], unique_steps[i], gap))

print(f"\nStep gaps > 1: {len(step_gaps)}")
if step_gaps[:10]:
    print("First 10 gaps:")
    for start, end, gap in step_gaps[:10]:
        print(f"  {start} -> {end} (gap of {gap})")

# Check if CE loss is in all rows
ce_key = "train train/loss_avg"
ce_alt_key = "train train/ce_loss_avg"

rows_with_ce = 0
for row in rows:
    if ce_key in row or ce_alt_key in row:
        val = row.get(ce_key) or row.get(ce_alt_key)
        if val is not None:
            rows_with_ce += 1

print(f"\n=== CE Loss presence ===")
print(f"Rows with CE loss: {rows_with_ce} / {len(rows)}")

# Sample some rows to see what keys they have
print(f"\n=== Sample row analysis ===")
sample_steps = unique_steps[:5] + unique_steps[len(unique_steps)//2:len(unique_steps)//2+5]
for step in sample_steps:
    matching = [r for r in rows if r.get('_step') == step]
    if matching:
        row = matching[0]
        has_ce = ce_key in row or ce_alt_key in row
        num_keys = len([k for k in row.keys() if not k.startswith('_')])
        print(f"Step {step}: {num_keys} metrics, has CE loss: {has_ce}")

# Check if multiple rows per step
print(f"\n=== Rows per step ===")
step_counts = Counter(steps)
multi_row_steps = [(s, c) for s, c in step_counts.items() if c > 1]
print(f"Steps with multiple rows: {len(multi_row_steps)}")
if multi_row_steps[:5]:
    print("Examples:", multi_row_steps[:5])

# Compare with history() DataFrame
print(f"\n=== Compare with history() DataFrame ===")
df = run.history(samples=100000, pandas=True)
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame unique _step: {df['_step'].nunique()}")

# Check CE loss in DataFrame
if ce_key in df.columns:
    non_null = df[df[ce_key].notna()]
    print(f"Rows with non-null '{ce_key}': {len(non_null)}")
    print(f"At steps: {sorted(non_null['_step'].unique())[:20]}...")
elif ce_alt_key in df.columns:
    non_null = df[df[ce_alt_key].notna()]
    print(f"Rows with non-null '{ce_alt_key}': {len(non_null)}")
    print(f"At steps: {sorted(non_null['_step'].unique())[:20]}...")