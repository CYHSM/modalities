import os
from pathlib import Path

eval_dir = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp/2026-07-13__09-51-14_d871934c61d04dc8/eid_2026-07-13__09-51-14_d871934c61d04dc8-seen_steps_18360-seen_tokens_38503710720-target_steps_18360-target_tokens_38503710720/hf_checkpoint/olmes_eval_18360")
out_file = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/emnlp/sbatch_revisions/dir_contents.txt")

lines = []
if eval_dir.exists():
    for item in sorted(os.listdir(eval_dir)):
        p = eval_dir / item
        size = p.stat().st_size
        lines.append(f"{item} ({size} bytes)")
else:
    lines.append(f"Directory {eval_dir} does not exist")

with open(out_file, "w") as f:
    f.write("\n".join(lines))
print("Done listing")
