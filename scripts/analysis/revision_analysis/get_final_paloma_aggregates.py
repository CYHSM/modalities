#!/usr/bin/env python3
import json
import re
from pathlib import Path

runs = {
    15: Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp/2026-07-13__09-51-14_d871934c61d04dc8"),
    16: Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions/2026-07-11__16-27-04_a562caf0ddd3d822"),
    17: Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions/2026-07-10__02-28-23_dc2828ca749af197")
}

def get_steps_from_dirname(name):
    m = re.search(r"seen_steps_(\d+)", name)
    return int(m.group(1)) if m else 0

def find_hf_folder(run_dir):
    eid_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("eid_")], key=lambda x: get_steps_from_dirname(x.name))
    if not eid_dirs:
        return None
    latest_eid = eid_dirs[-1]
    hf_path = latest_eid / "hf_checkpoint"
    return hf_path if hf_path.exists() else None

def read_metrics_from_file(folder, prefixes, key):
    if not folder or not folder.exists():
        return None
    for p in folder.glob("*.json"):
        for prefix in prefixes:
            if p.name.startswith(prefix) and p.name.endswith("-metrics.json") and "verbose" not in p.name:
                try:
                    with open(p) as f:
                        data = json.load(f)
                        if "metrics" in data:
                            return float(data["metrics"][key])
                except Exception:
                    pass
    return None

print("| Config Index | Paloma C4 | Paloma WikiText-103 | 15-Task BPB (no Paloma) | 17-Task BPB (Figure 1) |")
print("|---|---|---|---|---|")

for idx, run_path in runs.items():
    hf_folder = find_hf_folder(run_path)
    if not hf_folder:
        print(f"| **[{idx}]** | *No checkpoint found* | | | |")
        continue

    # Main olmes evaluation folder
    main_eval = hf_folder / "olmes_eval_18360"
    
    # Paloma evaluation folders (Config 15 had them in main_eval, 16 & 17 have separate folders)
    if idx == 15:
        c4_folder = main_eval
        wiki_folder = main_eval
    else:
        c4_folder = hf_folder / "olmes_eval_18360_paloma_c4"
        wiki_folder = hf_folder / "olmes_eval_18360_paloma_wiki"
        
    c4_bpb = read_metrics_from_file(c4_folder, ["task-000-paloma_c4_en"], "bits_per_byte")
    wiki_bpb = read_metrics_from_file(wiki_folder, ["task-001-paloma_wikitext_103"], "bits_per_byte")
    
    # 6 Commonsense BPBs
    arcc_bpb = read_metrics_from_file(main_eval, ["task-007-arc_challenge", "task-015-arc_challenge"], "bits_per_byte_corr")
    arce_bpb = read_metrics_from_file(main_eval, ["task-008-arc_easy", "task-016-arc_easy"], "bits_per_byte_corr")
    hella_bpb = read_metrics_from_file(main_eval, ["task-009-hellaswag", "task-017-hellaswag"], "bits_per_byte_corr")
    piqa_bpb = read_metrics_from_file(main_eval, ["task-012-piqa", "task-020-piqa"], "bits_per_byte_corr")
    siqa_bpb = read_metrics_from_file(main_eval, ["task-011-socialiqa", "task-019-socialiqa"], "bits_per_byte_corr")
    wino_bpb = read_metrics_from_file(main_eval, ["task-010-winogrande", "task-018-winogrande"], "bits_per_byte_corr")
    cs_bpbs = [arcc_bpb, arce_bpb, hella_bpb, piqa_bpb, siqa_bpb, wino_bpb]
    
    # 7 Math BPBs
    math_subtasks = [
        "task-000-minerva_math_algebra",
        "task-001-minerva_math_counting_and_probability",
        "task-002-minerva_math_geometry",
        "task-003-minerva_math_intermediate_algebra",
        "task-004-minerva_math_number_theory",
        "task-005-minerva_math_prealgebra",
        "task-006-minerva_math_precalculus"
    ]
    math_bpbs = [read_metrics_from_file(main_eval, [sub], "bits_per_byte_corr") for sub in math_subtasks]
    
    # 2 Reading / QA BPBs
    lambada_bpb = read_metrics_from_file(main_eval, ["task-014-lambada", "task-022-lambada"], "bits_per_byte_corr")
    qasper_bpb = read_metrics_from_file(main_eval, ["task-013-qasper_yesno", "task-021-qasper_yesno"], "bits_per_byte_corr")
    qa_bpbs = [lambada_bpb, qasper_bpb]
    
    # Aggregation
    non_paloma_bpbs = cs_bpbs + math_bpbs + qa_bpbs
    mean_non_paloma = sum(non_paloma_bpbs) / len(non_paloma_bpbs) if all(x is not None for x in non_paloma_bpbs) else None
    
    all_bpbs = non_paloma_bpbs + [c4_bpb, wiki_bpb]
    mean_all = sum(all_bpbs) / len(all_bpbs) if all(x is not None for x in all_bpbs) else None

    c4_str = f"{c4_bpb:.4f}" if c4_bpb is not None else "Pending..."
    wiki_str = f"{wiki_bpb:.4f}" if wiki_bpb is not None else "Pending..."
    non_pal_str = f"{mean_non_paloma:.4f}" if mean_non_paloma is not None else "N/A"
    all_str = f"{mean_all:.4f}" if mean_all is not None else "Pending..."
    
    print(f"| **[{idx}]** | {c4_str} | {wiki_str} | {non_pal_str} | **{all_str}** |")
