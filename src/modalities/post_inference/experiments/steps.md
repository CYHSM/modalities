CUDA_VISIBLE_DEVICES=5 python capture_math_nomath.py --model "google/gemma-2-9b-it" --max_new_tokens 1 --test welch --n_samples 2000 && CUDA_VISIBLE_DEVICES=5 python capture_math_nomath.py --model "google/gemma-2-9b" --max_new_tokens 1 --test welch --n_samples 2000

CUDA_VISIBLE_DEVICES=5 python visualize_math_nomath.py --experiment_path /raid/s3/opengptx/mfrey/activation/experiments/google_gemma-2-9b_math_vs_nonmath/ --downsample 256 --stats_only


Patch all Layers vs. Layer Specifics (lm_head, model.norm, ...)

CUDA_VISIBLE_DEVICES=6 python patch_by_layer.py --model "Qwen/Qwen3-8B" --h5_path /raid/s3/opengptx/mfrey/activation/experiments/Qwen_Qwen3-8B_correct_vs_incorrect/statistics.h5 --scale 1.0 --d_threshold 0.0 --std_threshold 0.0 --n_eval_samples 256 --group_by_layer