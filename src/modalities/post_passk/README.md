## Run pass@k analysis
Run with: 

```
CUDA_VISIBLE_DEVICES=5 python evaluate_passk.py --model_name Qwen/Qwen2.5-7B --n_samples 128 --subset_size 256 --n_fewshots 8 --temperatures 0.3 0.6 1.0 --output_dir /change/this/results
```

Then plot results with: 
```
python plot_results.py --results_dir /change/this/results/Qwen2.5_7B_8Shot_passk_n128_subset256 --k_values 1 2 4 8 16 32 64 128
```

## Notes:
- This implements the Low-Variance pass@k Estimation from the Yue et al. paper (https://arxiv.org/abs/2504.13837) so might be interesting to compare to actual pass@k
- I changed the way the answer is extracted for gsm8k to use the first boxed answer as Teuken was often continuing generation with a new problem and a new answer. This seems to also improve the baseline from Qwen2.5. 