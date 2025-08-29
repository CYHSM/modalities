# Install problems
Just slight problems with flash attention, as the first time around it did not really work. After reinstall worked perfectly. Tried the new version >2.7.4 but got an error

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --seed --python 3.11 --prompt modalities
source .venv/bin/activate
uv pip install torch
uv pip install ninja
uv pip install --no-build-isolation flash-attn==2.7.4.post1
# for developer: use [tests,linting] and install pre-commit hooks
uv pip install -e .[tests,linting]
pre-commit install --install-hooks
```

Tried running a full command but errored: 
CUDA_VISIBLE_DEVICES=4 torchrun --rdzv-endpoint localhost:29515 --nnodes 1 --nproc_per_node 1 $(which odalities) run --config_file_path config_files/training/config_lorem_ipsum_long_fsdp2.yaml

# FineWeb

For now just used the sample from FineWeb Edu

```sh
wget https://huggingface.co/datasets/ModalitiesTeam/FW_EDU_SUBSET_500k_docs/resolve/main/fineweb_edu_num_docs_483606.jsonl?download=true -O fineweb_edu_num_docs_483606.jsonl
# Inspect data
head -n 3 fineweb_edu_num_docs_483606.jsonl | jq '.'
# Generate index
modalities data create_raw_index --index_path /home/markus_frey/Github/modalities/tutorials/modalities_in_15_mins/data/preprocessed/fineweb_edu_num_docs_483606.idx /raid/s3/opengptx/mfrey/fineweb-edu-subset/fineweb_edu_num_docs_483606.jsonl
```

# Tokenizer
Downloaded from https://huggingface.co/Eurolingua/Modalities_8B_3.73T/blob/main/tokenizer.model and tested with test_tokenizer.py in tutorials/2b_fineweb/tokenizer. Observations:
- "Artificial" and "Intelligence" get their own tokens, so its likely trained on a lot of ML docs?
- the last 20 tokens in the vocabulary are just spaces:
    - ID 250860: '▁▁▁▁▁'
    - ID 250861: '▁▁▁▁▁▁'
    - ID 250862: '▁▁▁▁▁▁▁'
    - ID 250879: '▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁'

To pack it we can run:

```sh
modalities data pack_encoded_data configs/tokenization_config.yaml
```
There were some hiccups as I used the config from the tutorial as a baseline, which needs some adjustment. In particular we need to use the SentencePiece tokenizer and change the eod token.

# Training

Was a bit unclear whats the most up-to date config file, so I ran into a bunch of key errors that were missing from older configs I used. In the end I could start training with:
```sh
CUDA_VISIBLE_DEVICES=3,4,6 torchrun --rdzv-endpoint localhost:29515 --nnodes 1 --nproc_per_node 3 $(which modalities) run --config_file_path configs/2b_config.yaml
```
but only with duplicating the validation set. So now have to actually get a sensible dataset and run the above again. 
For this we use load_fineweb2 which loads the languages which we specified ["eng_Latn", "deu_Latn", "fra_Latn", "ita_Latn", "spa_Latn", "nob_Latn"] and tries to get roughly 5B tokens from each of them (likely more as I use a conservative estimate of tokens per document)

So then we need to create our indices again: 
```sh
modalities data create_raw_index --index_path /home/markus_frey/Github/modalities/tutorials/2b_fineweb/data/preprocessed/eng_Latn_train.idx /raid/s3/opengptx/mfrey/fineweb-30B/eng_Latn_train.jsonl
modalities data create_raw_index --index_path /home/markus_frey/Github/modalities/tutorials/2b_fineweb/data/preprocessed/eng_Latn_val.idx /raid/s3/opengptx/mfrey/fineweb-30B/eng_Latn_val.jsonl
```
or see create_indices.sh

## Evaluation

So the first model is running (https://wandb.ai/cyhsm/2b_config/runs/k564d3jf) and I can evaluate one of the checkpoints. So do to this I created the 2b_generation_config file which then should be able to run with: 
```sh
modalities generate_text --config_file_path configs/2b_generation_config.yaml
```
I get the error *settings.step_profile.sequence_length' not found* so I assume there has been some update to the codebase:
- *'loss_fn.config.prediction_key' not found*
- I think it makes sense to have the same yaml file for generation with maybe one additional passer (the checkpoint path)
- *Input should be an instance of TokenizerWrapper [type=is_instance_of, input_value=None, input_type=NoneType]*

Okay after putting the tokenizer directly inside the text_inference_component it works.

## Evaluation on LightEval

Now how do I evaluate it on LightEval. Probably best to do a conversion to Huggingface format:
```sh
python -m modalities convert_pytorch_to_hf_checkpoint \
    --config_file_path /home/markus_frey/Github/modalities/tutorials/2b_fineweb/configs/2b_config.yaml \
    --output_hf_checkpoint_dir /raid/s3/opengptx/mfrey/fineweb-30B/checkpoints/2025-08-07__17-50-58_89893f7f/hf \
    --prediction_key "logits"
```
Okay this fails, and Im currently checking with Timm and Alex whats the best way of doing this. Definitely something where we need to update the docs. so I tried with 
```sh
python convert_gpt2.py --num_testruns 5 --device_modalities cuda:1 --device_hf cuda:2 /home/markus_frey/Github/modalities/tutorials/2b_fineweb/configs/2b_config_for_conversion.yaml /raid/s3/opengptx/mfrey/fineweb-30B/checkpoints/2025-08-07__17-50-58_89893f7f/hf
```
but it also failed, so I guess I have to change the branch? 
Okay we needed to change the key "pytorch_implementation" from "manual" to "pytorch_flash" to make sure the logits of the hf model are the same as from the modalities model. Now it runs through and we can maybe run lighteval.

```sh
accelerate launch --num_processes=1 --gpu_ids="1" -m lighteval.accelerate_main \
    --model_name_or_path /raid/s3/opengptx/mfrey/fineweb-30B/checkpoints/2025-08-07__17-50-58_89893f7f/hf \
    --tasks leaderboard|truthfulqa:mc|0|0
```
Okay coming back to this later.

Trying lighteval on these tasks:
lighteval|arc:easy|3|1,leaderboard|hellaswag|3|1,helm|mmlu|3|1,leaderboard|gsm8k|3|1,leaderboard|truthfulqa:mc|3|1

export HF_HOME="/raid/s3/opengptx/mfrey/huggingface"
CUDA_VISIBLE_DEVICES=6 accelerate launch -m lighteval accelerate     "model_name=/raid/s3/opengptx/mfrey/3.73T-Tokens/checkpoints,use_chat_template=False,trust_remote_code=True"     "lighteval|arc:easy|3|1"     --max-samples 100

Command: RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
Error: CUDA_VISIBLE_DEVICES=6 lighteval accelerate     "model_name=/raid/s3/opengptx/mfrey/3.73T-Tokens/checkpoints,use_chat_template=False,trust_remote_code=True,batch_size=1"     "lighteval|arc:easy|3|1"

Okay fixing conversion with a new convert_tokenizer.py file fixes this! 

## Warmstart

So now we want to run the checkpoint with a new learning cycle from the checkpoints, for this we use a new config (2b_config_warmstart) and run the model using: 
```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv-endpoint localhost:29515 --nnodes 1 --nproc_per_node 4 $(which modalities) run --config_file_path configs/2b_config_warmstart.yaml
```
Lots of struggles with setting the correct number of tokens, for this config I have:
4 (GPUs) × 3 (local_train_micro_batch_size) × 2048 (sequence_length) × 5 (gradient_accumulation_steps) = 122,880 tokens/step

I had it start with the OneCycle LR again but it really seems that is not the correct way of doing this as the loss just shot up and now slowly comes down. We dont have enough GPUs to run stuff so I will cancel this run for now. I am now running with a very small max_lr and it seems to continue learning so thats good! 


## Hyperparameters / Extensions
- Is Grouped Query Attention not used??

## New Run
Okay fixed some issues with the first run and started a new one, this time fully using the RAM of the GPU. I scaled up to 8 micro_batches with 4 grad accumulation steps and am using a fixed lr now 3.14e-4. Maybe its too big. The model uses 23 Gb during inference and 79Gb during training so barely fits. Thats nearly a 3.5x increase. 

Theres a tool that allows you to calculate the perfect optimizer settings to reach a "global" local minimum: https://step-law.github.io/#steplawtool
For our model with 1.974 billion parameters and trained on 30B tokens, its set to these values: 
- max_lr: 6.95e-4
- div_factor: 100 
- final_div_factor: 0.695
- pct_start: 0.037
Note that the final learning rate is calculate like this: min_lr = initial_lr/final_div_factor so the final_div_factor needs to be calculated with respect to the inital_lr and not the max_lr! 

## Datasets
There are so many datasets currently out there, its probably best to start with https://huggingface.co/datasets/allenai/olmo-mix-1124
For a full dataset which can be used for finetuning (or pre-training) see here: https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1

## Finetuning
I started to use TRL to finetune the model but cant set the temperature yet. Doesnt matter, most benchmarks are evaluated at temp 0 anyway. So now I just want to mostly focus on GSM8k so I think I will use
Standard practice would be sth like: lighteval|arc:easy|25|1,leaderboard|hellaswag|10|1,helm|mmlu|5|1,leaderboard|gsm8k|8|1,leaderboard|truthfulqa:mc|0|1

I wonder how to improve GSM8K performance. Like the model only sees math responses and trains on them and in a way it should learn some math in that regard, but I really wonder how I need to set my learning rate, because the loss might first go down and it kinda learns the overall way math questions are answered but does not really get the specific numbers, or the actual math. MathInstruct2 paper uses a learning rate of 2e-5 with a batch size of 256. 

CUDA_VISIBLE_DEVICES=6,7 python main.py --model-path "/raid/s3/opengptx/mfrey/instruct/checkpoints/checkpoint-70000" --output-dir "/raid/s3/opengptx/mfrey/instruct/checkpoints/grpo" --dataset "nvidia/OpenMathInstruct-2" --dataset-split "train_1M" --training-method grpo --logging-steps 1 --eval-steps 200 --save-steps 200 --num-generations 4 --reward-correct 1.0 --reward-incorrect -0.1 --loss-type dr_grpo --learning-rate 1e-6 --eval-gpu 5 --batch-size 1 --max-completion-length 256 --verbose-rewards --reward-log-freq 1


So the model with 3e-5 and a batch size of 4 (with packing & liger) outperforms all others (gsm8k around 0.4) while still being at 0.6 for hellaswag, while a model using a larger lr (1e-4) with 32 batches also gets to around 0.4 gsm8k but drops in hellaswag (catastrophic forgetting).

Full run with the original 3.73T model: 
|                     Task                      |Version|    Metric    |Value |   |Stderr|
|-----------------------------------------------|------:|--------------|-----:|---|-----:|
|all                                            |       |em            |0.5561|±  |0.0357|
|                                               |       |qem           |0.5487|±  |0.0353|
|                                               |       |pem           |0.5564|±  |0.0357|
|                                               |       |pqem          |0.6700|±  |0.0342|
|                                               |       |acc           |0.5650|±  |0.0221|
|                                               |       |acc_norm      |0.6690|±  |0.0202|
|                                               |       |truthfulqa_mc1|0.2960|±  |0.0204|
|                                               |       |truthfulqa_mc2|0.4542|±  |0.0184|

With the finetuned model: 
|                     Task                      |Version|    Metric    |Value |   |Stderr|
|-----------------------------------------------|------:|--------------|-----:|---|-----:|
|all                                            |       |em            |0.5457|±  |0.0358|
|                                               |       |qem           |0.5423|±  |0.0356|
|                                               |       |pem           |0.5462|±  |0.0358|
|                                               |       |pqem          |0.6480|±  |0.0345|
|                                               |       |acc           |0.5520|±  |0.0221|
|                                               |       |acc_norm      |0.6770|±  |0.0200|
|                                               |       |truthfulqa_mc1|0.3100|±  |0.0207|
|                                               |       |truthfulqa_mc2|0.4511|±  |0.0187|

Okay so lots of finetuning behind me. It seems that openmathinstruct2 is the best dataset to easily push gsm8k scores, BUT only up to 0.4. Then it looks like theres additional reasoning needed to actually solve these tasks, which our 8B base model is not capable off. I tried using Tülu3 dataset which contains also general reasoning but it just dropped the scores dramatically. I also analysed which layers change the most between base model and finetuned model and it seems it ramps up to layer 20 (which has the largest changes) and then comes down slightly. 

Serving the model with vllm works with:  
CUDA_VISIBLE_DEVICES=6 trl vllm-serve --model /raid/s3/opengptx/mfrey/instruct/checkpoints/checkpoint-70000 --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --trust-remote-code --vllm-model-impl transformers

GRPO is a bit like magic when it works but mostly it doesnt. The things I play around with:
- The instruction. How much CoT do you put in there. I now tell it in the beginning and end to reason and put the result in boxed
- 

Note there is a weird error that happens from time to time if per_device_eval_batch_size=1 is < num_generations,

## GRPO
GRPO is a method from RL which generates several completions which are then scored by a reward model (or function) and normalized. These normalized scores are then used to update the model in order for completions which have a higher score to be sampled more often. Now it was very tough to make GRPO work well, so heres some anecdotes:
- The prompt has to match the SFT data, so its easier if you train to put the answer in \boxed if your SFT data also put the answer there.
- With gradient checkpointing the model fits easily into memory so you can train with a batch size of 16 and therefore number of generations of 16! 