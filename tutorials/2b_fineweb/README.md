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

## Warmstart

So now we want to run the checkpoint with a new learning cycle from the checkpoints, for this we use a new config (2b_config_warmstart) and run the model using: 
```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv-endpoint localhost:29515 --nnodes 1 --nproc_per_node 4 $(which modalities) run --config_file_path configs/2b_config_warmstart.yaml
```
Lots of struggles with setting the correct number of tokens, for this config I have:
4 (GPUs) × 3 (local_train_micro_batch_size) × 2048 (sequence_length) × 5 (gradient_accumulation_steps) = 122,880 tokens/step

I had it start with the OneCycle LR again but it really seems that is not the correct way of doing this as the loss just shot up and now slowly comes down. We dont have enough GPUs to run stuff so I will cancel this run for now. 


## Hyperparameters / Extensions
- Is Grouped Query Attention not used??