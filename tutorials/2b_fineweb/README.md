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

Was a bit unclear whats the most up-to date config file, so I ran into a bunch of key errors that were missing from older configs I used. 