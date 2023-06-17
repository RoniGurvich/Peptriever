# Protein Language Model Pretraining

This module contains the code to train a tokenizer based on PDB sequences. In the future
it will also contain masked language modeling code

## Data Flow

1. Train a tokenizer using [build_tokenizer.py](./build_tokenizer.py)
2. Publish the tokenizer to huggingface hub
   using [publish_tokenizer.py](./publish_tokenizer.py)
3. Pretrain two BERT models using [mlm_pretraining.py](./mlm_pretraining.py)