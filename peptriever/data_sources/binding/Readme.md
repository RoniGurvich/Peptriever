# Protein Binding Data Source

This module downloads protein binding data
from [Huang Laboratory](http://huanglab.phys.hust.edu.cn)
and [Propedia](http://bioinfo.dcc.ufmg.br/propedia/) and combines it to one binding
training set minimizing data leakage by using the genes associated with each chain.

The resulting training set is published
to [huggingface hub](https://huggingface.co/datasets/ronig/protein_binding_sequences)

## Data Flow

1. Download binding data from both sources using [download_binding_data.sh](./download_binding_data.sh)
2. Combine the datasets using [prepare_binding_training_set](./prepare_binding_training_set.py)
3. Publish the training set using [publish.py](./publish.py)