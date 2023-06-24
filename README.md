# Peptriever

[![CI](https://github.com/RoniGurvich/Peptriever/actions/workflows/ci.yml/badge.svg)](https://github.com/RoniGurvich/Peptriever/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About

This repo contains all the code needed in order to train Peptriever end to end.

## Local Setup

The dependencies are managed using [Poetry](https://python-poetry.org/). Make sure you
install poetry using `pipx` or
the [official instructions](https://python-poetry.org/docs/#installing-with-the-official-installer)

```bash
pipx install poetry
```

After Poetry is installed you can run `make setup` to install all the requirements in a
virtual environment.

## System Architecture Diagram

```mermaid
flowchart TD
    subgraph legend[Legend]
        data[Data]
        process{{Process}}
    end

    subgraph data_sources[Data Sources]

        subgraph pdb_seq[PDB Sequences]
            pdb_dump[PDB Data Dump] --> extract_sequences{{Extract Sequences}} --> pdb_sequences[PDB Sequences]
            click pdb_sequences "https://huggingface.co/datasets/ronig/pdb_sequences" "huggingface dataset"
        end
        
        subgraph binding[Binding]
            huang_data[Huang Lab Data]
            propedia_data[Propedia Data]
            yapp_data[YAPP-Cd]
            huang_data --> preprocess_train_data{{Prepare Binding Training Set}}
            propedia_data --> preprocess_train_data
            yapp_data --> preprocess_train_data
            preprocess_train_data --> binding_sequences[Binding Sequences]
            click binding_sequences "https://huggingface.co/datasets/ronig/protein_binding_sequences" "huggingface dataset"
        end
        
    end

    subgraph pretraining[Pretraining]
        pdb_sequences --> train_tokenizer{{Train Tokenizer}} --> tokenizer[Tokenizer]
        tokenizer --> mlm_pretraining{{Masked Language Pretraining}}
        mlm_pretraining --> pretrained_mlm[Pretrained Models]
        click tokenizer "https://huggingface.co/ronig/pdb_bpe_tokenizer_1024_mlm" "huggingface model"
    end
    
    subgraph training[Training]
        pretrained_mlm --> finetune{{Finetune Models}}
        binding_sequences --> finetune
        finetune --> trained_model[Trained Model]
        click trained_model "https://huggingface.co/ronig/protein_biencoder" "huggingface model"
    end
    
    subgraph indexing[Indexing]
        trained_model --> build_index{{Build Index}}
        pdb_sequences --> build_index
        build_index --> knn_index[(Index)]
        knn_index --> publish_index_model{{Publish Index and Model}}
        click knn_index "https://huggingface.co/datasets/ronig/protein_index" "huggingface dataset"
    end

    publish_index_model --> search_app((Search App))
    click search_app "https://peptriever.app" "Peptriever App"

```

## Model Details

### Model Architecture

Peptriever is a Bi Encoder Bert model, combined with a Byte-Pair Encoding tokenizer.

```mermaid
flowchart TD
    protein_sequence[Protein Sequence] --> protein_encoder[Protein BERT] --> protein_vector[Protein Vector]
    peptide_sequence[Peptide Sequence] --> peptide_encoder[Peptide BERT] --> peptide_vector[Peptide Vector]
    peptide_vector --> euclidean[Euclidean Distance <--> Binding Score] 
    protein_vector --> euclidean
```

### Evaluation Results

The model was evaluated on the test set
from [Johansson-Akhe et al.](https://www.frontiersin.org/articles/10.3389/fbinf.2022.959160/full)

![Precision-Recall](./doc/img/test_pr.png)
![ROC](./doc/img/test_roc.png)
