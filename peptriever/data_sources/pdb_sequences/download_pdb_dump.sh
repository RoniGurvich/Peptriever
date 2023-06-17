#!/bin/bash
DATA_PATH=/data/training_sets/external/pdb

COMPRESSED_MODELS_PATH=$DATA_PATH/compressed_models/
MODELS_PATH=$DATA_PATH/models
mkdir -p $COMPRESSED_MODELS_PATH $MODELS_PATH

echo "downloading sequences"
cd $DATA_PATH  && wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
cd $DATA_PATH  && gunzip pdb_seqres.txt.gz

echo "Downloading Gene Lookup"
cd $DATA_PATH && wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_ensembl.csv.gz && gunzip pdb_chain_ensembl.csv.gz

echo "Downloading uniprot metadata"
cd $DATA_PATH && wget https://storage.googleapis.com/public-552/datasets/uniprot_metadata.tsv