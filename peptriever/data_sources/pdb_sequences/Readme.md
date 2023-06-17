# PDB Data Source

This module downloads a full dump of the [Protein Data Bank](https://www.rcsb.org/),
extracts the sequences and uploads them to
a [Huggingface Dataset](https://huggingface.co/datasets/ronig/pdb_sequences)

## Data Flow

1. Download latest PDB dump using [download_pdb_dump.sh](./download_pdb_dump.sh)
   script. (this will download over 100GB of data)
2. Convert the downloaded PDB files to one text file with sequences
   using [extract_pdb_sequences.py](./extract_pdb_sequences.py)
3. Publish the sequences file to huggingface using [publish.py](./publish.py)