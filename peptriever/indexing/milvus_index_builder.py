import os
from typing import Callable

import torch
from pymilvus import Collection, CollectionSchema, utility, FieldSchema, DataType
from torch.utils.data import DataLoader
from tqdm import tqdm

from peptriever.indexing.config import IndexingConfig
from peptriever.indexing.sequence_collator import SequenceCollator
from peptriever.indexing.indexing_dataset import PDBChainDataset


class MilvusIndexBuilder:
    def __init__(self, model, config: IndexingConfig):
        self.model = model
        self.config = config
        _drop_collection_if_exists(config.milvus_collection_name)
        schema = _get_schema(config.encoded_sequence_dims)
        self.collection = Collection(name=config.milvus_collection_name, schema=schema)
        self.insert_batch_size = config.milvus_insert_batch_size

    def build_milvus(self):
        extractors = self.setup_vector_extractors()
        self.insert_vectors_to_milvus(extractors)
        self.build_milvus_index()

    def insert_vectors_to_milvus(self, extractors):
        for extractor_i, extractor in enumerate(extractors):
            batch = []
            msg = f"extracting vectors {extractor_i}"
            for entry_i, entry in tqdm(enumerate(extractor), desc=msg):
                if all(
                    [
                        entry["genes"] is not None,
                        entry["organism"] is not None,
                        entry["uniprot_id"] is not None,
                    ]
                ):
                    entry["is_peptide"] = entry.pop("index_i") == 0
                    batch.append(entry)

                if len(batch) >= self.insert_batch_size:
                    self.collection.insert(batch)
                    batch = []

            if batch:
                self.collection.insert(batch)

    def build_milvus_index(self):
        index_params = {"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}}
        self.collection.create_index(field_name="vector", index_params=index_params)
        self.collection.load()

    def setup_vector_extractors(self):
        n_workers = os.cpu_count() or 1
        pep_dataloader = self.setup_pep_dataloader(n_workers)
        pep_vectors = extract_vectors(
            forward=self.model.forward1,
            dataloader=pep_dataloader,
            device=self.model.device,
            index_i=0,
        )
        prot_dataloader = self.setup_prot_dataloader(n_workers)
        prot_vectors = extract_vectors(
            forward=self.model.forward2,
            dataloader=prot_dataloader,
            device=self.model.device,
            index_i=1,
        )
        return pep_vectors, prot_vectors

    def setup_pep_dataloader(self, n_workers: int):
        dataset = self.get_pep_dataset()
        collator = SequenceCollator(
            hf_tokenizer_repo=self.config.hf_tokenizer_repo,
            max_length=self.config.max_length1,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1024,
            num_workers=n_workers,
            shuffle=False,
            collate_fn=collator,
            prefetch_factor=10,
        )
        return dataloader

    def setup_prot_dataloader(self, n_workers: int):
        dataset = self.get_prot_dataset()
        collator = SequenceCollator(
            hf_tokenizer_repo=self.config.hf_tokenizer_repo,
            max_length=self.config.max_length2,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=256,
            num_workers=n_workers,
            shuffle=False,
            collate_fn=collator,
            prefetch_factor=10,
        )
        return dataloader

    def get_pep_dataset(self):
        return PDBChainDataset(
            pdb_seq_repo=self.config.hf_pdb_seq_repo,
            min_seq_length=5,
            max_seq_length=50,
        )

    def get_prot_dataset(self):
        return PDBChainDataset(
            pdb_seq_repo=self.config.hf_pdb_seq_repo,
            min_seq_length=25,
            max_seq_length=int(1e6),
        )

    def get_organisms(self, min_prots: int = 1000):
        prot_dataset = self.get_prot_dataset()
        organism_counts = prot_dataset.organism_counts
        organisms = [
            organism
            for organism, count in organism_counts.items()
            if count >= min_prots
        ]
        return organisms


def extract_vectors(forward: Callable, dataloader: DataLoader, device, index_i):
    with torch.no_grad():
        for batch in dataloader:
            encoded = batch.pop("encoded")
            encoded["input_ids"] = encoded["input_ids"].to(device)
            vecs = forward(encoded)
            for sample_i, entry in enumerate(batch["batch_data"]):
                vec = vecs[sample_i, :]
                entry["vector"] = vec.squeeze().cpu().numpy()
                entry["index_i"] = index_i
                entry.pop("sequence")
                yield entry


def _drop_collection_if_exists(collection_name):
    check_collection = utility.has_collection(collection_name)
    if check_collection:
        utility.drop_collection(collection_name)


def _get_schema(dim):
    entry_id = FieldSchema(name="entry_id", dtype=DataType.INT64, is_primary=True)
    pdb_name = FieldSchema(name="pdb_name", dtype=DataType.VARCHAR, max_length=10)
    chain_id = FieldSchema(name="chain_id", dtype=DataType.VARCHAR, max_length=2)
    genes = FieldSchema(name="genes", dtype=DataType.VARCHAR, max_length=20)
    organism = FieldSchema(name="organism", dtype=DataType.VARCHAR, max_length=100)
    uniprot_id = FieldSchema(name="uniprot_id", dtype=DataType.VARCHAR, max_length=10)
    train_part = FieldSchema(name="train_part", dtype=DataType.VARCHAR, max_length=5)
    index_i = FieldSchema(name="is_peptide", dtype=DataType.BOOL)
    vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(
        fields=[
            entry_id,
            pdb_name,
            chain_id,
            genes,
            organism,
            uniprot_id,
            train_part,
            index_i,
            vector,
        ],
        auto_id=True,
        description="",
    )
    return schema
