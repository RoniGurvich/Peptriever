import collections
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from peptriever.indexing.config import IndexingConfig
from peptriever.indexing.indexing_dataset import PDBChainDataset
from peptriever.indexing.knn_index import KNNIndex


class IndexBuilder:
    def __init__(self, model, config: IndexingConfig):
        self.model = model
        self.config = config

    def build_index(self, output_dir: Path):
        extractors = self.setup_vector_extractors()
        indexes = self._insert_vectors_to_inedexes(extractors)
        self._build_and_save_indexes(indexes, output_dir)

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

    def _insert_vectors_to_inedexes(self, extractors: Iterable):
        indexes: Dict[Tuple[str, int], KNNIndex] = collections.defaultdict(
            lambda: KNNIndex(
                n_dims=self.config.encoded_sequence_dims,
                dist_func=self.config.distance_function,
            )
        )
        for extractor_i, extractor in enumerate(extractors):
            msg = f"extracting vectors {extractor_i}"
            for entry in tqdm(extractor, desc=msg):
                index_i = entry["index_i"]
                full_index = indexes[("PDB", index_i)]
                full_index.insert_vector(entry)

                organism = entry["organism"]
                if organism is not None:
                    organism_index = indexes[(organism, index_i)]
                    organism_index.insert_vector(entry)
        return indexes

    def _build_and_save_indexes(self, indexes, output_dir):
        above_threshold_indexes = self._filter_above_threshold_indexes(indexes)
        for (index_name, index_i), index in above_threshold_indexes.items():
            index_name = index_name.replace("/", "-")
            index.build(n_trees=self.config.n_trees)
            subindex_dir = output_dir / index_name
            os.makedirs(subindex_dir, exist_ok=True)
            index.save(
                index_path=subindex_dir / f"index_{index_i}.ann",
                metadata_path=subindex_dir / f"metadata_{index_i}.json",
            )

    def _filter_above_threshold_indexes(self, indexes):
        index_counter = collections.defaultdict(int)
        for index_key, index in indexes.items():
            index_name = index_key[0]
            index_counter[index_name] += len(index)
        all_keys = set(indexes.keys())

        filtered_indexes = {}
        for index_key in all_keys:
            index_name = index_key[0]
            if index_counter[index_name] > self.config.min_indexed_entries:
                filtered_indexes[index_key] = indexes[index_key]

        return filtered_indexes


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


class SequenceCollator:
    def __init__(self, hf_tokenizer_repo, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_repo)
        self.max_length = max_length

    def __call__(self, batch: List[dict]):
        texts = [sample["sequence"] for sample in batch]
        encoded = self._encode(texts, self.max_length)
        return {"encoded": encoded, "batch_data": batch}

    def _encode(self, texts, max_length):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
