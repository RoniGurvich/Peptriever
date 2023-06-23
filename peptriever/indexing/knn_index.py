import json
from pathlib import Path
from typing import List

from annoy import AnnoyIndex


class KNNIndex:
    def __init__(self, n_dims: int, dist_func: str):
        self.annoy_index = AnnoyIndex(n_dims, dist_func)  # type: ignore
        self.metadata: List[dict] = []

    def __len__(self):
        return len(self.metadata)

    def insert_vector(self, entry):
        entry = entry.copy()
        ind = len(self.metadata)
        self.annoy_index.add_item(ind, entry.pop("vector"))
        self.metadata.append(entry)

    def build(self, n_trees: int):
        self.annoy_index.build(n_trees=n_trees)

    def save(self, index_path: Path, metadata_path: Path):
        self.annoy_index.save(str(index_path))
        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.metadata, f)
