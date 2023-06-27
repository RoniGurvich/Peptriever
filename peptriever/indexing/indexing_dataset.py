import collections

from torch.utils.data import Dataset

from peptriever.pdb_seq_lookup import get_pdb_lookup_with_meta


class PDBChainDataset(Dataset):
    def __init__(
        self,
        pdb_seq_repo: str,
        min_seq_length: int,
        max_seq_length: int,
    ):
        self.pdb_lookup = get_filtered_pdb_lookup(
            max_seq_length, min_seq_length, pdb_seq_repo
        )
        self.pdb_names = sorted(self.pdb_lookup.keys())

    def __len__(self):
        return len(self.pdb_names)

    def __getitem__(self, index: int):
        pdb_name, chain_id = self.pdb_names[index]
        record = self.pdb_lookup[(pdb_name, chain_id)]

        return {"pdb_name": pdb_name, "chain_id": chain_id, **record}

    @property
    def organism_counts(self):
        counter = collections.Counter()
        counter.update(
            v["organism"] for v in self.pdb_lookup.values() if v["organism"] is not None
        )
        return counter


def get_filtered_pdb_lookup(max_seq_length, min_seq_length, pdb_seq_repo):
    pdb_lookup = get_pdb_lookup_with_meta(pdb_seq_repo)
    pdb_lookup = {
        key: record
        for key, record in pdb_lookup.items()
        if min_seq_length < len(record["sequence"]) < max_seq_length
    }
    return pdb_lookup
