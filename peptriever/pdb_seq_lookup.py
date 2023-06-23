from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def get_pdb_lookup(repo_name, train_part: Optional[str] = None):
    dataset = load_dataset(repo_name)
    lookup = {}
    df = dataset["train"].to_pandas()

    if train_part is not None:
        df = df[df["train_part"] == train_part]

    for row in tqdm(df.to_dict("records"), total=len(df)):
        pdb_name = row["name"]
        chain = row["chain"]
        if chain is not None:
            if train_part is None or train_part == row["train_part"]:
                lookup[(pdb_name, chain)] = row["sequence"]
    return lookup


def get_pdb_lookup_with_meta(repo_name):
    dataset = load_dataset(repo_name)
    lookup = {}
    for row in dataset["train"].to_pandas().to_dict("records"):
        pdb_name = row.pop("name")
        chain = row.pop("chain")
        if chain is not None:
            lookup[(pdb_name, chain)] = row
    return lookup
