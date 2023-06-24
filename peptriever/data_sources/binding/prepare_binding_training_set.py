import csv
import itertools
from pathlib import Path

from peptriever.data_sources.binding.config import BindingDataConfig
from peptriever.data_sources.binding.huang import get_huang_training_samples
from peptriever.data_sources.binding.train_partitioner import TrainParitioner
from peptriever.data_sources.binding.yapp import get_yapp_training_samples
from peptriever.data_sources.binding.propedia import get_propedia_training_samples


def preprocess_binding_training_set(
    pdb_path: Path,
    data_path: Path,
    output_path: Path,
    test_ratio: float,
    val_ratio: float,
    pdb_hf_repo: str,
):
    train_partitioner = TrainParitioner(
        binding_data_path=data_path,
        pdb_path=pdb_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pdb_hf_repo=pdb_hf_repo,
    )
    huang_training_samples = get_huang_training_samples(
        data_path=data_path, train_partitioner=train_partitioner
    )
    propedia_training_samples = list(
        get_propedia_training_samples(
            data_path=data_path,
            train_partitioner=train_partitioner,
        )
    )
    yapp_training_samples = get_yapp_training_samples(
        data_path=data_path, train_partitioner=train_partitioner
    )
    training_samples = combine_dedup_entries(
        [huang_training_samples, propedia_training_samples, yapp_training_samples]
    )
    write_training_set(output_fname=output_path, training_set=training_samples)


def combine_dedup_entries(data_sources):
    entries = itertools.chain(*data_sources)
    used = set()
    filtered_entries = []
    for entry in entries:
        key = entry["peptide"] + entry["receptor"]
        if key not in used:
            used.update([key])
            filtered_entries.append(entry)
    return filtered_entries


def write_training_set(output_fname, training_set):
    fieldnames = list(training_set[0].keys())
    with open(output_fname, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_set)


if __name__ == "__main__":
    config = BindingDataConfig()
    preprocess_binding_training_set(
        pdb_path=config.pdb_path,
        data_path=config.binding_path,
        output_path=config.training_set_path,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        pdb_hf_repo=config.hf_pdb_repo,
    )
