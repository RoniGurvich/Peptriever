from huggingface_hub import HfApi

from peptriever.data_sources.pdb_sequences.config import PDBDataConfig


def push_dataset_to_hf(config: PDBDataConfig):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=config.sequence_dataset_path,
        path_in_repo=config.sequences_fname,
        repo_id=config.hf_sequence_repo_name,
        repo_type="dataset",
    )


if __name__ == "__main__":
    push_dataset_to_hf(config=PDBDataConfig())
