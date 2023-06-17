import os.path
from pathlib import Path

from huggingface_hub import HfApi

from peptriever.data_sources.binding.config import BindingDataConfig


def push_dataset_to_hf(config: BindingDataConfig):
    csv_path = config.training_set_path

    api = HfApi()
    api.create_repo(repo_id=config.hf_dataset_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(csv_path),
        path_in_repo=os.path.basename(csv_path),
        repo_id=config.hf_dataset_repo,
        repo_type="dataset",
    )
    here = Path(os.path.dirname(__file__))
    model_card_path = here / "dataset_readme.md"
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=config.hf_dataset_repo,
        repo_type="dataset",
    )


if __name__ == "__main__":
    push_dataset_to_hf(config=BindingDataConfig())
