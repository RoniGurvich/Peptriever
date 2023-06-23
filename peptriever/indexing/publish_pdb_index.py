from pathlib import Path

from huggingface_hub import HfApi


def publish_pdb_index(hf_index_repo, index_path: Path):
    api = HfApi()
    api.create_repo(
        repo_id=hf_index_repo, private=True, repo_type="dataset", exist_ok=True
    )
    api.upload_folder(
        folder_path=str(index_path),
        repo_id=hf_index_repo,
        repo_type="dataset",
        allow_patterns=["*.ann", "*.json"],
    )
