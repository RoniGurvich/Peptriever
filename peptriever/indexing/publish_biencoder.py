import os
import tempfile
from glob import glob
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, ModelCardData, snapshot_download
from transformers import AutoTokenizer

from peptriever.indexing.config import IndexingConfig
from peptriever.model.bi_encoder import BiEncoder


def publish_biencoder(config: IndexingConfig, model_name: str):
    api = HfApi()
    repo_id = config.hf_model_repo
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True, repo_type="model")
    tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer_repo)
    model = BiEncoder.from_pretrained(str(config.models_path / model_name))

    template_path = _get_template_path()

    card = ModelCard.from_template(
        ModelCardData(language="en", license="mit"),
        dataset=config.hf_binding_dataset_repo,
        model_repo=config.hf_model_repo,
        template_path=str(template_path),
        model_id=model_name,
    )

    commit_msg = f"updating model {model_name}"
    model.push_to_hub(repo_id=repo_id, commit_message=commit_msg)
    _push_tokenizer_to_model_repo(api, commit_msg, config, repo_id, tokenizer)
    card.push_to_hub(repo_id, commit_message=commit_msg)


def _push_tokenizer_to_model_repo(api, commit_msg, config, repo_id, tokenizer):
    tokenizer.push_to_hub(repo_id=repo_id, commit_message=commit_msg)
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_download(
            repo_id=config.hf_tokenizer_repo,
            cache_dir=temp_dir,
            allow_patterns=["*.txt", "*.json"],
        )
        tokenizer_path = glob(f"{temp_dir}/models*/snapshots/*")[0]
        api.upload_folder(
            folder_path=tokenizer_path,
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*.txt", "*.json"],
            commit_message=commit_msg,
        )


def _get_template_path():
    here = Path(os.path.dirname(__file__))
    template_path = here / "templates" / "biencoder_card_template.md"
    return template_path
