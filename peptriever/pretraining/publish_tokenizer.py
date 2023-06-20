import os.path
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

from peptriever.pretraining.config import PretrainingConfig
from peptriever.pretraining.load_tokenizer import load_local_tokenizer


def push_tokenizer_to_hf(config: PretrainingConfig):
    fast_tokenizer = load_local_tokenizer(config.tokenizer_path)
    repo_id = config.hf_tokenizer_repo
    fast_tokenizer.push_to_hub(  # pylint: disable=not-callable
        repo_id=repo_id, private=False
    )
    template_path = _get_template_path()

    card = ModelCard.from_template(
        ModelCardData(language="en", license="mit"),
        dataset=config.hf_sequence_repo_name,
        template_path=str(template_path),
        vocab_size=config.vocab_size,
    )
    card.push_to_hub(repo_id)


def _get_template_path():
    here = Path(os.path.dirname(__file__))
    template_path = here / "templates" / "tokenizer_card_template.md"
    return template_path


if __name__ == "__main__":
    push_tokenizer_to_hf(config=PretrainingConfig())
