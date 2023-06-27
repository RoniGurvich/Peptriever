import typer
from huggingface_hub import HfApi

from peptriever.acceleration import get_device
from peptriever.indexing.config import IndexingConfig
from peptriever.indexing.annoy_index_builder import IndexBuilder
from peptriever.indexing.publish_biencoder import publish_biencoder
from peptriever.indexing.publish_pdb_index import publish_pdb_index
from peptriever.model.bi_encoder import BiEncoder


def build_and_publish(model_name: str):
    config = IndexingConfig()
    model = _load_model(config=config, model_name=model_name)

    index_builder = IndexBuilder(model=model, config=config)
    index_builder.build_index(output_dir=config.local_indexes_path)

    publish_biencoder(config=config, model_name=model_name)
    publish_pdb_index(
        hf_index_repo=config.hf_index_repo,
        index_path=config.local_indexes_path,
    )
    restart_demo(config)


def _load_model(config: IndexingConfig, model_name: str):
    model = BiEncoder.from_pretrained(str(config.models_path / model_name))
    model.eval()
    device = get_device()
    model = model.to(device)
    return model


def restart_demo(config):
    api = HfApi()
    api.restart_space(config.hf_demo_space)


if __name__ == "__main__":
    typer.run(build_and_publish)
