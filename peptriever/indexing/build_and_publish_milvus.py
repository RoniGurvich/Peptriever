import tempfile

import typer
from huggingface_hub import HfApi
from pymilvus import connections

from peptriever.acceleration import get_device
from peptriever.indexing.config import IndexingConfig
from peptriever.indexing.milvus_index_builder import MilvusIndexBuilder
from peptriever.indexing.publish_biencoder import publish_biencoder
from peptriever.model.bi_encoder import BiEncoder


def index_to_milvus(model_name: str, milvus_token: str):
    config = IndexingConfig()

    connections.connect(
        "default",
        uri=config.milvus_uri,
        token=milvus_token,
        db_name=config.milvus_db_name,
    )

    model = _load_model(config=config, model_name=model_name)

    index_builder = MilvusIndexBuilder(model=model, config=config)
    index_builder.build_milvus()
    organisms = index_builder.get_organisms()
    publish_biencoder(config=config, model_name=model_name)
    upload_organisms_to_hf(organisms, config.hf_demo_space)


def _load_model(config: IndexingConfig, model_name: str):
    model = BiEncoder.from_pretrained(str(config.models_path / model_name))
    model.eval()
    device = get_device()
    model = model.to(device)
    return model


def upload_organisms_to_hf(organisms, demo_space: str):
    api = HfApi()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8") as f:
        for organism in sorted(organisms):
            f.write(organism + "\n")
        f.flush()
        api.upload_file(
            repo_id=demo_space,
            repo_type="space",
            path_or_fileobj=f.name,
            path_in_repo="available_organisms.txt",
        )


if __name__ == "__main__":
    typer.run(index_to_milvus)
