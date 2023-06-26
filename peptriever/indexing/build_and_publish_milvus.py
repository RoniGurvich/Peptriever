import typer
from pymilvus import connections

from peptriever.acceleration import get_device
from peptriever.indexing.config import IndexingConfig
from peptriever.indexing.milvus_index_builder import MilvusIndexBuilder
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


def _load_model(config: IndexingConfig, model_name: str):
    model = BiEncoder.from_pretrained(str(config.models_path / model_name))
    model.eval()
    device = get_device()
    model = model.to(device)
    return model


if __name__ == "__main__":
    typer.run(index_to_milvus)
