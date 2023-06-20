from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from peptriever.model.bert_embedding import BertEmbeddingConfig, BertForEmbedding


class PeptrieverConfig(BertEmbeddingConfig):
    max_length1: int
    max_length2: int


class Peptriever(PreTrainedModel):
    config_class = PeptrieverConfig

    def __init__(self, config: PeptrieverConfig):
        super().__init__(config)
        config1 = _replace_max_length(config, "max_length1")
        self.bert1 = BertForEmbedding(config1)
        config2 = _replace_max_length(config, "max_length2")
        self.bert2 = BertForEmbedding(config2)
        self.post_init()

    def forward(self, x1, x2):
        y1 = self.forward1(x1)
        y2 = self.forward2(x2)
        return {"y1": y1, "y2": y2}

    def forward2(self, x2):
        y2 = self.bert2(input_ids=x2["input_ids"])
        return y2

    def forward1(self, x1):
        y1 = self.bert1(input_ids=x1["input_ids"])
        return y1


class PeptrieverWithMaskedLM(PreTrainedModel):
    config_class = PeptrieverConfig

    def __init__(self, config: PeptrieverConfig):
        super().__init__(config=config)
        config1 = _replace_max_length(config, "max_length1")
        self.bert1 = BertForEmbedding(config1)
        self.lm_head1 = BertOnlyMLMHead(config=config1)

        config2 = _replace_max_length(config, "max_length2")
        self.bert2 = BertForEmbedding(config2)
        self.lm_head2 = BertOnlyMLMHead(config=config2)
        self.post_init()

    def forward(self, x1, x2):
        y1, state1 = self.bert1.forward_with_state(input_ids=x1["input_ids"])
        y2, state2 = self.bert2.forward_with_state(input_ids=x2["input_ids"])
        scores1 = self.lm_head1(state1)
        scores2 = self.lm_head2(state2)
        outputs = {"y1": y1, "y2": y2, "scores1": scores1, "scores2": scores2}
        return outputs


def _replace_max_length(config, length_key):
    c1 = config.__dict__.copy()
    c1["max_position_embeddings"] = c1.pop(length_key)
    config1 = BertEmbeddingConfig(**c1)
    return config1
