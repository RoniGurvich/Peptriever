from typing import Optional, Tuple

import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel, PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BertEmbeddingConfig(BertConfig):
    n_output_dims: int
    distance_func: str = "euclidean"


class BiEncoderConfig(BertEmbeddingConfig):
    max_length1: int
    max_length2: int


class BiEncoder(PreTrainedModel):
    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig):
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


class BiEncoderWithMaskedLM(PreTrainedModel):
    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig):
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


class L2Norm:
    def __call__(self, x):
        return x / torch.norm(x, p=2, dim=-1, keepdim=True)


class BertForEmbedding(BertPreTrainedModel):
    config_class = BertEmbeddingConfig

    def __init__(self, config: BertEmbeddingConfig):
        super().__init__(config)
        n_output_dims = config.n_output_dims
        self.fc = torch.nn.Linear(config.hidden_size, n_output_dims)
        self.bert = BertModel(config)
        self.activation = _get_activation(config.distance_func)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        embedding, _ = self.forward_with_state(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return embedding

    def forward_with_state(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output = encoded.pooler_output
        logits = self.fc(pooler_output)
        embedding = self.activation(logits)
        return embedding, encoded.last_hidden_state


def _get_activation(distance_func: str):
    if distance_func == "euclidean":
        activation = torch.nn.Tanh()
    elif distance_func == "angular":
        activation = L2Norm()  # type: ignore
    else:
        raise NotImplementedError()
    return activation
