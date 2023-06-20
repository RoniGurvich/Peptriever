from typing import Optional, Tuple

import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel


class BertEmbeddingConfig(BertConfig):
    n_output_dims: int
    distance_func: str = "euclidean"


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
