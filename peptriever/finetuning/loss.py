import math
from typing import Dict

import torch
from torch.nn import CrossEntropyLoss


class EuclideanMarginLoss:
    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def __call__(self, outputs: Dict[str, torch.Tensor], _):
        y1, y2 = outputs["y1"], outputs["y2"]
        all_dist = torch.cdist(y1.double(), y2.double(), p=2)
        pos_dist = torch.diag(all_dist)
        n_dim = y1.shape[-1]
        max_dist = 2 * math.sqrt(n_dim)
        loss = torch.clip(
            pos_dist[:, None] - all_dist + self.margin * max_dist,
            0,
            max_dist * (1 + self.margin)
        )
        loss = loss.fill_diagonal_(0).sum(dim=-1) / (loss.shape[1] - 1)
        return loss.mean()


class EuclideanCombinedMLMMarginLoss:
    def __init__(self, margin: float = 0.05, ce_coeff: float = 0.1):
        self.margin_loss = EuclideanMarginLoss(margin=margin)
        self.ce_coeff = ce_coeff
        self.ce = CrossEntropyLoss()

    def __call__(self, outputs: Dict[str, torch.Tensor], labels):
        labels1, labels2 = labels
        scores1, scores2 = outputs["scores1"], outputs["scores2"]

        m_loss = self.margin_loss(outputs, labels)
        scores1, labels1 = self._flatten_scores_and_labels(scores1, labels1)
        scores2, labels2 = self._flatten_scores_and_labels(scores2, labels2)
        ce1 = self.ce(scores1, labels1).mean()
        ce2 = self.ce(scores2, labels2).mean()
        ce_loss = (ce1 + ce2) / 2
        loss = m_loss + self.ce_coeff * ce_loss
        return loss

    def _flatten_scores_and_labels(self, scores, labels):
        vocab_size = scores.shape[-1]
        return scores.view(-1, vocab_size), labels.view(-1)
