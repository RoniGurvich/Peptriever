from typing import List

import torch


def angular_batch_accuracy(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dot = _cosine_similarity(y1, y2)
    _, max_ind = torch.max(all_dot, dim=-1)
    diag_ind = torch.arange(all_dot.shape[0], device=y1.device)
    return torch.eq(max_ind, diag_ind).float().mean()


def angular_margin(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dot = _cosine_similarity(y1, y2)
    all_margins = torch.diag(all_dot) - all_dot
    all_margins.fill_diagonal_(1e8)
    margin = torch.amin(all_margins, dim=-1)
    return margin.mean()


def angular_mean_margin(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dot = _cosine_similarity(y1, y2)
    all_margins = torch.diag(all_dot) - all_dot
    all_margins.fill_diagonal_(0)
    margin = torch.sum(all_margins, dim=-1) / (all_margins.shape[0] - 1)
    return margin.mean()


def _cosine_similarity(y1, y2):
    y1_norm = torch.norm(y1, dim=-1, keepdim=True, p=2)
    y2_norm = torch.norm(y2, dim=-1, keepdim=True, p=2).transpose(1, 0)
    all_dot = torch.matmul(y1, y2.transpose(1, 0)) / (y1_norm * y2_norm)
    return all_dot


def euclidean_accuracy(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dist = torch.cdist(y1, y2, p=2)
    _, min_ind = torch.min(all_dist, dim=-1)
    diag_ind = torch.arange(all_dist.shape[0], device=y1.device)
    return torch.eq(min_ind, diag_ind).float().mean()


def euclidean_margin(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dist = torch.cdist(y1, y2, p=2)
    pos_dist = torch.diag(all_dist)
    all_neg_dist = all_dist.fill_diagonal_(1e8)
    neg_dist = torch.amin(all_neg_dist, dim=-1)
    margin = neg_dist - pos_dist
    return margin.mean()


def euclidean_mean_margin(outputs: List[torch.Tensor]):
    y1, y2 = outputs
    all_dist = torch.cdist(y1, y2, p=2)
    pos_dist = torch.diag(all_dist)
    all_margin = all_dist - pos_dist
    all_margin = all_margin.fill_diagonal_(0)
    mean_margin = torch.sum(all_margin, dim=-1) / (all_margin.shape[0] - 1)
    return mean_margin.mean()


def classification_accuracy(labels, logits):
    preds = torch.argmax(logits, dim=-1)
    is_relevant = labels != -100
    n_correct = torch.sum((preds == labels) * is_relevant, dim=-1, keepdim=True)
    n_relevant = torch.sum(is_relevant, dim=-1, keepdim=True)
    accuracy = n_correct / n_relevant
    accuracy = accuracy[torch.isfinite(accuracy)]
    return accuracy.mean()
