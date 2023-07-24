import math

import pytest
import torch

from peptriever.finetuning.loss import EuclideanMarginLoss


@pytest.mark.parametrize("bs, n_dim", [(4, 12)])
def test_contrastive_loss_shape(bs, n_dim):
    peptide_embeddings = torch.rand((bs, n_dim), dtype=torch.float32)
    protein_embeddings = torch.rand((bs, n_dim), dtype=torch.float32)
    loss_f = EuclideanMarginLoss()
    loss = loss_f({"y1": peptide_embeddings, "y2": protein_embeddings}, None)
    assert loss.ndim == 0


@pytest.mark.parametrize("n_dim", [(16)])
def test_loss_min(n_dim):
    pep1_embeddings = torch.ones((n_dim,))
    prot1_embeddings = torch.ones((n_dim,))
    pep2_embeddings = -1 * torch.ones((n_dim,))
    prot2_embeddings = -1 * torch.ones((n_dim,))
    prots = torch.stack([prot1_embeddings, prot2_embeddings])
    peps = torch.stack([pep1_embeddings, pep2_embeddings])
    loss_f = EuclideanMarginLoss()
    loss = loss_f({"y1": peps, "y2": prots}, None)
    assert torch.eq(loss, 0)


@pytest.mark.parametrize("n_dim", [(16)])
def test_loss_max(n_dim):
    pep1_embeddings = -1 * torch.ones((n_dim,))
    prot1_embeddings = torch.ones((n_dim,))
    pep2_embeddings = torch.ones((n_dim,))
    prot2_embeddings = -1 * torch.ones((n_dim,))
    peps = torch.stack([pep1_embeddings, pep2_embeddings])
    prots = torch.stack([prot1_embeddings, prot2_embeddings])
    loss_f = EuclideanMarginLoss()
    loss = loss_f({"y1": peps, "y2": prots}, None)
    max_loss = 2 * math.sqrt(n_dim)
    assert loss == max_loss
