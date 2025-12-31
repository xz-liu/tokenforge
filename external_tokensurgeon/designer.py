"""Core data structures shared across the donor-aware designing scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


_EPS = 1e-8


@dataclass
class DesignConfig:
    """Configuration for sparse support design.

    Attributes
    ----------
    k:
        Sparsity level for the OMP solver (number of support tokens).
    lambda_penalty:
        Weight applied to donor penalties when balancing base/donor behaviour.
    eta_penalty:
        Weight applied to negative base examples (discouraging overlap).
    gamma:
        Scaling factor for the quadratic objective on the base reconstruction term.
    ridge:
        Positive value added to the diagonal of Gram matrices for numerical stability.
    normalize:
        Whether to unit-normalize overlap embeddings before running OMP.
    """

    k: int = 16
    lambda_penalty: float = 0.0
    eta_penalty: float = 0.0
    gamma: float = 1.0
    ridge: float = 1e-4
    normalize: bool = True

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative")
        if self.eta_penalty < 0:
            raise ValueError("eta_penalty must be non-negative")


@dataclass
class DesignResult:
    """Structured result returned by the sparse designer."""

    alpha: torch.Tensor
    support_indices: List[int]
    support_tokens: List[str]
    base_embedding: torch.Tensor
    donor_embedding: torch.Tensor
    objective: float
    residual: torch.Tensor
    scores: torch.Tensor


def _normalize_rows(matrix: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Unit-normalize each row of ``matrix``.

    Parameters
    ----------
    matrix:
        Input tensor of shape ``(num_rows, dim)``.
    eps:
        Numerical stability constant to avoid dividing by zero.
    """

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(matrix.shape)}")
    norms = matrix.norm(dim=1, keepdim=True).clamp_min(eps)
    return matrix / norms


def stack_vectors(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Stack an iterable of vectors into a 2-D tensor.

    Missing vectors (``None``) are ignored. The function ensures that all
    non-missing vectors share the same dimensionality.
    """

    items = [vec for vec in vectors if vec is not None]
    if not items:
        raise ValueError("No vectors provided to stack")
    dim = items[0].shape[-1]
    for vec in items:
        if vec.ndim != 1:
            raise ValueError("Each vector must be 1-D")
        if vec.shape[0] != dim:
            raise ValueError("Vectors must share the same dimensionality")
    return torch.stack(items, dim=0)
