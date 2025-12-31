"""Small CLI helpers shared by the standalone scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def _load_vector(path: str, device: torch.device) -> torch.Tensor:
    """Load a single vector from ``path``.

    The checkpoint can either contain a flat tensor directly or a dictionary
    with one of the commonly-used keys (``vector``, ``mu``, ``embedding``).
    """

    payload = torch.load(Path(path), map_location=device)
    candidates = [payload]
    if isinstance(payload, dict):
        candidates.extend(
            payload.get(key) for key in ["vector", "mu", "embedding", "mu_base", "mu_donor"]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        tensor = torch.as_tensor(candidate, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            return tensor
    raise ValueError(f"Could not locate a 1-D vector in {path}")


def load_matrix(path: str, device: torch.device) -> torch.Tensor:
    """Load a 2-D tensor from disk.

    Similar to :func:`_load_vector`, but enforces two-dimensionality.
    """

    payload = torch.load(Path(path), map_location=device)
    candidates = [payload]
    if isinstance(payload, dict):
        for key in ["matrix", "vectors", "centers", "components", "rows"]:
            if key in payload:
                candidates.append(payload[key])
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            continue
        tensor = torch.as_tensor(candidate, dtype=torch.float32, device=device)
        if tensor.ndim == 2:
            return tensor
    raise ValueError(f"Could not locate a 2-D tensor in {path}")


def load_token_list(path: str) -> List[str]:
    """Load a newline-separated list of tokens."""

    tokens: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            tokens.append(token)
    if not tokens:
        raise ValueError(f"No tokens found in {path}")
    return tokens


def save_tensor(path: str, tensor: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)


def save_token_list(path: str, tokens: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(tokens) + "\n", encoding="utf-8")
