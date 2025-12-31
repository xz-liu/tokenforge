# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
"""Pure OMP-based sparse designer that can use BOTH input embeddings and LM-head rows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from mergekit.common import ModelReference
from mergekit.options import MergeOptions
from mergekit.scripts.tokensurgeon import get_stuff
from mergekit.tokensurgeon import batch_omp

from .cli import _load_vector
from .designer import DesignConfig, DesignResult, _normalize_rows

__all__ = ["design_with_pure_omp", "DesignConfig", "DesignResult"]

logger = logging.getLogger(__name__)


# -----------------------
# Utilities
# -----------------------

def _parse_model_ref(path: str) -> ModelReference:
    return ModelReference.model_validate({"model": path})


def _safe_same_ptr(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return a.data_ptr() == b.data_ptr()
    except Exception:
        return False


def _build_overlap(
    base_ref: ModelReference,
    donor_ref: ModelReference,
    merge_options: MergeOptions,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Return shared-token overlaps for:
      - input embeddings
      - lm_head rows (if available)
    Shapes: (|T|, d_model) for each matrix.
    """
    base_vocab, base_embed, base_head = get_stuff(base_ref, merge_options, device=str(device))
    donor_vocab, donor_embed, donor_head = get_stuff(donor_ref, merge_options, device=str(device))

    shared_tokens = sorted(set(base_vocab.keys()) & set(donor_vocab.keys()), key=lambda t: donor_vocab[t])
    if not shared_tokens:
        raise RuntimeError("Models do not share any tokens; cannot build overlap.")

    base_idx = torch.tensor([base_vocab[t] for t in shared_tokens], dtype=torch.long, device=base_embed.device)
    donor_idx = torch.tensor([donor_vocab[t] for t in shared_tokens], dtype=torch.long, device=donor_embed.device)

    base_embed_overlap = base_embed.index_select(0, base_idx).detach()
    donor_embed_overlap = donor_embed.index_select(0, donor_idx).detach()

    # lm_head may be None on some backends; if tied, get_stuff should return the same storage
    if base_head is None:
        base_head_overlap = base_embed_overlap
        logger.info("Base lm_head not found; using embeddings for head overlap.")
    else:
        base_head_overlap = base_head.index_select(0, base_idx).detach().to(base_embed_overlap.device)

    if donor_head is None:
        donor_head_overlap = donor_embed_overlap
        logger.info("Donor lm_head not found; using embeddings for head overlap.")
    else:
        donor_head_overlap = donor_head.index_select(0, donor_idx).detach().to(donor_embed_overlap.device)

    tied_base = _safe_same_ptr(base_embed_overlap, base_head_overlap)
    tied_donor = _safe_same_ptr(donor_embed_overlap, donor_head_overlap)
    if tied_base:
        logger.info("Base model appears tied (input embedding and lm_head share storage).")
    if tied_donor:
        logger.info("Donor model appears tied (input embedding and lm_head share storage).")

    return {
        "shared_tokens": shared_tokens,
        "base_embed_overlap": base_embed_overlap,
        "base_head_overlap": base_head_overlap,
        "donor_embed_overlap": donor_embed_overlap,
        "donor_head_overlap": donor_head_overlap,
        "tied_base": tied_base,
        "tied_donor": tied_donor,
    }


def _load_matrix(spec: str, device: torch.device) -> torch.Tensor:
    """Load a 2-D matrix with flexible 'path[:key[:rows]]' syntax."""
    parts = spec.split(":")
    file_str = parts[0]
    key: Optional[str] = None
    count: Optional[int] = None
    if len(parts) >= 2 and parts[1] != "":
        try:
            count = int(parts[1])
        except ValueError:
            key = parts[1]
    if len(parts) >= 3 and parts[2] != "":
        count = int(parts[2])

    payload = torch.load(Path(file_str), map_location=device)
    matrix: Optional[torch.Tensor] = None
    if key is not None:
        if not isinstance(payload, dict) or key not in payload:
            raise ValueError(f"Key '{key}' not found in payload from {file_str}")
        matrix = torch.as_tensor(payload[key], dtype=torch.float32, device=device)
    else:
        if isinstance(payload, dict):
            for k in ("matrix", "mat", "vectors", "centers", "cluster_centers", "components"):
                if k in payload:
                    matrix = torch.as_tensor(payload[k], dtype=torch.float32, device=device)
                    break
        if matrix is None:
            matrix = torch.as_tensor(payload, dtype=torch.float32, device=device)

    if matrix.ndim != 2:
        raise ValueError(f"Payload from {spec} is not 2D; got shape {tuple(matrix.shape)}")
    if count is not None:
        if count <= 0 or count > matrix.shape[0]:
            raise ValueError(f"Requested {count} rows from {file_str}, but matrix has {matrix.shape[0]}")
        matrix = matrix[:count]
    return matrix


def _orthonormalize_rows(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {tuple(matrix.shape)}")
    if matrix.numel() == 0:
        return matrix
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    tol = float(eps) * float(S.max().item() if S.numel() else 1.0)
    keep = S > tol
    if not bool(keep.any()):
        return _normalize_rows(matrix)
    basis = Vh[keep]
    return basis.to(dtype=matrix.dtype, device=matrix.device).contiguous()


def _combine_views(
    embed: torch.Tensor,
    head: torch.Tensor,
    *,
    use_embed: bool,
    use_head: bool,
    w_embed: float,
    w_head: float,
    normalize_each: bool,
    normalize_final: bool,
) -> torch.Tensor:
    """
    Build a single per-token dictionary row by combining embedding & head.
    Each view can be unit-normalized first; final combined row can be normalized as well.
    """
    assert embed.shape == head.shape, "embed/head must share dimensionality"
    rows = []
    if use_embed and w_embed != 0.0:
        e = embed
        if normalize_each:
            e = _normalize_rows(e)
        rows.append(e * float(w_embed))
    if use_head and w_head != 0.0:
        h = head
        if normalize_each:
            h = _normalize_rows(h)
        rows.append(h * float(w_head))
    if not rows:
        # fallback to embeddings
        e = embed if not normalize_each else _normalize_rows(embed)
        out = e
    else:
        out = torch.stack(rows, dim=0).sum(dim=0)
    if normalize_final:
        out = _normalize_rows(out)
    return out


# -----------------------
# Core solves
# -----------------------

def _solve_fixed_support(
    dict_base: torch.Tensor,      # (s, d) combined base dictionary rows (phi)
    dict_donor: torch.Tensor,     # (s, d) combined donor dictionary rows (phi)
    mu_base: torch.Tensor,        # (d,)
    *,
    mu_donor: Optional[torch.Tensor],
    mu_base_neg: Optional[torch.Tensor],
    mu_donor_matrix: Optional[torch.Tensor],
    config: DesignConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Least-squares polish on fixed support for the composite dictionary.
    Returns α, base_comb (in composite space), donor_comb (composite), residual.
    """
    effective_target = mu_base
    if mu_base_neg is not None:
        effective_target = effective_target - config.eta_penalty * mu_base_neg

    rhs = dict_base @ effective_target
    if mu_donor is not None:
        rhs = rhs - config.lambda_penalty * (dict_donor @ mu_donor)

    gram = dict_base @ dict_base.T
    if config.gamma != 0.0:
        gram = gram * (2.0 * config.gamma)
    if mu_donor_matrix is not None and config.lambda_penalty != 0.0:
        proj = dict_donor @ mu_donor_matrix.T              # (s, m)
        gram = gram + config.lambda_penalty * (proj @ proj.T)

    ridge = config.ridge + 1e-8
    gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)

    alpha = torch.linalg.solve(gram, rhs)
    base_comb = alpha @ dict_base
    donor_comb = alpha @ dict_donor
    residual = effective_target - base_comb
    return alpha, base_comb, donor_comb, residual


def _penalized_support(
    dict_base: torch.Tensor,
    dict_donor: torch.Tensor,
    mu_base: torch.Tensor,
    *,
    mu_donor: Optional[torch.Tensor],
    mu_base_neg: Optional[torch.Tensor],
    mu_donor_matrix: Optional[torch.Tensor],
    config: DesignConfig,
) -> List[int]:
    """
    Greedy support expansion with optional donor penalties, operating on the composite dictionary.
    """
    selected: List[int] = []
    effective_target = mu_base
    if mu_base_neg is not None:
        effective_target = effective_target - config.eta_penalty * mu_base_neg

    residual = effective_target.clone()
    banned = torch.zeros(dict_base.shape[0], dtype=torch.bool, device=dict_base.device)

    donor_proj_sq: Optional[torch.Tensor] = None
    if mu_donor_matrix is not None and config.lambda_penalty != 0.0:
        proj = dict_donor @ mu_donor_matrix.T
        donor_proj_sq = proj.pow(2).sum(dim=1)

    for _ in range(config.k):
        scores = dict_base @ residual
        if mu_donor is not None and config.lambda_penalty != 0.0:
            scores = scores - config.lambda_penalty * (dict_donor @ mu_donor)
        if mu_base_neg is not None and config.eta_penalty != 0.0:
            scores = scores - config.eta_penalty * (dict_base @ mu_base_neg)
        if donor_proj_sq is not None:
            scores = scores - config.lambda_penalty * donor_proj_sq

        scores[banned] = float("-inf")
        idx = int(torch.argmax(scores).item())
        if torch.isneginf(scores[idx]):
            break

        selected.append(idx)
        banned[idx] = True

        B_sel = dict_base[selected]
        D_sel = dict_donor[selected]
        _, _, _, residual = _solve_fixed_support(
            B_sel,
            D_sel,
            mu_base,
            mu_donor=mu_donor,
            mu_base_neg=mu_base_neg,
            mu_donor_matrix=mu_donor_matrix,
            config=config,
        )
        if residual.norm().item() < 1e-6:
            break

    return selected


# -----------------------
# Public API
# -----------------------

def design_with_pure_omp(
    base_embed_overlap: torch.Tensor,
    base_head_overlap: torch.Tensor,
    donor_embed_overlap: torch.Tensor,
    donor_head_overlap: torch.Tensor,
    shared_tokens: Sequence[str],
    mu_base: torch.Tensor,
    *,
    mu_donor: Optional[torch.Tensor] = None,
    mu_base_neg: Optional[torch.Tensor] = None,
    mu_donor_matrix: Optional[torch.Tensor] = None,
    config: Optional[DesignConfig] = None,
    penalized_support: bool = False,
    preserve_mu_donor_norm: bool = False,
    mu_donor_orthonormalize: bool = True,
    # NEW: view mixing
    dict_embed_weight: float = 0.5,
    dict_head_weight: float = 0.5,
    normalize_each_view: bool = True,
    normalize_final_row: bool = True,
) -> Tuple[DesignResult, Dict[str, Any]]:
    """
    OMP with a composite dictionary row per token:
      phi_j = normalize( w_e * e_j + w_h * h_j )
    All penalties operate on the same composite space; separate view-wise
    combinations are also returned for patching (embed/head).
    """
    if config is None:
        config = DesignConfig()
    config.validate()

    device = base_embed_overlap.device
    d_base = base_embed_overlap.shape[1]
    d_donor = donor_embed_overlap.shape[1]
    mu_base = mu_base.to(device=device, dtype=torch.float32)
    if mu_base.shape[0] != d_base:
        raise ValueError(
            f"µ_base dimension ({mu_base.shape[0]}) does not match base hidden size ({d_base})"
        )

    # Normalize overlaps if requested
    if config.normalize:
        base_embed_norm = _normalize_rows(base_embed_overlap)
        base_head_norm  = _normalize_rows(base_head_overlap)
        donor_embed_norm = _normalize_rows(donor_embed_overlap)
        donor_head_norm  = _normalize_rows(donor_head_overlap)
    else:
        base_embed_norm = base_embed_overlap
        base_head_norm  = base_head_overlap
        donor_embed_norm = donor_embed_overlap
        donor_head_norm  = donor_head_overlap

    # Build composite dictionaries
    use_embed = (dict_embed_weight != 0.0)
    use_head  = (dict_head_weight  != 0.0)

    dict_base = _combine_views(
        base_embed_norm, base_head_norm,
        use_embed=use_embed, use_head=use_head,
        w_embed=dict_embed_weight, w_head=dict_head_weight,
        normalize_each=normalize_each_view,
        normalize_final=normalize_final_row,
    )
    dict_donor = _combine_views(
        donor_embed_norm, donor_head_norm,
        use_embed=use_embed, use_head=use_head,
        w_embed=dict_embed_weight, w_head=dict_head_weight,
        normalize_each=normalize_each_view,
        normalize_final=normalize_final_row,
    )

    # Donor suppression matrix preparation
    mu_donor_matrix = (
        mu_donor_matrix.to(device=device, dtype=torch.float32) if mu_donor_matrix is not None else None
    )
    if mu_donor_matrix is not None:
        if mu_donor_matrix.shape[1] != d_donor:
            raise ValueError(
                f"µ_donor_matrix dimension ({mu_donor_matrix.shape[1]}) does not match donor hidden size ({d_donor})"
            )
        if config.normalize and not preserve_mu_donor_norm:
            mu_donor_matrix = _normalize_rows(mu_donor_matrix)
        if mu_donor_orthonormalize:
            mu_donor_matrix = _orthonormalize_rows(mu_donor_matrix)

    # Optional µ_donor vector
    mu_donor = mu_donor.to(device=device, dtype=torch.float32) if mu_donor is not None else None
    if mu_donor is not None and mu_donor.shape[0] != d_donor:
        raise ValueError(
            f"µ_donor dimension ({mu_donor.shape[0]}) does not match donor hidden size ({d_donor})"
        )

    # Optional µ_base_neg vector
    mu_base_neg = (
        mu_base_neg.to(device=device, dtype=torch.float32) if mu_base_neg is not None else None
    )
    if mu_base_neg is not None and mu_base_neg.shape[0] != d_base:
        raise ValueError(
            f"µ_base_neg dimension ({mu_base_neg.shape[0]}) does not match base hidden size ({d_base})"
        )

    # Support selection
    if penalized_support:
        support = _penalized_support(
            dict_base,
            dict_donor,
            mu_base,
            mu_donor=mu_donor,
            mu_base_neg=mu_base_neg,
            mu_donor_matrix=mu_donor_matrix,
            config=config,
        )
    else:
        # Greedy OMP on composite dictionary
        support_indices, _ = batch_omp(mu_base.unsqueeze(0), dict_base, config.k)
        support = support_indices[0].tolist()

    if not support:
        raise RuntimeError("OMP failed to select any support tokens; consider increasing k")

    B_sel = dict_base[support]
    D_sel = dict_donor[support]
    alpha, base_comb_phi, donor_comb_phi, residual = _solve_fixed_support(
        B_sel,
        D_sel,
        mu_base,
        mu_donor=mu_donor,
        mu_base_neg=mu_base_neg,
        mu_donor_matrix=mu_donor_matrix,
        config=config,
    )

    # Also produce view-wise linear combinations for patching
    base_sel_embed = base_embed_norm[support]
    base_sel_head  = base_head_norm[support]
    donor_sel_embed = donor_embed_norm[support]
    donor_sel_head  = donor_head_norm[support]

    base_embed_comb = alpha @ base_sel_embed    # (d,)
    base_head_comb  = alpha @ base_sel_head     # (d,)
    donor_embed_comb = alpha @ donor_sel_embed  # (d,)
    donor_head_comb  = alpha @ donor_sel_head   # (d,)

    # Scores for debugging/inspection (composite space)
    scores = dict_base @ mu_base

    # Objective computed in composite space
    objective = torch.dot(base_comb_phi, mu_base).item()
    if mu_donor is not None:
        objective -= config.lambda_penalty * torch.dot(donor_comb_phi, mu_donor).item()
    if mu_base_neg is not None:
        objective -= config.eta_penalty * torch.dot(base_comb_phi, mu_base_neg).item()
    if mu_donor_matrix is not None and config.lambda_penalty != 0.0:
        penalty_val = (donor_comb_phi @ mu_donor_matrix.T).pow(2).sum().item()
        objective -= config.lambda_penalty * penalty_val

    # Package result
    result = DesignResult(
        alpha=alpha.detach().cpu(),
        support_indices=support,
        support_tokens=[shared_tokens[idx] for idx in support],
        # Keep legacy names: store EMBEDDING-space vector here (for backwards compatibility)
        base_embedding=base_embed_comb.detach().cpu(),
        donor_embedding=donor_embed_comb.detach().cpu(),
        objective=objective,
        residual=residual.detach().cpu(),         # residual in composite (phi) space
        scores=scores.detach().cpu(),
    )

    # Verification: donor-side recovery in composite space
    verification: Dict[str, Any] = {
        "matches": False,
        "recovered_indices": None,
        "recovered_coeffs": None,
        "coefficient_cosine": None,
        "diff_percentages": None,
        "max_diff_percent": None,
        # Extra diagnostics we now expose:
        "dict_embed_weight": float(dict_embed_weight),
        "dict_head_weight": float(dict_head_weight),
        "normalize_each_view": bool(normalize_each_view),
        "normalize_final_row": bool(normalize_final_row),
    }

    donor_support, donor_coeffs = batch_omp(
        donor_comb_phi.unsqueeze(0), dict_donor, len(support)
    )
    recovered = donor_support[0].tolist()
    verification["recovered_indices"] = recovered
    verification["recovered_coeffs"] = donor_coeffs[0].tolist()

    support_set = set(support)
    recovered_set = set(recovered)
    support_match = (support_set == recovered_set)

    if support_match:
        recovered_map = {idx: coeff for idx, coeff in zip(recovered, donor_coeffs[0].tolist())}
        recovered_aligned = torch.tensor([recovered_map[idx] for idx in support], dtype=torch.float32, device=alpha.device)
        designed_alpha = alpha.to(dtype=torch.float32)
        cosine = F.cosine_similarity(recovered_aligned.unsqueeze(0), designed_alpha.unsqueeze(0), dim=1).item()
        verification["coefficient_cosine"] = cosine
        diffs: List[float] = []
        for a, b in zip(designed_alpha.tolist(), recovered_aligned.tolist()):
            denom = max(abs(a), abs(b), 1e-8)
            diffs.append(abs(b - a) / denom)
        verification["diff_percentages"] = diffs
        verification["max_diff_percent"] = max(diffs) if diffs else None
        verification["matches"] = max(diffs or [0.0]) <= 0.05

    # Attach view-wise combos so the patcher can set both lm_head & input embed
    # (They'll be present in the saved payload; DesignResult stays unchanged.)
    extras = {
        "base_head_embedding": base_head_comb.detach().cpu(),
        "donor_head_embedding": donor_head_comb.detach().cpu(),
        "base_phi_embedding": base_comb_phi.detach().cpu(),
        "donor_phi_embedding": donor_comb_phi.detach().cpu(),
    }

    return result, extras | verification


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure OMP sparse designer (composite dictionary: embedding + LM-head)."
    )
    parser.add_argument("--base-model", required=True, help="Base model repo id or path")
    parser.add_argument("--donor-model", required=True, help="Donor model repo id or path")
    parser.add_argument("--mu-base", help="Path to µ_base tensor (optional if --target-vector provided)")
    parser.add_argument("--mu-donor", help="Path to µ_donor tensor")
    parser.add_argument("--mu-base-neg", help="Path to µ_base_neg tensor")

    parser.add_argument(
        "--mu-donor-matrix",
        action="append",
        help="Donor suppression matrix spec: 'path[:N]' or 'path:key[:N]'",
    )
    parser.add_argument(
        "--preserve-mu-donor-norm",
        action="store_true",
        help="Skip row normalization for µ_donor matrices when overlaps are normalized",
    )
    parser.add_argument(
        "--no-mu-donor-orthonormalize",
        action="store_true",
        help="Disable orthonormalizing µ_donor matrices (preserve magnitudes from source)",
    )

    parser.add_argument("--target-vector", help="Direct target embedding (replaces µ_base)")
    parser.add_argument("--k", type=int, default=16, help="OMP sparsity level")
    parser.add_argument("--lambda-penalty", type=float, default=0.0, help="Weight for donor penalty term")
    parser.add_argument("--eta", type=float, default=0.0, help="Weight for base negative term")
    parser.add_argument("--gamma", type=float, default=1.0, help="Quadratic regularization for base reconstruction")
    parser.add_argument("--ridge", type=float, default=1e-4, help="Diagonal ridge term for linear solve")
    parser.add_argument("--no-normalize", action="store_true", help="Disable unit-norm scaling of overlap vectors")

    parser.add_argument(
        "--penalized-support",
        action="store_true",
        help="Use donor-penalized greedy selection during OMP support selection",
    )

    parser.add_argument(
        "--auto-mu-donor-from-overlap",
        action="store_true",
        help="If --mu-donor is absent, derive donor µ by ridge-regressing donor overlap on base overlap.",
    )
    parser.add_argument("--auto-mu-donor-ridge", type=float, default=1e-4)

    # NEW: view mixing options
    parser.add_argument("--dict-embed-weight", type=float, default=0.5,
                        help="Weight of input embedding rows in the composite dictionary")
    parser.add_argument("--dict-head-weight", type=float, default=0.5,
                        help="Weight of LM-head rows in the composite dictionary")
    parser.add_argument("--no-normalize-each-view", action="store_true",
                        help="Do not unit-normalize each view (embed/head) before mixing")
    parser.add_argument("--no-normalize-final-row", action="store_true",
                        help="Do not unit-normalize the final mixed row")

    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", help="Optional path to save the resulting design artifact (.pt)")
    parser.add_argument("--verify", action="store_true", help="Donor-side OMP recovery consistency (composite space)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    merge_options = MergeOptions(trust_remote_code=args.trust_remote_code, device=str(device))

    base_ref = _parse_model_ref(args.base_model)
    donor_ref = _parse_model_ref(args.donor_model)

    overlap = _build_overlap(base_ref, donor_ref, merge_options, device)
    shared_tokens: Sequence[str] = overlap["shared_tokens"]
    base_embed = overlap["base_embed_overlap"].to(device=device, dtype=torch.float32)
    base_head  = overlap["base_head_overlap"].to(device=device, dtype=torch.float32)
    donor_embed = overlap["donor_embed_overlap"].to(device=device, dtype=torch.float32)
    donor_head  = overlap["donor_head_overlap"].to(device=device, dtype=torch.float32)

    # Load targets
    if args.target_vector:
        mu_base = _load_vector(args.target_vector, device)
    elif args.mu_base:
        mu_base = _load_vector(args.mu_base, device)
    else:
        raise ValueError("Either --target-vector or --mu-base must be provided")

    mu_donor = _load_vector(args.mu_donor, device) if args.mu_donor else None
    # Optional auto µ_donor from overlaps
    if mu_donor is None and args.auto_mu_donor_from_overlap:
        # Fit in EMBEDDING space (could also blend; embed is safer for most stacks)
        B = base_embed if args.no_normalize else _normalize_rows(base_embed)
        D = donor_embed if args.no_normalize else _normalize_rows(donor_embed)
        Bt = B.transpose(0, 1)
        gram = Bt @ B
        ridge = float(args.auto_mu_donor_ridge)
        if ridge != 0.0:
            gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        BtD = Bt @ D
        W = torch.linalg.solve(gram, BtD)
        mu_donor = (mu_base @ W).to(device=device, dtype=torch.float32)
        logger.info("Derived µ_donor via overlap ridge map (ridge=%s), ||µ_donor||=%.6f", ridge, mu_donor.norm().item())

    mu_base_neg = _load_vector(args.mu_base_neg, device) if args.mu_base_neg else None

    # Optional donor suppression matrix
    mu_donor_matrix: Optional[torch.Tensor] = None
    if args.mu_donor_matrix:
        matrices = [_load_matrix(spec, device) for spec in args.mu_donor_matrix]
        mu_donor_matrix = torch.cat(matrices, dim=0)

    config = DesignConfig(
        k=args.k,
        lambda_penalty=args.lambda_penalty,
        eta_penalty=args.eta,
        gamma=args.gamma,
        ridge=args.ridge,
        normalize=not args.no_normalize,
    )

    result, verification = design_with_pure_omp(
        base_embed,
        base_head,
        donor_embed,
        donor_head,
        shared_tokens,
        mu_base,
        mu_donor=mu_donor,
        mu_base_neg=mu_base_neg,
        mu_donor_matrix=mu_donor_matrix,
        config=config,
        penalized_support=args.penalized_support,
        preserve_mu_donor_norm=args.preserve_mu_donor_norm,
        mu_donor_orthonormalize=not args.no_mu_donor_orthonormalize,
        dict_embed_weight=float(args.dict_embed_weight),
        dict_head_weight=float(args.dict_head_weight),
        normalize_each_view=not args.no_normalize_each_view,
        normalize_final_row=not args.no_normalize_final_row,
    )

    print("Selected support tokens:")
    for idx, (tok, coeff) in enumerate(zip(result.support_tokens, result.alpha.tolist())):
        print(f"  {idx:2d}: {tok!r} (coeff={coeff:+.6f})")
    print(f"Objective (composite) : {result.objective:.6f}")
    print(f"Residual norm (phi)   : {result.residual.norm().item():.6f}")

    if args.verify:
        matches = verification.get("matches")
        status = "matched" if matches else "did not match"
        cosine = verification.get("coefficient_cosine")
        cos_str = f", coeff cosine={cosine:.4f}" if cosine is not None else ""
        print(f"Donor-side OMP verification {status}{cos_str}. Support {verification.get('recovered_indices')}")

    # Save payload (include head vectors for the patcher)
    if args.output:
        payload: Dict[str, Any] = {
            "alpha": result.alpha,
            "support_indices": result.support_indices,
            "support_tokens": result.support_tokens,
            "base_embedding": result.base_embedding,       # EMBEDDING-space (legacy)
            "donor_embedding": result.donor_embedding,     # EMBEDDING-space (legacy)
            "objective": result.objective,
            "residual": result.residual,
            "scores": result.scores,
            "verification": verification,
        }
        # Extras produced by design_with_pure_omp
        for k in ("base_head_embedding", "donor_head_embedding", "base_phi_embedding", "donor_phi_embedding"):
            if k in verification and isinstance(verification[k], torch.Tensor):
                payload[k] = verification[k]
        torch.save(payload, args.output)
        print(f"Saved design artifact to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
