from __future__ import annotations

"""
Unified shared-basis designer for FOCUS / CLP / WECHSEL.

Pipeline (per method):

  1. Build shared anchors T from base/donor using SharedBasisTransplanter.
  2. Choose a sparse anchor support S (top-K in base space) and solve a
     least-squares fit:
         sum_{j in S} alpha_j * base_anchor_j  ≈  target_base_vec
  3. Convert alpha_j -> non-negative weights w_j in a method-specific way:
       - focus / wechsel:  w = softmax(beta * alpha)
       - clp            :  w_j ∝ max(0, alpha_j)
  4. Form donor init vector e_d0 = sum_j w_j * donor_anchor_j.
  5. Apply PCA-based donor penalty:
         (I + λ P^T P) e_d_init = e_d0
  6. Use e_d_init as initialisation for a short gradient descent that optimises
     the *actual* shared-basis transplant approximation (FOCUS/CLP/WECHSEL) in
     base space plus donor inertness and norm penalties.

The artifact contains:
  - donor_embedding      : donor-side input embedding for the breaker token
  - donor_head_embedding : donor-side LM-head vector (same as input if tied)
  - config               : dict with method, anchor_topk, focus_beta, lambda_penalty,
                           target_scale; used by apply_sb_transplant to
                           re-create the same operator.
"""

import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from non_mergekit_methods.sb_transplant import SharedBasisTransplanter  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_vector(payload, index: int) -> torch.Tensor:
    """
    Extract a single target vector from a saved .pt payload.
    Supports:
      - bare 1D/2D tensor
      - dict with 'base_embedding' / 'base_phi_embedding' / 'embedding'
      - dict with matrix-like keys: 'base_vectors' / 'vectors' / 'data' / 'matrix'
    """
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 1:
            return payload
        if payload.ndim == 2:
            if not (0 <= index < payload.shape[0]):
                raise ValueError(
                    f"target-vector-index {index} out of range for tensor with "
                    f"{payload.shape[0]} rows"
                )
            return payload[index]
        raise ValueError(f"Unsupported tensor rank {payload.ndim} for target payload")

    if isinstance(payload, dict):
        for key in ("base_embedding", "base_phi_embedding", "embedding"):
            if key in payload:
                return torch.as_tensor(payload[key])
        for key in ("base_vectors", "vectors", "data", "matrix"):
            if key in payload:
                mat = torch.as_tensor(payload[key])
                if mat.ndim == 1:
                    return mat
                if mat.ndim == 2:
                    if not (0 <= index < mat.shape[0]):
                        raise ValueError(
                            f"target-vector-index {index} out of range for matrix with "
                            f"{mat.shape[0]} rows in key '{key}'"
                        )
                    return mat[index]
        raise ValueError("Unable to locate target vector inside payload")
    raise ValueError("Unsupported target payload type for target vector")


def _load_matrix_spec(spec: str, device: torch.device) -> Optional[torch.Tensor]:
    """
    Flexible 'path[:key[:rows]]' loader for donor PCA matrices.

    Examples:
      - 'donor_subspace.pt'
      - 'donor_subspace.pt:components:256'
      - 'file.pt:matrix'
    """
    parts = spec.split(":")
    file_path = parts[0]
    key: Optional[str] = None
    rows: Optional[int] = None

    if len(parts) >= 2 and parts[1] != "":
        try:
            rows = int(parts[1])
        except ValueError:
            key = parts[1]
    if len(parts) >= 3 and parts[2] != "":
        rows = int(parts[2])

    payload = torch.load(file_path, map_location="cpu")
    matrix: Optional[torch.Tensor] = None

    if isinstance(payload, dict):
        if key and key in payload:
            matrix = torch.as_tensor(payload[key], dtype=torch.float32)
        else:
            for candidate in (
                "components",
                "matrix",
                "mat",
                "vectors",
                "centers",
                "cluster_centers",
                "data",
            ):
                if candidate in payload:
                    matrix = torch.as_tensor(payload[candidate], dtype=torch.float32)
                    break
            if matrix is None:
                for v in payload.values():
                    if isinstance(v, torch.Tensor):
                        matrix = v.to(torch.float32)
                        break
    else:
        matrix = torch.as_tensor(payload, dtype=torch.float32)

    if matrix is None:
        return None
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if rows is not None:
        if rows <= 0 or rows > matrix.shape[0]:
            raise ValueError(
                f"Requested {rows} rows from matrix with shape {tuple(matrix.shape)}"
            )
        matrix = matrix[:rows]
    return matrix.to(device=device, dtype=torch.float32).contiguous()


def _load_donor_pca_components(
    specs: Optional[List[str]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not specs:
        return None
    mats: List[torch.Tensor] = []
    for spec in specs:
        try:
            mat = _load_matrix_spec(spec, device)
        except Exception as exc:
            print(f"[sb_omp] warning: failed to load donor PCA matrix from {spec}: {exc}")
            continue
        if mat is not None and mat.numel() != 0:
            mats.append(mat)
    if not mats:
        return None
    P = torch.cat(mats, dim=0)
    # simple row-normalisation; alignment doesn’t require full orthonormality here
    P = F.normalize(P, p=2, dim=1)
    print(f"[sb_omp] loaded donor PCA components: {P.shape}")
    return P


# ---------------------------------------------------------------------------
# Differentiable transplant (matches sb_transplant operators)
# ---------------------------------------------------------------------------

class DifferentiableTransplant(nn.Module):
    """
    Wraps SharedBasisTransplanter plus full LM heads for base/donor.

    We do *not* backprop through full LM forward passes; we only need:
      - input embeddings / anchors,
      - LM-head weights (to detect tied/untied),
      - WECHSEL alignment map for input & head spaces.
    """

    def __init__(
        self,
        base_model_name: str,
        donor_model_name: str,
        transplanter: SharedBasisTransplanter,
        *,
        device: torch.device,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.device = device
        self.transplanter = transplanter

        print("[sb_omp] Loading CausalLM heads to detect ties …")
        base_clm = AutoModelForCausalLM.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code
        ).to(device)
        donor_clm = AutoModelForCausalLM.from_pretrained(
            donor_model_name, trust_remote_code=trust_remote_code
        ).to(device)
        base_clm.eval()
        donor_clm.eval()

        base_in = base_clm.get_input_embeddings().weight.detach()
        base_out = base_clm.get_output_embeddings().weight.detach()
        donor_in = donor_clm.get_input_embeddings().weight.detach()
        donor_out = donor_clm.get_output_embeddings().weight.detach()

        self.tied_base = base_in.data_ptr() == base_out.data_ptr()
        self.tied_donor = donor_in.data_ptr() == donor_out.data_ptr()

        if self.tied_base:
            print("[sb_omp] Base appears tied (embed/head share weights).")
        else:
            print("[sb_omp] Base appears untied.")

        # shared anchors (indices) from transplanter
        base_idx, donor_idx = transplanter.get_shared_vocab()

        # input anchors
        self.register_buffer(
            "base_in_anchors",
            transplanter.base_embeddings[torch.tensor(base_idx, device=device)].clone(),
        )
        self.register_buffer(
            "donor_in_anchors",
            transplanter.donor_embeddings[torch.tensor(donor_idx, device=device)].clone(),
        )
        self.register_buffer(
            "base_in_anchors_norm",
            F.normalize(self.base_in_anchors, p=2, dim=1),
        )
        self.register_buffer(
            "donor_in_anchors_norm",
            F.normalize(self.donor_in_anchors, p=2, dim=1),
        )

        # full base embeddings (for WECHSEL)
        self.register_buffer(
            "base_full_embeddings", transplanter.base_embeddings.clone()
        )
        self.register_buffer(
            "base_full_embeddings_norm",
            F.normalize(self.base_full_embeddings, p=2, dim=1),
        )

        # LM-head anchors
        self.register_buffer(
            "base_head_anchors",
            base_out[torch.tensor(base_idx, device=device)].clone(),
        )
        self.register_buffer(
            "donor_head_anchors",
            donor_out[torch.tensor(donor_idx, device=device)].clone(),
        )
        self.register_buffer(
            "base_head_anchors_norm",
            F.normalize(self.base_head_anchors, p=2, dim=1),
        )
        self.register_buffer(
            "donor_head_anchors_norm",
            F.normalize(self.donor_head_anchors, p=2, dim=1),
        )

        # full base heads (for WECHSEL head view)
        self.register_buffer(
            "base_full_head_embeddings",
            base_out.clone(),
        )
        self.register_buffer(
            "base_full_head_embeddings_norm",
            F.normalize(self.base_full_head_embeddings, p=2, dim=1),
        )

        # WECHSEL map in input space
        base_idx_list, donor_idx_list = transplanter.get_shared_vocab()
        map_in = transplanter.compute_alignment_map(base_idx_list, donor_idx_list)
        self.register_buffer("wechsel_map_in", map_in.clone())
        self.register_buffer("mu_x_in", transplanter.mu_x.clone())
        self.register_buffer("mu_y_in", transplanter.mu_y.clone())

        # WECHSEL map in head space
        if not self.tied_base or not self.tied_donor:
            X = self.donor_head_anchors
            Y = self.base_head_anchors
            mu_x_out = X.mean(0)
            mu_y_out = Y.mean(0)
            Xc = X - mu_x_out
            Yc = Y - mu_y_out
            if X.shape[1] == Y.shape[1]:
                U, _, Vt = torch.linalg.svd(Yc.t() @ Xc)
                R = U @ Vt
                map_out = R.t()
            else:
                result = torch.linalg.lstsq(Xc, Yc)
                map_out = result.solution
            self.register_buffer("wechsel_map_out", map_out)
            self.register_buffer("mu_x_out", mu_x_out)
            self.register_buffer("mu_y_out", mu_y_out)
        else:
            self.register_buffer("wechsel_map_out", map_in.clone())
            self.register_buffer("mu_x_out", transplanter.mu_x.clone())
            self.register_buffer("mu_y_out", transplanter.mu_y.clone())

        del base_clm, donor_clm
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unified designer
# ---------------------------------------------------------------------------

class SharedBasisDesigner:
    """
    One designer that targets FOCUS / CLP / WECHSEL shared-basis transplant.

    It uses:
      - OMP-like LS init over base anchors.
      - PCA-based donor suppression (lambda_penalty).
      - Gradient polish that optimises the same sb_transplant operator used
        later during WECHSEL/FOCUS/CLP transplant.
    """

    def __init__(
        self,
        base_model: str,
        donor_model: str,
        *,
        mu_donor_matrix_specs: Optional[List[str]] = None,
        mu_base_vectors_path: Optional[str] = None,
        device: str = "cpu",
        trust_remote_code: bool = False,
        anchor_topk: int = 32,
        focus_beta: float = 10.0,
    ):
        self.base_model = base_model
        self.donor_model = donor_model
        self.device = torch.device(device)
        self.trust_remote_code = trust_remote_code

        self.anchor_topk = int(anchor_topk)
        self.focus_beta = float(focus_beta)

        # Shared-basis transplant object (only embeddings)
        self.transplanter = SharedBasisTransplanter(
            base_model,
            donor_model,
            device=device,
            trust_remote_code=trust_remote_code,
            top_k=self.anchor_topk,
            focus_beta=self.focus_beta,
        )

        # Differentiable transplant wrapper for simulation
        self.diff = DifferentiableTransplant(
            base_model,
            donor_model,
            self.transplanter,
            device=self.device,
            trust_remote_code=trust_remote_code,
        ).to(self.device)

        # Grab full embedding matrices for convenience
        self.base_embeddings = self.transplanter.base_embeddings.to(
            self.device, dtype=torch.float32
        )
        self.donor_embeddings = self.transplanter.donor_embeddings.to(
            self.device, dtype=torch.float32
        )
        self.donor_dim = self.donor_embeddings.shape[1]
        self.donor_med_norm = float(self.donor_embeddings.norm(dim=1).median().item())

        # donor PCA suppression subspace
        self.donor_pca_components = _load_donor_pca_components(
            mu_donor_matrix_specs, self.device
        )

        # optional base hidden-state matrix
        self.base_mu_matrix: Optional[torch.Tensor] = None
        if mu_base_vectors_path is not None:
            try:
                payload = torch.load(mu_base_vectors_path, map_location="cpu")
                if isinstance(payload, dict) and "base_vectors" in payload:
                    self.base_mu_matrix = torch.as_tensor(
                        payload["base_vectors"], dtype=torch.float32
                    ).to(self.device)
                    print(
                        f"[sb_omp] loaded base_vectors from {mu_base_vectors_path}: "
                        f"{self.base_mu_matrix.shape}"
                    )
            except Exception as exc:
                print(
                    f"[sb_omp] warning: failed to load base_vectors from "
                    f"{mu_base_vectors_path}: {exc}"
                )

    # ---------- donor penalties ----------

    def _apply_donor_penalty(
        self, e_d0: torch.Tensor, lambda_penalty: float
    ) -> torch.Tensor:
        """
        Closed-form PCA shrinkage (I + λ P^T P)^(-1) e_d0 for init.
        """
        if self.donor_pca_components is None or lambda_penalty <= 0.0:
            return e_d0

        P = self.donor_pca_components.to(e_d0.device, dtype=e_d0.dtype)  # (m, d)
        d = e_d0.shape[0]
        I = torch.eye(d, device=e_d0.device, dtype=e_d0.dtype)
        A = I + lambda_penalty * (P.t() @ P)
        return torch.linalg.solve(A, e_d0)

    def _donor_inert_loss(self, e: torch.Tensor) -> torch.Tensor:
        if self.donor_pca_components is None:
            return e.new_tensor(0.0)
        P = self.donor_pca_components.to(e.device, dtype=e.dtype)
        proj = P @ e
        return (proj * proj).sum()

    def _norm_loss(self, e: torch.Tensor) -> torch.Tensor:
        return (e.norm() - self.donor_med_norm) ** 2

    def _base_mu_l2(self, base_vec: torch.Tensor) -> torch.Tensor:
        if self.base_mu_matrix is None:
            return base_vec.new_tensor(0.0)
        diffs = self.base_mu_matrix - base_vec.unsqueeze(0)
        return diffs.norm(dim=1).mean()

    # ---------- shared-basis simulation (matches sb_transplant) ----------

    def _compute_base_from_donor(
        self,
        method: str,
        donor_in_vec: torch.Tensor,
        donor_head_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate the transplanted base input & head embeddings for a single
        donor embedding pair, using the same geometry as SharedBasisTransplanter.
        """
        dt = self.diff
        method = method.lower()

        if method == "wechsel":
            # input view: donor -> base via WECHSEL + KNN over full base vocab
            proxy_in = (donor_in_vec - dt.mu_x_in) @ dt.wechsel_map_in + dt.mu_y_in
            proxy_in = proxy_in / (proxy_in.norm() + 1e-12)
            sims = dt.base_full_embeddings_norm @ proxy_in  # (V_b,)
            k = min(self.anchor_topk, sims.shape[0])
            vals, idxs = torch.topk(sims, k)
            w = F.softmax(self.focus_beta * vals, dim=0)
            base_sel = dt.base_full_embeddings[idxs]
            e_b_in = (w.unsqueeze(1) * base_sel).sum(dim=0)

            if dt.tied_base:
                e_b_head = e_b_in
            else:
                proxy_h = (donor_head_vec - dt.mu_x_out) @ dt.wechsel_map_out + dt.mu_y_out
                proxy_h = proxy_h / (proxy_h.norm() + 1e-12)
                sims_h = dt.base_full_head_embeddings_norm @ proxy_h
                k_h = min(self.anchor_topk, sims_h.shape[0])
                vals_h, idxs_h = torch.topk(sims_h, k_h)
                w_h = F.softmax(self.focus_beta * vals_h, dim=0)
                head_sel = dt.base_full_head_embeddings[idxs_h]
                e_b_head = (w_h.unsqueeze(1) * head_sel).sum(dim=0)

        elif method == "focus":
            # focus: donor->anchors cosine, then mix base anchors
            e_hat = donor_in_vec / (donor_in_vec.norm() + 1e-12)
            sims = dt.donor_in_anchors_norm @ e_hat
            k = min(self.anchor_topk, sims.shape[0])
            vals, idxs = torch.topk(sims, k)
            w = F.softmax(self.focus_beta * vals, dim=0)
            base_sel = dt.base_in_anchors[idxs]
            e_b_in = (w.unsqueeze(1) * base_sel).sum(dim=0)

            if dt.tied_base:
                e_b_head = e_b_in
            else:
                e_hat_h = donor_head_vec / (donor_head_vec.norm() + 1e-12)
                sims_h = dt.donor_head_anchors_norm @ e_hat_h
                k_h = min(self.anchor_topk, sims_h.shape[0])
                vals_h, idxs_h = torch.topk(sims_h, k_h)
                w_h = F.softmax(self.focus_beta * vals_h, dim=0)
                head_sel = dt.base_head_anchors[idxs_h]
                e_b_head = (w_h.unsqueeze(1) * head_sel).sum(dim=0)

        elif method == "clp":
            # CLP: ReLU(sim) over anchors, linear normalized
            e_hat = donor_in_vec / (donor_in_vec.norm() + 1e-12)
            sims = dt.donor_in_anchors_norm @ e_hat
            k = min(self.anchor_topk, sims.shape[0])
            vals, idxs = torch.topk(sims, k)
            vals = F.relu(vals)
            denom = vals.sum().clamp_min(1e-9)
            w = vals / denom
            base_sel = dt.base_in_anchors[idxs]
            e_b_in = (w.unsqueeze(1) * base_sel).sum(dim=0)

            if dt.tied_base:
                e_b_head = e_b_in
            else:
                e_hat_h = donor_head_vec / (donor_head_vec.norm() + 1e-12)
                sims_h = dt.donor_head_anchors_norm @ e_hat_h
                k_h = min(self.anchor_topk, sims_h.shape[0])
                vals_h, idxs_h = torch.topk(sims_h, k_h)
                vals_h = F.relu(vals_h)
                denom_h = vals_h.sum().clamp_min(1e-9)
                w_h = vals_h / denom_h
                head_sel = dt.base_head_anchors[idxs_h]
                e_b_head = (w_h.unsqueeze(1) * head_sel).sum(dim=0)
        else:
            raise ValueError(f"Unknown method {method!r}")

        return e_b_in, e_b_head

    # ---------- OMP-like init over base anchors ----------

    def _omp_like_init(
        self,
        target_base_vec: torch.Tensor,
        method: str,
        lambda_penalty: float,
    ) -> torch.Tensor:
        """
        OMP-ish init:
          - pick anchor_topk base anchors by cosine to target.
          - solve LS on that support.
          - turn alpha->weights with method-specific nonlinearity.
          - map weights onto donor anchors, apply PCA penalty.
        """
        dt = self.diff
        B = dt.base_in_anchors.to(self.device)          # (N, d_b)
        B_norm = dt.base_in_anchors_norm.to(self.device)
        D = dt.donor_in_anchors.to(self.device)         # (N, d_d)

        t = target_base_vec.to(self.device, dtype=torch.float32)
        t_dir = F.normalize(t, p=2, dim=0)

        sims = B_norm @ t_dir  # (N,)
        K = min(self.anchor_topk, sims.shape[0])
        vals, idxs = torch.topk(sims, K)
        B_sel = B[idxs]        # (K, d_b)
        D_sel = D[idxs]        # (K, d_d)

        # LS solve: argmin ||B_sel^T alpha - t||
        Gram = B_sel @ B_sel.T
        Gram = Gram + 1e-8 * torch.eye(K, device=self.device, dtype=Gram.dtype)
        rhs = B_sel @ t
        alpha = torch.linalg.solve(Gram, rhs)  # (K,)

        method = method.lower()
        if method in {"focus", "wechsel"}:
            w = F.softmax(self.focus_beta * alpha, dim=0)
        elif method == "clp":
            pos = F.relu(alpha)
            s = pos.sum()
            if s <= 1e-9:
                w = torch.full_like(pos, 1.0 / float(pos.numel()))
            else:
                w = pos / s
        else:
            raise ValueError(f"Unknown method {method!r}")

        e_d0 = (w.unsqueeze(1) * D_sel).sum(dim=0)  # donor mix
        e_d = self._apply_donor_penalty(e_d0, lambda_penalty=lambda_penalty)
        norm = e_d.norm()
        if norm > 0:
            e_d = e_d / norm * self.donor_med_norm
        return e_d

    # ---------- main design ----------

    def design(
        self,
        *,
        target_vectors: Optional[Dict[str, torch.Tensor]] = None,
        target_base_token: Optional[str] = None,
        method: str = "focus",
        lambda_penalty: float = 10.0,
        steps: int = 2000,
        lr: float = 0.01,
        target_scale: float = 1.0,
        mu_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Optimise a donor embedding such that the shared-basis transplant
        (FOCUS/CLP/WECHSEL) yields a base embedding close to the target.

        Parameters
        ----------
        target_vectors:
            dict containing 'base_embedding' when target_base_token is None.
        target_base_token:
            base vocab token whose embedding is to be mimicked.
        method:
            'focus', 'clp', or 'wechsel'.
        lambda_penalty:
            strength of donor PCA inertness + norm regularisation.
        steps:
            gradient steps.
        lr:
            learning rate.
        target_scale:
            multiplicative scale applied to the target base embedding.
        mu_weight:
            weight for optional base hidden-state L2 term (if base_vectors loaded).
        """
        method = method.lower()
        if method not in {"focus", "clp", "wechsel"}:
            raise ValueError(f"Unknown method {method!r}")

        # --- resolve target base embedding ---
        if target_base_token is None and (
            target_vectors is None or "base_embedding" not in target_vectors
        ):
            raise ValueError(
                "Either target_base_token or target_vectors['base_embedding'] must be provided"
            )

        if target_base_token is not None:
            tok_id = self.transplanter.base_tokenizer.convert_tokens_to_ids(
                target_base_token
            )
            if tok_id == self.transplanter.base_tokenizer.unk_token_id:
                raise ValueError(
                    f"Target base token {target_base_token!r} is unknown to base tokenizer"
                )
            base_target_vec = self.base_embeddings[tok_id].to(
                self.device, dtype=torch.float32
            )
        else:
            base_target_vec = torch.as_tensor(
                target_vectors["base_embedding"],
                dtype=torch.float32,
                device=self.device,
            )

        base_target_vec = base_target_vec * float(target_scale)

        # For untied base, optionally use real head vector when token is given
        if not self.diff.tied_base and target_base_token is not None:
            base_clm = AutoModelForCausalLM.from_pretrained(
                self.base_model, trust_remote_code=self.trust_remote_code
            ).to(self.device)
            base_clm.eval()
            base_head_weights = (
                base_clm.get_output_embeddings().weight.detach().to(
                    self.device, dtype=torch.float32
                )
            )
            base_head_vec = base_head_weights[tok_id] * float(target_scale)
            del base_clm
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            base_head_vec = base_target_vec

        # --- OMP-like init for donor embedding ---
        with torch.no_grad():
            init_vec = self._omp_like_init(
                target_base_vec=base_target_vec,
                method=method,
                lambda_penalty=lambda_penalty,
            )

        donor_in_param = nn.Parameter(init_vec.clone())
        if self.diff.tied_donor:
            donor_head_param = donor_in_param
            params = [donor_in_param]
        else:
            donor_head_param = nn.Parameter(init_vec.clone())
            params = [donor_in_param, donor_head_param]

        optim = torch.optim.Adam(params, lr=lr)

        # --- gradient loop ---
        for step in range(steps):
            optim.zero_grad()

            e_b_in, e_b_head = self._compute_base_from_donor(
                method, donor_in_param, donor_head_param
            )

            loss_in = F.mse_loss(e_b_in, base_target_vec)
            if self.diff.tied_base:
                loss_head = e_b_in.new_tensor(0.0)
            else:
                loss_head = F.mse_loss(e_b_head, base_head_vec)
            loss_align = loss_in + loss_head

            inert_in = self._donor_inert_loss(donor_in_param)
            inert_head = (
                inert_in
                if donor_head_param is donor_in_param
                else self._donor_inert_loss(donor_head_param)
            )
            norm_in = self._norm_loss(donor_in_param)
            norm_head = (
                norm_in
                if donor_head_param is donor_in_param
                else self._norm_loss(donor_head_param)
            )
            loss_reg = inert_in + inert_head + norm_in + norm_head

            if mu_weight > 0.0:
                loss_mu = mu_weight * self._base_mu_l2(e_b_in)
            else:
                loss_mu = e_b_in.new_tensor(0.0)

            loss = loss_align + lambda_penalty * loss_reg + loss_mu
            loss.backward()
            optim.step()

        # final normalisation to typical donor norm
        with torch.no_grad():
            d_in = donor_in_param.detach()
            n = d_in.norm()
            if n > 0:
                d_in = d_in / n * self.donor_med_norm
            donor_in_final = d_in.cpu()
            if donor_head_param is donor_in_param:
                donor_head_final = donor_in_final.clone()
            else:
                d_head = donor_head_param.detach()
                n_h = d_head.norm()
                if n_h > 0:
                    d_head = d_head / n_h * self.donor_med_norm
                donor_head_final = d_head.cpu()

        return {
            "donor_embedding": donor_in_final,
            "donor_head_embedding": donor_head_final,
            "config": {
                "method": method,
                "anchor_topk": self.anchor_topk,
                "focus_beta": self.focus_beta,
                "lambda_penalty": float(lambda_penalty),
                "target_scale": float(target_scale),
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified shared-basis designer for FOCUS / CLP / WECHSEL."
    )
    p.add_argument("--base-model", required=True)
    p.add_argument("--donor-model", required=True)

    p.add_argument(
        "--target-vector-path",
        default=None,
        help="Path to .pt target vector (e.g. mu_base.pt / base_vectors.pt).",
    )
    p.add_argument(
        "--target-vector-index",
        type=int,
        default=0,
        help="Row index when target-vector-path contains multiple rows.",
    )
    p.add_argument(
        "--target-base-token",
        default=None,
        help="Base vocab token to mimic; overrides --target-vector-path.",
    )

    p.add_argument(
        "--mu-base-vectors",
        default=None,
        help="Optional base_vectors.pt for extra mu-based regularisation.",
    )
    p.add_argument(
        "--mu-weight",
        type=float,
        default=0.0,
        help="Weight for base hidden-state L2 term.",
    )

    p.add_argument(
        "--mu-donor-matrix",
        action="append",
        default=None,
        help="Repeatable donor PCA matrix spec 'path[:key[:rows]]'.",
    )

    p.add_argument(
        "--method",
        choices=["focus", "clp", "wechsel"],
        default="focus",
    )
    p.add_argument(
        "--lambda-penalty",
        type=float,
        default=10.0,
        help="Weight for donor inertness (PCA + norm) regularisation.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Gradient steps for polish.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate.",
    )
    p.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale for the target base embedding.",
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Top-K anchors/neighbors used in shared-basis rules.",
    )
    p.add_argument(
        "--focus-beta",
        type=float,
        default=10.0,
        help="Softmax temperature for FOCUS/WECHSEL.",
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--output",
        default="optimized_token.pt",
        help="Output artifact path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_base_token is None and args.target_vector_path is None:
        raise ValueError(
            "Either --target-base-token or --target-vector-path must be provided"
        )

    target_dict: Optional[Dict[str, torch.Tensor]] = None
    if args.target_base_token is None:
        payload = torch.load(args.target_vector_path, map_location="cpu")
        vec = _extract_vector(payload, args.target_vector_index)
        target_dict = {"base_embedding": vec}

    designer = SharedBasisDesigner(
        args.base_model,
        args.donor_model,
        mu_donor_matrix_specs=args.mu_donor_matrix,
        mu_base_vectors_path=args.mu_base_vectors,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        anchor_topk=args.top_k,
        focus_beta=args.focus_beta,
    )

    result = designer.design(
        target_vectors=target_dict,
        target_base_token=args.target_base_token,
        method=args.method,
        lambda_penalty=args.lambda_penalty,
        steps=args.steps,
        lr=args.lr,
        target_scale=args.target_scale,
        mu_weight=args.mu_weight,
    )

    torch.save(result, args.output)
    print(f"[sb_omp] Saved shared-basis design artifact to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
