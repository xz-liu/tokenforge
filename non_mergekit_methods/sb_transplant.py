from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


class SharedBasisTransplanter:
    """
    Shared-basis tokenizer transplant for WECHSEL / FOCUS / CLP / VIPI.

    This class is deliberately lightweight:

      * Loads base/donor models only long enough to grab the input embeddings,
        then frees the model modules (to reduce GPU memory).
      * Uses a shared anchor set T = V_b ∩ V_d via exact token string match.
      * Implements:
          - WECHSEL: donor->base linear map + KNN over full base vocab.
          - FOCUS  : KNN over shared anchors in donor space, mixing base anchors.
          - CLP    : ReLU + linear-normalized similarities over anchors.
          - VIPI   : subword mean in base vocab.

    Hyperparameters:
      - top_k      : number of neighbors/anchors (same K as designer).
      - focus_beta : softmax temperature for WECHSEL/FOCUS.
      - knn_batch_size : batch size over tokens for WECHSEL KNN to avoid
                         materializing giant (M x |V_base|) matrices.
    """

    def __init__(
        self,
        base_model_name: str,
        donor_model_name: str,
        *,
        device: str = "cpu",
        trust_remote_code: bool = False,
        top_k: int = 32,
        focus_beta: float = 10.0,
        knn_batch_size: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.trust_remote_code = trust_remote_code
        self.top_k = int(top_k)
        self.focus_beta = float(focus_beta)
        self.knn_batch_size = int(knn_batch_size)

        # --- tokenizers ---
        print(f"[sb_transplant] Loading base tokenizer: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code
        )
        print(f"[sb_transplant] Loading donor tokenizer: {donor_model_name}")
        self.donor_tokenizer = AutoTokenizer.from_pretrained(
            donor_model_name, trust_remote_code=trust_remote_code
        )

        # --- grab input embeddings, free models ---
        print(f"[sb_transplant] Loading base model embeddings …")
        base_model = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code
        ).to(self.device)
        base_model.eval()
        base_embed_weight = base_model.get_input_embeddings().weight.detach().to(
            self.device, dtype=torch.float32
        )
        print(f"[sb_transplant] Loading donor model embeddings …")
        donor_model = AutoModel.from_pretrained(
            donor_model_name, trust_remote_code=trust_remote_code
        ).to(self.device)
        donor_model.eval()
        donor_embed_weight = donor_model.get_input_embeddings().weight.detach().to(
            self.device, dtype=torch.float32
        )

        self.base_embeddings = base_embed_weight.clone()
        self.donor_embeddings = donor_embed_weight.clone()
        self.d_base = self.base_embeddings.shape[1]
        self.d_donor = self.donor_embeddings.shape[1]

        # free big modules
        del base_model, donor_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # placeholders for WECHSEL alignment
        self._map_ready = False
        self.mu_x = None
        self.mu_y = None
        self.map_matrix = None

        # cache shared vocab indices
        self._shared_base_idx: Optional[List[int]] = None
        self._shared_donor_idx: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # vocab utilities
    # ------------------------------------------------------------------

    def get_shared_vocab(self) -> Tuple[List[int], List[int]]:
        """
        Return (base_indices, donor_indices) for shared tokens.
        """
        if self._shared_base_idx is not None and self._shared_donor_idx is not None:
            return self._shared_base_idx, self._shared_donor_idx

        base_vocab = self.base_tokenizer.get_vocab()
        donor_vocab = self.donor_tokenizer.get_vocab()
        id2token_donor = {v: k for k, v in donor_vocab.items()}

        base_indices: List[int] = []
        donor_indices: List[int] = []
        shared = 0
        for donor_id, token in id2token_donor.items():
            if token in base_vocab:
                base_id = base_vocab[token]
                base_indices.append(base_id)
                donor_indices.append(donor_id)
                shared += 1
        print(f"[sb_transplant] Found {shared} shared tokens.")
        self._shared_base_idx = base_indices
        self._shared_donor_idx = donor_indices
        return base_indices, donor_indices

    def get_missing_vocab(self) -> List[str]:
        """
        Tokens in donor but not in base (donor-only / transplant targets).
        """
        base_vocab = self.base_tokenizer.get_vocab()
        donor_vocab = self.donor_tokenizer.get_vocab()
        base_tokens = set(base_vocab.keys())
        missing = [tok for tok in donor_vocab.keys() if tok not in base_tokens]
        print(f"[sb_transplant] Donor-only tokens (missing in base): {len(missing)}")
        return missing

    # ------------------------------------------------------------------
    # WECHSEL alignment (donor -> base linear map)
    # ------------------------------------------------------------------

    def compute_alignment_map(
        self, base_idxs: List[int], donor_idxs: List[int]
    ) -> torch.Tensor:
        """
        Compute donor->base linear map via Procrustes (dims match) or LS.
        """
        X = self.donor_embeddings[torch.tensor(donor_idxs, device=self.device)]
        Y = self.base_embeddings[torch.tensor(base_idxs, device=self.device)]

        self.mu_x = X.mean(0)
        self.mu_y = Y.mean(0)
        Xc = X - self.mu_x
        Yc = Y - self.mu_y

        if self.d_base == self.d_donor:
            print("[sb_transplant] Alignment: Procrustes (dims match).")
            U, _, Vt = torch.linalg.svd(Yc.t() @ Xc)
            R = U @ Vt
            self.map_matrix = R.t()
        else:
            print("[sb_transplant] Alignment: least-squares (dims differ).")
            result = torch.linalg.lstsq(Xc, Yc)
            self.map_matrix = result.solution

        self._map_ready = True
        return self.map_matrix

    def _ensure_alignment_map(self) -> None:
        if self._map_ready:
            return
        b_idx, d_idx = self.get_shared_vocab()
        self.compute_alignment_map(b_idx, d_idx)

    def _project_vectors_to_base(self, donor_vectors: torch.Tensor) -> torch.Tensor:
        """
        donor_vectors: (M, d_donor) -> base space (M, d_base) using WECHSEL map.
        """
        self._ensure_alignment_map()
        if donor_vectors.ndim == 1:
            donor_vectors = donor_vectors.unsqueeze(0)
        donor_vectors = donor_vectors.to(self.device)
        vecs_centered = donor_vectors - self.mu_x
        return vecs_centered @ self.map_matrix + self.mu_y

    # ------------------------------------------------------------------
    # donor-vector preparation
    # ------------------------------------------------------------------

    def _prepare_donor_vectors(
        self,
        tokens: Optional[List[str]],
        donor_vectors: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Collect donor vectors for given tokens (or donor-only vocab if None).
        Returns (vectors, actual_tokens) where vectors[i] corresponds to actual_tokens[i].
        """
        token_list = tokens if tokens is not None else self.get_missing_vocab()
        vectors: List[torch.Tensor] = []
        actual_tokens: List[str] = []

        donor_vocab = self.donor_tokenizer.get_vocab()
        for tok in token_list:
            vec: Optional[torch.Tensor] = None
            if donor_vectors is not None and tok in donor_vectors:
                v = donor_vectors[tok]
                vec = v.to(self.device, dtype=torch.float32)
                if vec.ndim == 1:
                    vec = vec.unsqueeze(0)
            else:
                if tok not in donor_vocab:
                    continue
                tok_id = donor_vocab[tok]
                base_vec = self.donor_embeddings[tok_id].to(
                    self.device, dtype=torch.float32
                )
                vec = base_vec.unsqueeze(0)

            if vec is None:
                continue

            if vec.shape[-1] != self.d_donor:
                raise ValueError(
                    f"Vector for token {tok!r} has dim {vec.shape[-1]} "
                    f"but donor dim is {self.d_donor}"
                )
            vectors.append(vec.squeeze(0))
            actual_tokens.append(tok)

        if not vectors:
            return torch.empty(0, self.d_donor, device=self.device), []

        return torch.stack(vectors, dim=0), actual_tokens

    # ------------------------------------------------------------------
    # Transplant operators
    # ------------------------------------------------------------------

    def transplant_wechsel(
        self,
        *,
        tokens: Optional[List[str]] = None,
        donor_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        WECHSEL: donor->base via alignment + KNN over full base vocab.

        For each donor-only token x_d:
          1. Project to base proxy: x_b_proxy = (x_d - μ_x) M + μ_y
          2. Normalize and compute cosine similarity to all base embeddings.
          3. Take top_k neighbors; softmax(focus_beta * sim) over them.
          4. Return their weighted average in base space.
        """
        donor_vecs, token_list = self._prepare_donor_vectors(tokens, donor_vectors)
        if donor_vecs.numel() == 0:
            print("[sb_transplant] WECHSEL: no tokens to transplant.")
            return torch.empty(0, self.d_base, device=self.device)

        M = donor_vecs.shape[0]
        k = min(self.top_k, self.base_embeddings.shape[0])

        print(
            f"[sb_transplant] WECHSEL on {M} tokens "
            f"(k={k}, beta={self.focus_beta}, batch={self.knn_batch_size})"
        )

        # project all donors to base space proxies
        proxies = self._project_vectors_to_base(donor_vecs)  # (M, d_base)
        proxies_norm = F.normalize(proxies, p=2, dim=1)

        base_norm = F.normalize(self.base_embeddings, p=2, dim=1)  # (V, d_base)

        new_embeds = torch.empty(M, self.d_base, device=self.device)
        batch_size = self.knn_batch_size

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch = proxies_norm[start:end]  # (B, d)
            # (B, V) similarity but only for this mini-batch (no giant MxV)
            sims = batch @ base_norm.T
            vals, idxs = torch.topk(sims, k, dim=1)
            weights = torch.softmax(self.focus_beta * vals, dim=1).unsqueeze(-1)
            neighbors = self.base_embeddings[idxs]  # (B, k, d)
            new_embeds[start:end] = (neighbors * weights).sum(dim=1)

        return new_embeds

    def transplant_focus(
        self,
        *,
        tokens: Optional[List[str]] = None,
        donor_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        FOCUS-style sharded interpolation:

          - Similarity is computed in donor anchor space.
          - We combine the corresponding base anchor embeddings.

        For each token x_d:
          1. Compute cosine sim between x_d and donor anchors.
          2. Take top_k anchors j.
          3. w_j = softmax(beta * sim_j).
          4. new_base = sum_j w_j * base_anchor_j.
        """
        donor_vecs, token_list = self._prepare_donor_vectors(tokens, donor_vectors)
        if donor_vecs.numel() == 0:
            print("[sb_transplant] FOCUS: no tokens to transplant.")
            return torch.empty(0, self.d_base, device=self.device)

        base_idx, donor_idx = self.get_shared_vocab()
        donor_anchor = self.donor_embeddings[torch.tensor(donor_idx, device=self.device)]
        base_anchor = self.base_embeddings[torch.tensor(base_idx, device=self.device)]

        donor_anchor_norm = F.normalize(donor_anchor, p=2, dim=1)
        donor_vecs_norm = F.normalize(donor_vecs, p=2, dim=1)

        M = donor_vecs.shape[0]
        N = donor_anchor.shape[0]
        k = min(self.top_k, N)
        print(
            f"[sb_transplant] FOCUS on {M} tokens "
            f"(anchors={N}, k={k}, beta={self.focus_beta})"
        )

        new_embeds = torch.empty(M, self.d_base, device=self.device)
        batch_size = self.knn_batch_size

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch = donor_vecs_norm[start:end]  # (B, d_donor)
            sims = batch @ donor_anchor_norm.T  # (B, N)
            vals, idxs = torch.topk(sims, k, dim=1)
            weights = torch.softmax(self.focus_beta * vals, dim=1).unsqueeze(-1)
            base_sel = base_anchor[idxs]  # (B, k, d_base)
            new_embeds[start:end] = (base_sel * weights).sum(dim=1)

        return new_embeds

    def transplant_clp(
        self,
        *,
        tokens: Optional[List[str]] = None,
        donor_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        CLP-style barycentric interpolation:

          - Similarity in donor anchor space.
          - Weights are ReLU(sim) with linear normalisation (no softmax).
          - Optionally restricted to top_k anchors for sparsity.
        """
        donor_vecs, token_list = self._prepare_donor_vectors(tokens, donor_vectors)
        if donor_vecs.numel() == 0:
            print("[sb_transplant] CLP: no tokens to transplant.")
            return torch.empty(0, self.d_base, device=self.device)

        base_idx, donor_idx = self.get_shared_vocab()
        donor_anchor = self.donor_embeddings[torch.tensor(donor_idx, device=self.device)]
        base_anchor = self.base_embeddings[torch.tensor(base_idx, device=self.device)]

        donor_anchor_norm = F.normalize(donor_anchor, p=2, dim=1)
        donor_vecs_norm = F.normalize(donor_vecs, p=2, dim=1)

        M = donor_vecs.shape[0]
        N = donor_anchor.shape[0]
        k = min(self.top_k, N)
        print(f"[sb_transplant] CLP on {M} tokens (anchors={N}, k={k})")

        new_embeds = torch.empty(M, self.d_base, device=self.device)
        batch_size = self.knn_batch_size

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            batch = donor_vecs_norm[start:end]  # (B, d_donor)
            sims = batch @ donor_anchor_norm.T  # (B, N)

            if k < N:
                vals, idxs = torch.topk(sims, k, dim=1)
                vals = F.relu(vals)
                denom = vals.sum(dim=1, keepdim=True).clamp_min(1e-9)
                weights = vals / denom
                weights = weights.unsqueeze(-1)  # (B, k, 1)
                base_sel = base_anchor[idxs]  # (B, k, d_base)
                new_embeds[start:end] = (base_sel * weights).sum(dim=1)
            else:
                sims_pos = F.relu(sims)
                denom = sims_pos.sum(dim=1, keepdim=True).clamp_min(1e-9)
                weights = sims_pos / denom  # (B, N)
                weights = weights.unsqueeze(-1)  # (B, N, 1)
                base_sel = base_anchor.unsqueeze(0).expand(batch.shape[0], -1, -1)
                new_embeds[start:end] = (base_sel * weights).sum(dim=1)

        return new_embeds

    def transplant_vipi(
        self,
        *,
        tokens: Optional[List[str]] = None,
        donor_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        VIPI / subword-mean approximation:

          For each donor-only token string t:
            - Tokenize t with the *base* tokenizer (no special tokens).
            - If it splits into subwords ids, average their embeddings.
            - If it yields empty or all-UNK, fall back to base embedding mean.
        """
        _ = donor_vectors  # unused; VIPI ignores donor geometry
        token_list = tokens if tokens is not None else self.get_missing_vocab()
        if not token_list:
            print("[sb_transplant] VIPI: no tokens to transplant.")
            return torch.empty(0, self.d_base, device=self.device)

        print(f"[sb_transplant] VIPI on {len(token_list)} tokens …")

        new_embeds: List[torch.Tensor] = []
        base_vocab_size = self.base_embeddings.shape[0]
        fallback = self.base_embeddings.mean(dim=0)

        for tok in tqdm(token_list, desc="VIPI reconstruction"):
            ids = self.base_tokenizer.encode(tok, add_special_tokens=False)
            if not ids:
                new_embeds.append(fallback)
                continue

            # filter ids to valid range
            ids = [i for i in ids if 0 <= i < base_vocab_size]
            if not ids:
                new_embeds.append(fallback)
                continue

            sub_vecs = self.base_embeddings[torch.tensor(ids, device=self.device)]
            new_embeds.append(sub_vecs.mean(dim=0))

        return torch.stack(new_embeds, dim=0)
