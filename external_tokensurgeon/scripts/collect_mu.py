"""Collect token-conditioned representations from language models on a corpus.

Adds optional:
  --apply-final-norm-base / --apply-final-norm-donor  (default: off)
  --prepend-bos-general                              (default: off)

Performance knobs (defaults preserve original numerics & behavior):
  --dtype fp32|bf16|fp16|auto        (default: fp32)
  --attn-impl eager|flash_attention_2 (default: eager)

Defaults preserve the original behavior: last hidden state, no extra norm, no BOS.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..cli import load_token_list, save_tensor


# -------------------------- datatypes --------------------------

@dataclass
class TokenSpec:
    token: str
    ids: List[int]
    max_count: int
    vectors: List[torch.Tensor] = field(default_factory=list)
    count: int = 0

    def add_vector(self, vector: torch.Tensor) -> None:
        self.vectors.append(vector.detach().cpu())
        self.count += 1

    @property
    def is_satisfied(self) -> bool:
        return self.count >= self.max_count


# -------------------------- utils --------------------------

def _resolve_hidden_size(model: AutoModelForCausalLM) -> int:
    """
    Handle both legacy configs (hidden_size / n_embd / d_model on the root) and
    newer modular configs (e.g. Gemma-2) where those values live under a
    nested text_config or use alternative field names.
    """
    cfg = model.config

    def _try_sources(sources, keys):
        for source in sources:
            if source is None:
                continue
            for key in keys:
                value = getattr(source, key, None)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
        return None

    # Direct attributes first (legacy layout)
    roots = [cfg]
    size = _try_sources(roots, ("hidden_size", "n_embd", "d_model"))
    if size is not None:
        return size

    # Nested configs (Gemma-2 + similar expose text_config)
    nested_cfg = getattr(cfg, "text_config", None)
    size = _try_sources(
        [nested_cfg],
        ("hidden_size", "n_embd", "d_model", "hidden_dim", "model_dim", "embed_dim"),
    )
    if size is not None:
        return size

    # Alternative names on the root (some configs prefer hidden_dim/model_dim/embed_dim)
    size = _try_sources(
        roots, ("hidden_dim", "model_dim", "embed_dim")
    )
    if size is not None:
        return size

    # Last resort: inspect the serialized dict in case custom configs stash it there
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        cfg_dict = {}
    if isinstance(cfg_dict, dict):
        for key in ("hidden_size", "n_embd", "d_model", "hidden_dim", "model_dim", "embed_dim"):
            if key in cfg_dict and cfg_dict[key] is not None:
                try:
                    return int(cfg_dict[key])
                except (TypeError, ValueError):
                    continue
        nested_dict = cfg_dict.get("text_config")
        if isinstance(nested_dict, dict):
            for key in ("hidden_size", "n_embd", "d_model", "hidden_dim", "model_dim", "embed_dim"):
                if key in nested_dict and nested_dict[key] is not None:
                    try:
                        return int(nested_dict[key])
                    except (TypeError, ValueError):
                        continue

    raise ValueError("Unable to determine hidden size for model")


def _encode_token(tokenizer: AutoTokenizer, token: str) -> List[int]:
    encoding = tokenizer(token, add_special_tokens=False)
    ids = encoding.get("input_ids", [])
    if not ids:
        raise ValueError(f"Token {token!r} is unknown to tokenizer {tokenizer.name_or_path}")
    return ids


def _build_token_specs(
    tokenizer: AutoTokenizer, tokens: Sequence[str], max_count: int
) -> Dict[str, TokenSpec]:
    specs: Dict[str, TokenSpec] = {}
    for token in tokens:
        try:
            ids = _encode_token(tokenizer, token)
        except ValueError as exc:
            print(f"[warn] {exc}")
            continue
        specs[token] = TokenSpec(token=token, ids=ids, max_count=max_count)
    return specs


def _find_matches(ids: Sequence[int], pattern: Sequence[int]) -> List[int]:
    matches: List[int] = []
    pat_len = len(pattern)
    if pat_len == 0 or len(ids) < pat_len:
        return matches
    for start in range(len(ids) - pat_len + 1):
        if ids[start : start + pat_len] == list(pattern):
            # return index of the last token in the pattern (position to read)
            matches.append(start + pat_len - 1)
    return matches


def _prepare_document_chunks(
    ids: Sequence[int],
    chunk_length: int,
    max_chunks: int,
    rng: random.Random,
) -> List[List[int]]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    ids_list = list(ids)
    if not ids_list:
        return []
    if len(ids_list) <= chunk_length:
        chunks = [ids_list]
    else:
        chunks = [ids_list[i : i + chunk_length] for i in range(0, len(ids_list), chunk_length)]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return []
    if max_chunks > 0 and len(chunks) > max_chunks:
        selected = rng.sample(range(len(chunks)), max_chunks)
        selected.sort()
        chunks = [chunks[idx] for idx in selected]
    return chunks


# ----------- model-specific helpers: final norm and tie detection -----------

def _get_attr_path(root, path: str):
    cur = root
    for name in path.split("."):
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur

def _get_final_norm_module(model: AutoModelForCausalLM):
    """
    Try common places for the final pre-logit normalization module.
    Works for Llama/Mistral-style (model.norm), GPT2-style (transformer.ln_f), etc.
    Returns a torch.nn.Module or None.
    """
    candidates = [
        "model.norm",                # Llama/Mistral
        "transformer.ln_f",          # GPT2
        "base_model.model.norm",     # some wrappers
        "model.final_layernorm",     # Falcon variants
        "model.rms_norm",            # alternative naming
        "model.ln_f",                # misc
        "norm",                      # sometimes directly on top level
    ]
    for p in candidates:
        mod = _get_attr_path(model, p)
        if mod is not None:
            return mod
    return None

def _is_head_tied(model: AutoModelForCausalLM) -> Optional[bool]:
    """
    Heuristic: compare data_ptr of input vs output embedding weights if both exist.
    Returns True/False if determinable, else None.
    """
    try:
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
    except Exception:
        return None
    if inp is None or out is None:
        return None
    try:
        return inp.weight.data_ptr() == out.weight.data_ptr()
    except Exception:
        return None


# -------------------------- performance-safe loaders --------------------------

def _resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    """
    Map CLI dtype string to torch dtype. Default 'fp32' preserves original numerics.
    """
    name = (name or "fp32").lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"auto"}:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unknown dtype {name}")

def _safe_load_causal_lm(
    model_id: str,
    *,
    trust_remote_code: bool,
    dtype: Optional[torch.dtype],
    attn_impl: Optional[str],
    device: torch.device,
) -> AutoModelForCausalLM:
    """
    Load with requested dtype/attn implementation; fall back cleanly if unsupported.
    """
    base_kwargs = dict(
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if dtype is not None:
        base_kwargs["torch_dtype"] = dtype

    try:
        if attn_impl:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation=attn_impl, **base_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs)
    except TypeError:
        # transformers or model doesn't accept attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs)

    return model.to(device)


# -------------------------- extraction --------------------------

def _extract_hidden(
    model: AutoModelForCausalLM,
    chunk: Sequence[int],
    device: torch.device,
    *,
    apply_final_norm: bool = False,
) -> torch.Tensor:
    """
    Returns last hidden state per position, optionally passed through the model's
    final pre-logit normalization (RMSNorm / LN). Shape: (T, d)
    """
    hs = _resolve_hidden_size(model)
    if not chunk:
        return torch.empty((0, hs), dtype=torch.float32)

    ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(ids, device=device)

    # Use inference_mode for speed/memory; keep semantics identical.
    with torch.inference_mode():
        out = model(
            input_ids=ids,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]  # (1, T, d)

        if apply_final_norm:
            fn = _get_final_norm_module(model)
            if fn is not None:
                h = fn(h)

    h = h.squeeze(0).to(dtype=torch.float32)
    if h.ndim == 1:
        h = h.unsqueeze(0)
    return h.to(device="cpu")


def _flatten_vectors(specs: Dict[str, TokenSpec], hidden_size: int) -> torch.Tensor:
    matrices: List[torch.Tensor] = []
    for spec in specs.values():
        if spec.vectors:
            matrices.append(torch.stack(spec.vectors, dim=0))
    if not matrices:
        return torch.empty((0, hidden_size), dtype=torch.float32)
    return torch.cat(matrices, dim=0)


def _mean_over_specs(specs: Dict[str, TokenSpec]) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for spec in specs.values():
        if spec.vectors:
            vectors.extend(spec.vectors)
    if not vectors:
        raise RuntimeError("No vectors collected; cannot compute mean")
    return torch.stack(vectors, dim=0).mean(dim=0)


def _estimate_total_documents(dataset, max_documents: Optional[int]) -> Optional[int]:
    try:
        dataset_len = len(dataset)
    except TypeError:
        dataset_len = None
    total = max_documents
    if dataset_len is not None:
        total = dataset_len if total is None else min(total, dataset_len)
    return total


# -------------------------- collection --------------------------

def _collect_vectors_from_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    specs: Dict[str, TokenSpec],
    text: str,
    *,
    device: torch.device,
    chunk_length: int,
    max_chunks_per_document: int,
    rng: random.Random,
    apply_final_norm: bool,
    allow_prepend_bos: bool,
    bos_id: Optional[int],
) -> None:
    if not specs or all(spec.is_satisfied for spec in specs.values()):
        return

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
        truncation=False,
    )
    input_ids: List[int] = encoding.get("input_ids", [])
    if not input_ids:
        return

    chunks = _prepare_document_chunks(
        input_ids,
        chunk_length=chunk_length,
        max_chunks=max_chunks_per_document,
        rng=rng,
    )

    for chunk in chunks:
        if all(spec.is_satisfied for spec in specs.values()):
            break

        # In token-matching mode, we DO NOT prepend BOS to preserve exact matches.
        if allow_prepend_bos and bos_id is not None:
            if not chunk or chunk[0] != bos_id:
                chunk = [bos_id] + chunk

        hidden = _extract_hidden(model, chunk, device, apply_final_norm=apply_final_norm)
        if hidden.numel() == 0:
            continue
        if not torch.isfinite(hidden).all():
            finite_ratio = torch.isfinite(hidden).float().mean().item()
            print(
                f"[warn] Non-finite activations detected while collecting token vectors "
                f"(finite ratio={finite_ratio:.6f}); skipping chunk"
            )
            continue
        for spec in specs.values():
            if spec.is_satisfied:
                continue
            positions = _find_matches(chunk, spec.ids)
            for position in positions:
                if position < hidden.shape[0]:
                    spec.add_vector(hidden[position])
                if spec.is_satisfied:
                    break


def _collect_general_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    *,
    device: torch.device,
    max_documents: Optional[int],
    chunk_length: int,
    max_chunks_per_document: int,
    max_samples: Optional[int],
    rng: random.Random,
    show_progress: bool,
    desc: str,
    apply_final_norm: bool,
    prepend_bos: bool,
) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    remaining = max_samples if (max_samples is None or max_samples > 0) else None
    total = _estimate_total_documents(dataset, max_documents)
    progress = tqdm(total=total, disable=not show_progress, desc=desc, dynamic_ncols=True)

    bos_id = tokenizer.bos_token_id if prepend_bos else None

    documents_processed = 0
    for sample in dataset:
        if max_documents is not None and documents_processed >= max_documents:
            break
        text = sample.get("text") or sample.get("content") or ""
        documents_processed += 1
        progress.update(1)
        if not text:
            continue

        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
        )
        input_ids: List[int] = encoding.get("input_ids", [])
        chunks = _prepare_document_chunks(
            input_ids,
            chunk_length=chunk_length,
            max_chunks=max_chunks_per_document,
            rng=rng,
        )

        for chunk in chunks:
            if prepend_bos and bos_id is not None:
                if not chunk or chunk[0] != bos_id:
                    chunk = [bos_id] + chunk

            hidden = _extract_hidden(model, chunk, device, apply_final_norm=apply_final_norm)
            if hidden.numel() == 0:
                continue
            if not torch.isfinite(hidden).all():
                finite_ratio = torch.isfinite(hidden).float().mean().item()
                print(
                    f"[warn] Non-finite activations detected while collecting general states "
                    f"(finite ratio={finite_ratio:.6f}); skipping chunk"
                )
                continue

            if remaining is None:
                vectors.append(hidden)
            else:
                take = min(remaining, hidden.shape[0])
                if take <= 0:
                    break
                vectors.append(hidden[:take])
                remaining -= take
                if remaining <= 0:
                    break

        if remaining is not None and remaining <= 0:
            break

    progress.close()
    if not vectors:
        return torch.empty((0, _resolve_hidden_size(model)), dtype=torch.float32)
    return torch.cat(vectors, dim=0)


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect corpus-conditioned hidden states / µ vectors for base and donor models"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--donor-model", required=True)

    parser.add_argument(
        "--token-file",
        help="Tokens representing the positive cluster. If omitted, collect generic corpus states",
    )
    parser.add_argument(
        "--negative-token-file",
        help="Optional tokens whose base representations will be treated as negatives (requires --token-file)",
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-samples-per-token", type=int, default=64)
    parser.add_argument(
        "--max-general-samples",
        type=int,
        default=50000,
        help="Number of positions to collect when --token-file is omitted (<=0 disables the cap)",
    )
    parser.add_argument("--sequence-length", type=int, default=512, help="Chunk length for each document slice")
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=4,
        help="Maximum chunks sampled per document (<=0 keeps all chunks)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for chunk sampling")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    # NEW: performance knobs (defaults preserve original numerics)
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "auto"],
        help="Compute dtype. Default 'fp32' preserves original numerics; use 'bf16' on H100.",
    )
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["eager", "flash_attention_2"],
        help="Attention implementation. Default 'eager' preserves behavior; "
             "use 'flash_attention_2' on H100 for speed (falls back if unsupported).",
    )

    # norms and BOS control (default OFF to preserve behavior)
    parser.add_argument(
        "--apply-final-norm-base",
        action="store_true",
        help="Apply the model's final pre-logit normalization to base hidden states",
    )
    parser.add_argument(
        "--apply-final-norm-donor",
        action="store_true",
        help="Apply the model's final pre-logit normalization to donor hidden states",
    )
    parser.add_argument(
        "--prepend-bos-general",
        action="store_true",
        help="Prepend BOS for general-state collection (ignored in token-matching mode)",
    )


    return parser.parse_args()


# -------------------------- main --------------------------

def main() -> None:
    args = parse_args()

    print("!!!collect_mu.py parsed args gets", args.device)
    if 'cpu' in args.device:
        print("!!!collect_mu.py using cpu")
        raise RuntimeError("CPU device is not supported for collect_mu.py; please use a GPU device.")
        

    
    if args.negative_token_file and not args.token_file:
        raise ValueError("--negative-token-file requires --token-file to be set")

    # Enable TF32 only when opting into mixed/bfloat16 modes,
    # to keep fp32 parity by default.
    if torch.cuda.is_available() and args.dtype.lower() in {"bf16", "bfloat16", "fp16", "float16", "auto"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)

    general_mode = args.token_file is None
    max_documents: Optional[int] = args.max_documents if args.max_documents > 0 else None
    progress_enabled = not args.no_progress
    chunk_length = args.sequence_length

    # Token lists (donor specs are built later, when donor model is loaded)
    positive_tokens: Sequence[str] = []
    if not general_mode:
        positive_tokens = load_token_list(args.token_file)

    negative_tokens: Sequence[str] = []
    if args.negative_token_file:
        negative_tokens = load_token_list(args.negative_token_file)

    load_dtype = _resolve_torch_dtype(args.dtype)
    attn_impl = args.attn_impl

    # ---------------- BASE PHASE ----------------
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    base_model = _safe_load_causal_lm(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        dtype=load_dtype,
        attn_impl=attn_impl,
        device=device,
    )
    base_model.eval()

    base_hidden = _resolve_hidden_size(base_model)
    base_tied = _is_head_tied(base_model)
    base_has_norm = _get_final_norm_module(base_model) is not None
    print(f"[info] base head tied? {base_tied} ; has final norm? {base_has_norm}")

    # Will need these names later when saving negatives
    negative_specs: Dict[str, TokenSpec] = {}

    if general_mode:
        max_general_samples = (args.max_general_samples if args.max_general_samples > 0 else None)
        base_matrix = _collect_general_states(
            base_model,
            base_tokenizer,
            dataset,
            device=device,
            max_documents=max_documents,
            chunk_length=chunk_length,
            max_chunks_per_document=args.max_chunks_per_document,
            max_samples=max_general_samples,
            rng=random.Random(args.seed),
            show_progress=progress_enabled,
            desc="Base documents",
            apply_final_norm=args.apply_final_norm_base,
            prepend_bos=args.prepend_bos_general,
        )
        if base_matrix.numel() == 0:
            raise RuntimeError("No hidden states were collected for the base model")
        if not torch.isfinite(base_matrix).all():
            raise RuntimeError("Collected base hidden states contain non-finite values")
        mu_base = base_matrix.mean(dim=0)
        base_specs: Dict[str, TokenSpec] = {}  # for payload uniformity
        base_negative_matrix = None
        mu_base_neg = None
    else:
        base_specs = _build_token_specs(base_tokenizer, positive_tokens, args.max_samples_per_token)
        if negative_tokens:
            negative_specs = _build_token_specs(
                base_tokenizer, negative_tokens, args.max_samples_per_token
            )

        total = _estimate_total_documents(dataset, max_documents)
        progress = tqdm(total=total, disable=not progress_enabled,
                        desc="Documents (base)", dynamic_ncols=True)
        documents_processed = 0
        base_rng = random.Random(args.seed)
        neg_rng = random.Random(args.seed + 2)

        for sample in dataset:
            if max_documents is not None and documents_processed >= max_documents:
                break
            text = sample.get("text") or sample.get("content") or ""
            documents_processed += 1
            progress.update(1)
            if not text:
                continue

            _collect_vectors_from_text(
                base_model,
                base_tokenizer,
                base_specs,
                text,
                device=device,
                chunk_length=chunk_length,
                max_chunks_per_document=args.max_chunks_per_document,
                rng=base_rng,
                apply_final_norm=args.apply_final_norm_base,
                allow_prepend_bos=False,
                bos_id=None,
            )
            if negative_specs:
                _collect_vectors_from_text(
                    base_model,
                    base_tokenizer,
                    negative_specs,
                    text,
                    device=device,
                    chunk_length=chunk_length,
                    max_chunks_per_document=args.max_chunks_per_document,
                    rng=neg_rng,
                    apply_final_norm=args.apply_final_norm_base,
                    allow_prepend_bos=False,
                    bos_id=None,
                )

            if all(spec.is_satisfied for spec in base_specs.values()) and \
               (not negative_specs or all(spec.is_satisfied for spec in negative_specs.values())):
                break

        progress.close()
        if not any(spec.vectors for spec in base_specs.values()):
            raise RuntimeError("No positive samples were collected for the base model")

        base_matrix = _flatten_vectors(base_specs, base_hidden)
        if not torch.isfinite(base_matrix).all():
            raise RuntimeError("Collected base hidden states contain non-finite values")
        mu_base = _mean_over_specs(base_specs)

        base_negative_matrix = None
        mu_base_neg = None
        if negative_specs and any(spec.vectors for spec in negative_specs.values()):
            base_negative_matrix = _flatten_vectors(negative_specs, base_hidden)
            mu_base_neg = _mean_over_specs(negative_specs)
        elif negative_tokens:
            print("[warn] Negative tokens file provided but no samples were collected")

    # Free base model before touching donor
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- DONOR PHASE ----------------
    donor_tokenizer = AutoTokenizer.from_pretrained(
        args.donor_model, trust_remote_code=args.trust_remote_code
    )
    donor_model = _safe_load_causal_lm(
        args.donor_model,
        trust_remote_code=args.trust_remote_code,
        dtype=load_dtype,
        attn_impl=attn_impl,
        device=device,
    )
    donor_model.eval()

    donor_hidden = _resolve_hidden_size(donor_model)
    donor_tied = _is_head_tied(donor_model)
    donor_has_norm = _get_final_norm_module(donor_model) is not None
    print(f"[info] donor head tied? {donor_tied} ; has final norm? {donor_has_norm}")

    if general_mode:
        max_general_samples = (args.max_general_samples if args.max_general_samples > 0 else None)
        donor_matrix = _collect_general_states(
            donor_model,
            donor_tokenizer,
            dataset,
            device=device,
            max_documents=max_documents,
            chunk_length=chunk_length,
            max_chunks_per_document=args.max_chunks_per_document,
            max_samples=max_general_samples,
            rng=random.Random(args.seed + 1),
            show_progress=progress_enabled,
            desc="Donor documents",
            apply_final_norm=args.apply_final_norm_donor,
            prepend_bos=args.prepend_bos_general,
        )
        if donor_matrix.numel() == 0:
            raise RuntimeError("No hidden states were collected for the donor model")
        if not torch.isfinite(donor_matrix).all():
            raise RuntimeError("Collected donor hidden states contain non-finite values")
        mu_donor = donor_matrix.mean(dim=0)
        donor_specs: Dict[str, TokenSpec] = {}
    else:
        donor_specs = _build_token_specs(donor_tokenizer, positive_tokens, args.max_samples_per_token)

        total = _estimate_total_documents(dataset, max_documents)
        progress = tqdm(total=total, disable=not progress_enabled,
                        desc="Documents (donor)", dynamic_ncols=True)
        documents_processed = 0
        donor_rng = random.Random(args.seed + 1)

        for sample in dataset:
            if max_documents is not None and documents_processed >= max_documents:
                break
            text = sample.get("text") or sample.get("content") or ""
            documents_processed += 1
            progress.update(1)
            if not text:
                continue

            _collect_vectors_from_text(
                donor_model,
                donor_tokenizer,
                donor_specs,
                text,
                device=device,
                chunk_length=chunk_length,
                max_chunks_per_document=args.max_chunks_per_document,
                rng=donor_rng,
                apply_final_norm=args.apply_final_norm_donor,
                allow_prepend_bos=False,
                bos_id=None,
            )
            if all(spec.is_satisfied for spec in donor_specs.values()):
                break

        progress.close()
        if not any(spec.vectors for spec in donor_specs.values()):
            raise RuntimeError("No positive samples were collected for the donor model")

        donor_matrix = _flatten_vectors(donor_specs, donor_hidden)
        if not torch.isfinite(donor_matrix).all():
            raise RuntimeError("Collected donor hidden states contain non-finite values")
        mu_donor = _mean_over_specs(donor_specs)

    # Done with donor; free it
    del donor_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- assemble payload ----------------
    if general_mode:
        payload = {
            "tokens": [],
            "dataset": {
                "name": args.dataset,
                "config": args.dataset_config,
                "split": args.split,
                "max_documents": args.max_documents,
                "sequence_length": args.sequence_length,
                "max_chunks_per_document": args.max_chunks_per_document,
                "max_general_samples": args.max_general_samples,
                "seed": args.seed,
            },
            "mu_base": mu_base,
            "mu_donor": mu_donor,
            "base_vectors": base_matrix,
            "donor_vectors": donor_matrix,
            "meta": {
                "base": {
                    "tied_head": base_tied,
                    "has_final_norm": base_has_norm,
                    "apply_final_norm": bool(args.apply_final_norm_base),
                    "prepend_bos_general": bool(args.prepend_bos_general),
                },
                "donor": {
                    "tied_head": donor_tied,
                    "has_final_norm": donor_has_norm,
                    "apply_final_norm": bool(args.apply_final_norm_donor),
                    "prepend_bos_general": bool(args.prepend_bos_general),
                },
            },
        }
    else:
        payload = {
            "tokens": list(base_specs.keys()),
            "dataset": {
                "name": args.dataset,
                "config": args.dataset_config,
                "split": args.split,
                "max_documents": args.max_documents,
                "max_samples_per_token": args.max_samples_per_token,
                "sequence_length": args.sequence_length,
                "max_chunks_per_document": args.max_chunks_per_document,
                "seed": args.seed,
            },
            "mu_base": mu_base,
            "mu_donor": mu_donor,
            "base_vectors": base_matrix,
            "donor_vectors": donor_matrix,
            "base_vectors_by_token": {
                token: torch.stack(spec.vectors, dim=0)
                for token, spec in base_specs.items()
                if spec.vectors
            },
            "donor_vectors_by_token": {
                token: torch.stack(spec.vectors, dim=0)
                for token, spec in donor_specs.items()
                if spec.vectors
            },
            "meta": {
                "base": {
                    "tied_head": base_tied,
                    "has_final_norm": base_has_norm,
                    "apply_final_norm": bool(args.apply_final_norm_base),
                    "prepend_bos_general": False,
                },
                "donor": {
                    "tied_head": donor_tied,
                    "has_final_norm": donor_has_norm,
                    "apply_final_norm": bool(args.apply_final_norm_donor),
                    "prepend_bos_general": False,
                },
            },
        }

        if base_negative_matrix is not None and mu_base_neg is not None:
            payload["negative_tokens"] = negative_tokens
            payload["mu_base_neg"] = mu_base_neg
            payload["base_negative_vectors"] = base_negative_matrix
            payload["base_negative_vectors_by_token"] = {
                token: torch.stack(spec.vectors, dim=0)
                for token, spec in negative_specs.items()
                if spec.vectors
            }
            save_tensor(out_dir / "mu_base_neg.pt", mu_base_neg)

    # ---------------- save ----------------
    torch.save(payload, out_dir / "mu_vectors.pt")
    save_tensor(out_dir / "mu_base.pt", payload["mu_base"])
    save_tensor(out_dir / "mu_donor.pt", payload["mu_donor"])
    save_tensor(out_dir / "base_vectors.pt", payload["base_vectors"])
    save_tensor(out_dir / "donor_vectors.pt", payload["donor_vectors"])

    print(f"Saved hidden state statistics to {out_dir / 'mu_vectors.pt'}")


if __name__ == "__main__":
    main()
