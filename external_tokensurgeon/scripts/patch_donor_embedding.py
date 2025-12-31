"""Patch donor model input embeddings and LM-head with designed token vectors.

Optimizations:
  - Load the model on CPU (no GPU residency required).
  - Allow memory-saving dtype (e.g., --dtype bf16).
  - Safe loader with low_cpu_mem_usage.
  - SafeTensors saving by default.

Behavior preserved:
  - Add tokens to tokenizer, resize embeddings, detect tie, patch rows, save model+tokenizer.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Artifact loading / selection
# ----------------------------

def _copy_dynamic_modules(model: AutoModelForCausalLM, output_dir: Path, source_id: str) -> None:
    """
    Copy any dynamically loaded remote-code modules into the saved model directory so
    reloading from that path succeeds without reaching back to the Hub.
    """
    modules_to_copy = set()
    config = getattr(model, "config", None)
    if config is not None:
        auto_map = getattr(config, "auto_map", None)
        if isinstance(auto_map, dict):
            for value in auto_map.values():
                entries = [value] if isinstance(value, str) else [
                    item for item in value if isinstance(item, str)
                ]
                for entry in entries:
                    head = entry.split(":", 1)[0]
                    module_name = head.rsplit(".", 1)[0] if "." in head else head
                    if module_name:
                        modules_to_copy.add(module_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    for module_name in modules_to_copy:
        module = None
        src_path = None
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = None
        if module is not None:
            module_path = getattr(module, "__file__", None)
            if module_path:
                src_candidate = Path(module_path)
                if src_candidate.exists():
                    src_path = src_candidate

        if src_path is None or not src_path.exists():
            if Path(source_id).is_dir():
                alt = Path(source_id) / Path(*module_name.split("."))
                if alt.is_dir():
                    src_path = alt
                else:
                    py_alt = alt.with_suffix(".py")
                    if py_alt.exists():
                        src_path = py_alt
            else:
                try:
                    src_path = Path(
                        hf_hub_download(
                            repo_id=source_id,
                            filename=f"{module_name.replace('.', '/')}.py",
                            revision=getattr(model.config, "_commit_hash", None),
                        )
                    )
                except Exception:
                    src_path = None

        if src_path is None or not src_path.exists():
            continue
        module_spec = getattr(module, "__spec__", None) if module is not None else None
        if src_path.is_dir():
            package_dest = output_dir / Path(*module_name.split("."))
            try:
                shutil.copytree(src_path, package_dest, dirs_exist_ok=True)
            except Exception:
                continue
            continue
        if module_spec and module_spec.origin and module_spec.origin.endswith("__init__.py"):
            # package module: copy the entire package directory
            package_src = Path(module_spec.origin).parent
            package_dest = output_dir / Path(*module_name.split("."))
            try:
                shutil.copytree(package_src, package_dest, dirs_exist_ok=True)
            except Exception:
                continue
            continue
        if src_path.suffix != ".py":
            continue
        dest_path = output_dir / Path(*module_name.split("."))
        dest_path = dest_path.with_suffix(".py")
        if dest_path.exists():
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(src_path, dest_path)
        except Exception:
            continue

def _as_1d(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.ndim == 2 and min(x.shape) == 1:
        x = x.view(-1)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D vector, got shape {tuple(x.shape)}")
    return x


def _maybe_get(d: Dict, *keys: str) -> Optional[torch.Tensor]:
    cur = d
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return None


def _load_design_pair(path: str, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, str]]:
    """
    Load (embed_vec, head_vec) from a design artifact. Prefer donor-side fields.
    Fallback order:
      - new_token.donor.{embedding, lm_head}
      - {donor_embedding, donor_head_embedding, donor_lm_head}
      - {embedding, lm_head, head_embedding}
      - vector (as embedding only)
      - base_* only if donor_* absent (last resort; includes base_head_embedding/base_lm_head)
    Returns (embed_vec, head_vec, info_dict).
    """
    # Always map to CPU; we’ll cast to param dtype/device later.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    info = {"embed_key": "None", "head_key": "None"}

    # Try nested 'new_token' first
    embed = None
    head = None
    if isinstance(payload, dict):
        # donor side
        embed = _maybe_get(payload, "new_token", "donor", "embedding")
        head  = _maybe_get(payload, "new_token", "donor", "lm_head")
        if embed is not None:
            info["embed_key"] = "new_token.donor.embedding"
        if head is not None:
            info["head_key"] = "new_token.donor.lm_head"

        # top-level donor_* fallbacks
        if embed is None and "donor_embedding" in payload:
            embed = payload["donor_embedding"]
            info["embed_key"] = "donor_embedding"
        if head is None and "donor_head_embedding" in payload:
            head = payload["donor_head_embedding"]
            info["head_key"] = "donor_head_embedding"
        if head is None and "donor_lm_head" in payload:
            head = payload["donor_lm_head"]
            info["head_key"] = "donor_lm_head"

        # generic names
        if embed is None and "embedding" in payload:
            embed = payload["embedding"]
            info["embed_key"] = "embedding"
        if head is None and "head_embedding" in payload:
            head = payload["head_embedding"]
            info["head_key"] = "head_embedding"
        if head is None and "lm_head" in payload:
            head = payload["lm_head"]
            info["head_key"] = "lm_head"

        # vector fallback (embedding only)
        if embed is None and "vector" in payload:
            embed = payload["vector"]
            info["embed_key"] = "vector"

        # last-resort base_* (compat)
        if embed is None and "base_embedding" in payload:
            embed = payload["base_embedding"]
            info["embed_key"] = "base_embedding"
        if head is None and "base_head_embedding" in payload:
            head = payload["base_head_embedding"]
            info["head_key"] = "base_head_embedding"
        if head is None and "base_lm_head" in payload:
            head = payload["base_lm_head"]
            info["head_key"] = "base_lm_head"

    else:
        # raw tensor -> embedding
        embed = payload
        info["embed_key"] = "<raw-tensor>"

    if embed is not None:
        embed = _as_1d(torch.as_tensor(embed, dtype=torch.float32, device="cpu"))
    if head is not None:
        head = _as_1d(torch.as_tensor(head, dtype=torch.float32, device="cpu"))

    return embed, head, info


# ----------------------------
# Performance-safe loader
# ----------------------------

def _resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    """
    Map CLI dtype string to torch dtype. Default 'fp32' (original numerics).
    """
    name = (name or "fp32").lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"auto"}:
        # reasonable default for memory reduction
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unknown dtype {name}")


def _safe_load_causal_lm_cpu(
    model_id: str,
    *,
    trust_remote_code: bool,
    torch_dtype: Optional[torch.dtype],
) -> AutoModelForCausalLM:
    """
    Load the model on CPU with memory-friendly options.
    """
    kwargs = dict(
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    # NOTE: We deliberately do NOT move to CUDA; patching rows on CPU is cheaper.
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


# ----------------------------
# Arg parsing
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject designed token vectors into donor model (input & LM-head).")

    p.add_argument("--donor-model", required=True, help="HF repo id or local path for the donor model")
    p.add_argument("--design", action="append", required=True,
                   help="Path to design artifact (.pt). Repeatable (order must match tokens)")
    p.add_argument("--token", action="append",
                   help="Explicit token strings (must match number/order of --design)")
    p.add_argument("--auto-token-prefix",
                   help="If --token not given, create names like '<prefix>0', '<prefix>1', ...")
    p.add_argument("--as-special", action="store_true", help="Add tokens as special tokens")
    p.add_argument("--output", required=True, help="Directory to save the patched model")
    p.add_argument("--token-output", help="Optional path to write the list of new tokens")

    # Scaling
    p.add_argument("--embedding-scale", type=float, default=None,
                   help="Scale for input embedding vector before insertion")
    p.add_argument("--head-scale", type=float, default=None,
                   help="Scale for LM-head vector before insertion")
    p.add_argument("--output-scale", type=float, default=None,
                   help="(Deprecated) Single scale applied to BOTH embedding & head when specific scales are not given")

    # Behavior controls
    p.add_argument("--no-copy-embed-to-head", action="store_true",
                   help="If head vector is missing and model is untied, do NOT copy embedding into LM-head")
    p.add_argument("--tied-merge", choices=["embed", "head", "avg"], default="embed",
                   help="If model is TIED and both embed/head vectors are provided but differ, choose how to merge (default: embed)")

    # Performance / storage
    p.add_argument("--dtype", default="bf16",
                   choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "auto"],
                   help="Storage dtype at load time. Use 'bf16' to reduce RAM. Default 'fp32'.")
    p.add_argument("--device",type=str, default="cpu", help="Device for intermediate computations (default: cpu)")
    p.add_argument("--trust-remote-code", action="store_true", help="Enable remote code trust for model loading")
    p.add_argument("--no-safe-serialization", action="store_true",
                   help="Disable safetensors when saving (defaults to safe serialization).")

    return p.parse_args()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()

    # Token names
    if args.token and len(args.token) != len(args.design):
        raise ValueError("Number of --token entries must match number of --design entries")
    if args.token:
        tokens: List[str] = args.token
    elif args.auto_token_prefix:
        tokens = [f"{args.auto_token_prefix}{i}" for i in range(len(args.design))]
    else:
        raise ValueError("Either --token or --auto-token-prefix must be provided")

    # Scales (support legacy --output-scale)
    if args.output_scale is not None and (args.embedding_scale is not None or args.head_scale is not None):
        print("[warn] --output-scale is deprecated and ignored when --embedding-scale/--head-scale are provided.")
    embed_scale = args.embedding_scale if args.embedding_scale is not None else (args.output_scale if args.output_scale is not None else 1.0)
    head_scale  = args.head_scale if args.head_scale is not None else (args.output_scale if args.output_scale is not None else 1.0)

    # Load donor model + tokenizer ON CPU with memory-friendly dtype
    torch_dtype = _resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.donor_model, trust_remote_code=True)
    model = _safe_load_causal_lm_cpu(
        args.donor_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # Add tokens, resize embeddings
    added = tokenizer.add_tokens(tokens, special_tokens=args.as_special)
    if added != len(tokens):
        raise RuntimeError(f"Tokenizer added {added} of {len(tokens)} tokens (mismatch)")

    # Resize embeddings to new vocab size
    model.resize_token_embeddings(len(tokenizer))
    # Some models need a re-tie after resize; safe to call (no-op if not needed)
    try:
        model.tie_weights()
    except Exception:
        pass

    # Get weight tensors AFTER resize
    in_emb = model.get_input_embeddings()
    if in_emb is None or not hasattr(in_emb, "weight"):
        raise RuntimeError("Model has no accessible input embedding weight")
    W_in = in_emb.weight  # (V, d), on CPU with chosen dtype

    out_emb_mod = model.get_output_embeddings()
    W_out = out_emb_mod.weight if (out_emb_mod is not None and hasattr(out_emb_mod, "weight")) else None

    # Detect tying *now* (after resize/tie)
    tied = False
    if W_out is not None:
        try:
            tied = (W_in.data_ptr() == W_out.data_ptr())
        except Exception:
            tied = False

    if tied and (embed_scale != head_scale):
        print("[warn] Model is tied; embedding/head share storage. Forcing a single scale (embedding-scale).")
        head_scale = embed_scale

    # Preload designs on CPU to avoid repeated disk I/O
    designs: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, str]]] = []
    for design_path in args.design:
        designs.append(_load_design_pair(design_path, device=torch.device("cpu")))

    # Insert per token
    for token, (design_path, (embed_vec, head_vec, info)) in zip(tokens, zip(args.design, designs)):
        # Resolve token id after add_tokens
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Token {token!r} not present after tokenizer augmentation")

        # Helper: cast vectors to match param dtype/device (CPU)
        def to_param_dtype(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
            return v.to(dtype=like.dtype, device=like.device)

        if tied:
            # Single row controls both. Choose & scale.
            if embed_vec is None and head_vec is None:
                raise ValueError(f"{design_path}: no vector found for tied model (need embed and/or head)")
            if embed_vec is not None and head_vec is None:
                v = embed_vec * float(embed_scale)
                chosen = f"{info['embed_key']} (embed)"
            elif head_vec is not None and embed_vec is None:
                v = head_vec * float(head_scale)
                chosen = f"{info['head_key']} (head)"
            else:
                if args.tied_merge == "embed":
                    v = embed_vec * float(embed_scale)
                    chosen = f"{info['embed_key']} (tied_merge=embed)"
                elif args.tied_merge == "head":
                    v = head_vec * float(head_scale)
                    chosen = f"{info['head_key']} (tied_merge=head)"
                else:  # avg
                    v = 0.5 * (embed_vec * float(embed_scale) + head_vec * float(head_scale))
                    chosen = f"avg({info['embed_key']}, {info['head_key']})"
            v = to_param_dtype(v, W_in)
            with torch.no_grad():
                W_in[tok_id].copy_(v)
            print(f"[tied]  Patched {token!r} id={tok_id} from {design_path} using {chosen}; ||row||={float(v.norm()):.6f}")

        else:
            # Untied: patch embedding and head independently
            if embed_vec is None and head_vec is None:
                raise ValueError(f"{design_path}: no vector found (need at least embedding or head)")

            # --- embedding row ---
            if embed_vec is not None:
                v_in = to_param_dtype(embed_vec * float(embed_scale), W_in)
                with torch.no_grad():
                    W_in[tok_id].copy_(v_in)
                print(f"[embed] Patched {token!r} id={tok_id} from {design_path} key={info['embed_key']}; ||row||={float(v_in.norm()):.6f}")
            elif embed_vec is None and head_vec is not None:
                # Copy head vector into input row if no embedding vector provided
                v_in = to_param_dtype(head_vec * float(embed_scale), W_in)
                with torch.no_grad():
                    W_in[tok_id].copy_(v_in)
                print(f"[embed<-head] Patched {token!r} id={tok_id} (input) from head key={info['head_key']}; ||row||={float(v_in.norm()):.6f}")

            # --- LM-head row ---
            if W_out is not None:
                if head_vec is not None:
                    v_out = to_param_dtype(head_vec * float(head_scale), W_out)
                    with torch.no_grad():
                        W_out[tok_id].copy_(v_out)
                    print(f"[head]  Patched {token!r} id={tok_id} from {design_path} key={info['head_key']}; ||row||={float(v_out.norm()):.6f}")
                else:
                    if not args.no_copy_embed_to_head:
                        with torch.no_grad():
                            W_out[tok_id].copy_(W_in[tok_id])
                        print(f"[head<-embed] Copied input row to LM-head for {token!r} id={tok_id} (no head vector provided).")
                    else:
                        print(f"[head]  Skipped LM-head patch for {token!r} (no head vector and --no-copy-embed-to-head set).")
            else:
                print(f"[head]  Model exposes no LM-head weight; skipped head patch for {token!r}.")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=not args.no_safe_serialization)
    _copy_dynamic_modules(model, Path(out_dir), args.donor_model)
    tokenizer.save_pretrained(out_dir)

    if args.token_output:
        Path(args.token_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.token_output).write_text("\n".join(tokens) + "\n", encoding="utf-8")
        print(f"Wrote token list to {args.token_output}")

    print(f"Saved patched donor to {args.output}")


if __name__ == "__main__":
    main()
