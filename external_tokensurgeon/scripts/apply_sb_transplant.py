from __future__ import annotations

"""
Apply shared-basis transplant (FOCUS / CLP / WECHSEL / VIPI) to produce
a patched *base* model.

Flow:
  1. You have already patched the donor with the breaker token
     (via patch_donor_embedding) and saved it as --donor-model.
  2. Victim wants to align donor tokenizer to base model using a
     shared-basis method (FOCUS / CLP / WECHSEL / VIPI).
  3. This script simulates that victim step efficiently:

       - Loads base tokenizer + model.
       - Loads *patched* donor tokenizer via SharedBasisTransplanter.
       - Finds all donor-only tokens (including breaker).
       - Applies the chosen shared-basis operator to compute their
         base-side input embeddings.
       - Adds those tokens to the base tokenizer and writes their
         embeddings directly into the resized embedding matrices.
       - Saves the patched base model once.

No calls to patch_donor_embedding for the base model.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from non_mergekit_methods.sb_transplant import SharedBasisTransplanter


def _read_tokens(path: Path) -> List[str]:
    tokens = [t.strip() for t in path.read_text(encoding="utf-8").splitlines()]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError(f"No tokens found in {path}")
    return tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply shared-basis transplant to produce a patched base model."
    )
    p.add_argument("--base-model", required=True)
    p.add_argument(
        "--donor-model",
        required=True,
        help="Patched donor directory (output of patch_donor_embedding).",
    )
    p.add_argument(
        "--tokens-file",
        required=True,
        help="Breaker tokens (one per line); used only for sanity checks / eval.",
    )
    p.add_argument(
        "--design",
        action="append",
        required=False,
        help=(
            "Optional design artefact(s) used for the breaker donor embedding. "
            "If present and they contain a 'config' dict, we reuse method/top_k/focus_beta."
        ),
    )
    p.add_argument(
        "--method",
        choices=["focus", "clp", "wechsel", "vipi"],
        default="focus",
        help="Shared-basis transplant operator.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override K for shared-basis methods. If omitted, use config or default 32.",
    )
    p.add_argument(
        "--focus-beta",
        type=float,
        default=None,
        help="Override beta for FOCUS/WECHSEL. If omitted, use config or default 10.0.",
    )
    p.add_argument("--output", required=True, help="Directory for patched base model.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--knn-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for WECHSEL KNN over base vocab.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    breaker_tokens = _read_tokens(Path(args.tokens_file))

    # Try to recover hyperparams from the first design artefact if available
    cfg_method: Optional[str] = None
    cfg_topk: Optional[int] = None
    cfg_beta: Optional[float] = None

    if args.design:
        first = Path(args.design[0])
        if first.exists():
            try:
                payload0 = torch.load(first, map_location="cpu")
                if isinstance(payload0, dict) and "config" in payload0:
                    cfg = payload0["config"] or {}
                    cfg_method = str(cfg.get("method", "")).lower() or None
                    cfg_topk = int(cfg.get("anchor_topk", 0) or 0) or None
                    cfg_beta = float(cfg.get("focus_beta", 0.0) or 0.0) or None
            except Exception as exc:  # noqa: BLE001
                print(f"[apply_sb] Warning: could not read config from {first}: {exc}")

    # Decide method, K, beta:
    method = args.method.lower()
    if cfg_method is not None and cfg_method != method:
        print(
            f"[apply_sb] Warning: designer used method={cfg_method}, "
            f"CLI requested method={method}. Using CLI value."
        )

    top_k = args.top_k if args.top_k is not None else (cfg_topk or 32)
    focus_beta = args.focus_beta if args.focus_beta is not None else (cfg_beta or 10.0)

    print(
        f"[apply_sb] Using shared-basis hyperparameters: "
        f"method={method}, top_k={top_k}, focus_beta={focus_beta}, "
        f"knn_batch_size={args.knn_batch_size}"
    )

    # Build transplanter between base and *patched* donor
    transplanter = SharedBasisTransplanter(
        base_model_name=args.base_model,
        donor_model_name=args.donor_model,
        device=str(device),
        trust_remote_code=args.trust_remote_code,
        top_k=top_k,
        focus_beta=focus_beta,
        knn_batch_size=args.knn_batch_size,
    )

    # Donor-only tokens will be fully transplanted (includes breaker)
    donor_only_tokens = transplanter.get_missing_vocab()

    # Sanity check: breaker tokens should appear among donor-only tokens
    missing_breakers = [t for t in breaker_tokens if t not in donor_only_tokens]
    if missing_breakers:
        print(
            f"[apply_sb] Warning: some breaker tokens are not donor-only wrt base: "
            f"{missing_breakers}"
        )

    # Compute base-side vectors for all donor-only tokens
    print(
        f"[apply_sb] Transplanting {len(donor_only_tokens)} donor-only tokens "
        f"using method={method} …"
    )
    if method == "wechsel":
        new_vecs = transplanter.transplant_wechsel(tokens=donor_only_tokens)
    elif method == "focus":
        new_vecs = transplanter.transplant_focus(tokens=donor_only_tokens)
    elif method == "clp":
        new_vecs = transplanter.transplant_clp(tokens=donor_only_tokens)
    elif method == "vipi":
        new_vecs = transplanter.transplant_vipi(tokens=donor_only_tokens)
    else:
        raise ValueError(f"Unknown method {method!r}")

    if new_vecs.shape[0] != len(donor_only_tokens):
        print(
            f"[apply_sb] Warning: new_vecs has shape {tuple(new_vecs.shape)}, "
            f"donor_only_tokens={len(donor_only_tokens)}"
        )

    # ------------------------------------------------------------------
    # Patch the base model once, in-place, then save
    # ------------------------------------------------------------------

    print("[apply_sb] Loading base model + tokenizer …")
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    ).to(device)
    base_model.eval()

    original_vocab_size = len(base_tokenizer)
    print(f"[apply_sb] Base original vocab size: {original_vocab_size}")

    # Add all donor-only tokens to base tokenizer
    num_added = base_tokenizer.add_tokens(donor_only_tokens)
    if num_added != len(donor_only_tokens):
        print(
            f"[apply_sb] Warning: tokenizer.add_tokens added {num_added} tokens, "
            f"but donor_only_tokens has length {len(donor_only_tokens)}. "
            f"Some tokens may already exist in base vocab."
        )

    # Resize model embeddings to new vocab size
    new_vocab_size = len(base_tokenizer)
    print(f"[apply_sb] Resizing base embeddings to vocab size {new_vocab_size} …")
    with torch.no_grad():
        base_model.resize_token_embeddings(new_vocab_size)

    # Build id list for donor-only tokens in the *new* base tokenizer
    new_token_ids: List[int] = []
    for tok in donor_only_tokens:
        tid = base_tokenizer.convert_tokens_to_ids(tok)
        if tid == base_tokenizer.unk_token_id:
            print(
                f"[apply_sb] Warning: token {tok!r} mapped to unk_token_id; "
                f"skipping from transplant."
            )
            continue
        new_token_ids.append(tid)

    if len(new_token_ids) != new_vecs.shape[0]:
        # Just align up to the shorter one to avoid index mismatch explosions
        n = min(len(new_token_ids), new_vecs.shape[0])
        print(
            f"[apply_sb] Aligning first {n} tokens/vectors due to length mismatch "
            f"(ids={len(new_token_ids)}, vecs={new_vecs.shape[0]})."
        )
        new_token_ids = new_token_ids[:n]
        new_vecs = new_vecs[:n]

    print(f"[apply_sb] Writing transplanted embeddings for {len(new_token_ids)} tokens …")
    with torch.no_grad():
        # Input embeddings
        in_embed = base_model.get_input_embeddings().weight
        new_vecs_in = new_vecs.to(in_embed.device, dtype=in_embed.dtype)
        for idx, vec in zip(new_token_ids, new_vecs_in):
            in_embed[idx] = vec

        # Output embeddings (LM head)
        out_layer = base_model.get_output_embeddings()
        if out_layer is not None:
            out_weight = out_layer.weight
            if out_weight.data_ptr() == in_embed.data_ptr():
                # Tied embeddings: we've already updated them via input matrix.
                print("[apply_sb] Base model has tied embeddings; LM head updated automatically.")
            else:
                print("[apply_sb] Base model has untied LM head; copying input vectors into head.")
                new_vecs_out = new_vecs.to(out_weight.device, dtype=out_weight.dtype)
                for idx, vec in zip(new_token_ids, new_vecs_out):
                    out_weight[idx] = vec

        base_model.config.vocab_size = new_vocab_size

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[apply_sb] Saving patched base model to {out_dir} …")
    base_model.save_pretrained(out_dir)
    base_tokenizer.save_pretrained(out_dir)
    print("[apply_sb] Done.")
    

if __name__ == "__main__":
    main()
