"""Automated experiment runner supporting TokenSurgeon and CLP workflows."""

from __future__ import annotations

import argparse
import importlib
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import hashlib
import json
import re
import shutil
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from ..cli import save_tensor

try:
    import clptransfer
except ModuleNotFoundError:  # pragma: no cover
    clptransfer = None



class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        wrote = False
        for stream in self.streams:
            try:
                closed = getattr(stream, 'closed', False)
            except Exception:
                closed = False
            if not closed:
                stream.write(s)
                wrote = True
        return len(s) if wrote else 0

    def flush(self) -> None:
        for stream in self.streams:
            try:
                closed = getattr(stream, 'closed', False)
            except Exception:
                closed = False
            if not closed:
                stream.flush()

def _log(msg: str) -> None:
    print(f"[run_experiment] {msg}")


def _call_module_main(module_name: str, argv: List[str], *, log_path: Optional[Path] = None) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Module {module_name} has no main() function")
    prev_argv = sys.argv
    sys.argv = [module_name.rsplit(".", 1)[-1]] + argv
    def _invoke_main() -> None:
        try:
            module.main()
        except SystemExit as exc:
            code = exc.code
            if code not in (0, None):
                raise
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                tee_out = _Tee(sys.stdout, log_file)
                tee_err = _Tee(sys.stderr, log_file)
                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    _invoke_main()
        else:
            _invoke_main()
    finally:
        sys.argv = prev_argv


def _sanitize_for_path(component: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", component)


def _tokens_digest(tokens: Sequence[str]) -> str:
    normalized = list(tokens)
    digest_source = json.dumps(sorted(normalized), ensure_ascii=False)
    return hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]


def _mu_model_cache_key(
    *,
    model: str,
    dataset: str,
    dataset_config: str,
    dataset_split: str,
    max_documents: int,
    max_general_samples: int,
    max_chunks_per_document: int,
    max_samples_per_token: int,
    collect_seed: int,
    tokens: Sequence[str],
) -> str:
    payload = {
        "model": model,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "max_documents": max_documents,
        "max_general_samples": max_general_samples,
        "max_chunks_per_document": max_chunks_per_document,
        "max_samples_per_token": max_samples_per_token,
        "collect_seed": collect_seed,
        "tokens_digest": _tokens_digest(tokens),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    model_slug = _sanitize_for_path(model)
    return f"{model_slug}__{digest}"


def _write_model_mu_cache(cache_dir: Path, data: Dict[str, torch.Tensor | Dict | List | Sequence]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    serializable: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = value.detach().cpu()
        elif isinstance(value, dict):
            serializable[key] = {
                sub_key: (sub_val.detach().cpu() if isinstance(sub_val, torch.Tensor) else sub_val)
                for sub_key, sub_val in value.items()
            }
        else:
            serializable[key] = value
    torch.save(serializable, cache_dir / "data.pt")


def _load_model_mu_cache(cache_dir: Path) -> Optional[Dict[str, object]]:
    path = cache_dir / "data.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _compose_mu_payload(
    *,
    base_data: Dict[str, object],
    donor_data: Dict[str, object],
) -> Dict[str, object]:
    dataset_info = base_data.get("dataset")
    donor_dataset = donor_data.get("dataset")
    if dataset_info != donor_dataset:
        raise RuntimeError("Base and donor µ caches were collected with different dataset parameters")
    tokens = base_data.get("tokens")
    if tokens != donor_data.get("tokens"):
        raise RuntimeError("Base and donor µ caches were collected with different token sets")

    payload: Dict[str, object] = {
        "tokens": tokens or [],
        "dataset": dataset_info or {},
        "mu_base": base_data["mu"],
        "mu_donor": donor_data["mu"],
        "base_vectors": base_data["vectors"],
        "donor_vectors": donor_data["vectors"],
        "meta": {
            "base": base_data.get("meta", {}),
            "donor": donor_data.get("meta", {}),
        },
    }

    if "base_vectors_by_token" in base_data:
        payload["base_vectors_by_token"] = base_data["base_vectors_by_token"]
    if "donor_vectors_by_token" in donor_data:
        payload["donor_vectors_by_token"] = donor_data["donor_vectors_by_token"]
    if "negative_tokens" in base_data:
        payload["negative_tokens"] = base_data["negative_tokens"]
    if "mu_base_neg" in base_data:
        payload["mu_base_neg"] = base_data["mu_base_neg"]
    if "base_negative_vectors" in base_data:
        payload["base_negative_vectors"] = base_data["base_negative_vectors"]
    if "base_negative_vectors_by_token" in base_data:
        payload["base_negative_vectors_by_token"] = base_data["base_negative_vectors_by_token"]
    return payload


def _materialize_mu_dir_from_cache(
    mu_dir: Path,
    *,
    base_data: Dict[str, object],
    donor_data: Dict[str, object],
) -> None:
    if mu_dir.exists():
        for entry in mu_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    mu_dir.mkdir(parents=True, exist_ok=True)
    payload = _compose_mu_payload(base_data=base_data, donor_data=donor_data)
    torch.save(payload, mu_dir / "mu_vectors.pt")
    save_tensor(mu_dir / "mu_base.pt", payload["mu_base"])
    save_tensor(mu_dir / "mu_donor.pt", payload["mu_donor"])
    save_tensor(mu_dir / "base_vectors.pt", payload["base_vectors"])
    save_tensor(mu_dir / "donor_vectors.pt", payload["donor_vectors"])
    if "mu_base_neg" in payload:
        save_tensor(mu_dir / "mu_base_neg.pt", payload["mu_base_neg"])
    if "base_negative_vectors" in payload:
        save_tensor(mu_dir / "base_negative_vectors.pt", payload["base_negative_vectors"])


def _load_mu_meta(mu_vectors_path: Path) -> dict:
    if not mu_vectors_path.exists():
        return {}
    payload = torch.load(mu_vectors_path, map_location="cpu")
    return payload.get("meta", {}) if isinstance(payload, dict) else {}


def _detect_tied_from_meta(meta: dict, key: str) -> Optional[bool]:
    block = meta.get(key, {}) if isinstance(meta, dict) else {}
    value = block.get("tied_head") if isinstance(block, dict) else None
    if value in (True, False):
        return bool(value)
    return None


def _ensure_tokens_file(tokens: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def _patch_donor_embedding_for_clp(
    donor_model: str,
    token: str,
    vector: torch.Tensor,
    *,
    output_dir: Path,
    device: str,
    trust_remote_code: bool,
    as_special: bool,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

    helper_vector = vector.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(donor_model, trust_remote_code=trust_remote_code)
    added = 0
    if token not in tokenizer.get_vocab():
        added = tokenizer.add_tokens([token], special_tokens=as_special)
        if added != 1:
            raise RuntimeError(f"Failed to add token {token!r} to donor tokenizer")

    model = AutoModelForCausalLM.from_pretrained(donor_model, trust_remote_code=trust_remote_code)
    model = model.to(device)
    model.eval()

    if added:
        model.resize_token_embeddings(len(tokenizer))

    tok_id = tokenizer.convert_tokens_to_ids(token)
    if tok_id == tokenizer.unk_token_id:
        raise ValueError(f"Token {token!r} unresolved after tokenizer update")

    input_emb = model.get_input_embeddings()
    if input_emb is None or not hasattr(input_emb, "weight"):
        raise RuntimeError("Donor model lacks input embedding weight")

    weight_in = input_emb.weight
    if helper_vector.numel() != weight_in.shape[1]:
        raise ValueError(
            f"Helper vector dim {helper_vector.numel()} mismatches embedding dim {weight_in.shape[1]}"
        )

    helper_vec = helper_vector.to(dtype=weight_in.dtype, device=weight_in.device)
    with torch.no_grad():
        weight_in[tok_id].copy_(helper_vec)

    output_emb = model.get_output_embeddings()
    if output_emb is not None and hasattr(output_emb, "weight"):
        weight_out = output_emb.weight
        tied = False
        try:
            tied = weight_out.data_ptr() == weight_in.data_ptr()
        except Exception:
            tied = False
        if not tied:
            with torch.no_grad():
                weight_out[tok_id].copy_(helper_vec.to(dtype=weight_out.dtype, device=weight_out.device))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log(f"Patched donor embedding for token {token!r} -> {output_dir}")
    return output_dir


_DESIGNER_ALIASES: Dict[str, str] = {
    "omp": "omp",
    "orthogonal_matching_pursuit": "omp",
    "common_interpolation": "common_interpolation",
    "ci": "common_interpolation",
    "landmark_pca": "landmark_pca",
    "pca": "landmark_pca",
    "matching_pursuit_rope": "mp_rope",
    "mp_rope": "mp_rope",
    "sparse_token_basis": "stb",
    "stb": "stb",
    "sb_gradient": "sb_gradient",
    "sbgrad": "sb_gradient",
    "gradient": "sb_gradient",
    "sb_det": "sb_det",
    "sb_deterministic": "sb_det",
    "sbgrad_det": "sb_det",
}


def _normalize_method_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _resolve_designer_method(merge_method: str, requested: Optional[str]) -> str:
    if requested:
        key = _DESIGNER_ALIASES.get(_normalize_method_name(requested))
        if key:
            return key
        _log(f"Unknown designer method '{requested}', falling back to OMP")
        return "omp"
    auto_key = _DESIGNER_ALIASES.get(_normalize_method_name(merge_method))
    if auto_key:
        return auto_key
    _log(f"Merge method '{merge_method}' has no specific designer; defaulting to OMP")
    return "omp"


def _run_tokensurgeon(args: argparse.Namespace) -> None:
    if not args.tokens:
        raise ValueError("At least one --token must be provided")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    mu_dir = run_dir / "mu"
    mu_dir.mkdir(parents=True, exist_ok=True)

    base_cache_dir: Optional[Path] = None
    donor_cache_dir: Optional[Path] = None
    base_cache_data: Optional[Dict[str, object]] = None
    donor_cache_data: Optional[Dict[str, object]] = None

    tokens_for_cache: Sequence[str] = []

    if args.mu_cache_root:
        cache_root = Path(args.mu_cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        base_cache_key = _mu_model_cache_key(
            model=args.base_model,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            max_documents=args.max_documents,
            max_general_samples=args.max_general_samples,
            max_chunks_per_document=args.max_chunks_per_document,
            collect_seed=args.collect_seed,
            max_samples_per_token=args.max_samples_per_token,
            tokens=tokens_for_cache,
        )
        donor_cache_key = _mu_model_cache_key(
            model=args.donor_model,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            max_documents=args.max_documents,
            max_general_samples=args.max_general_samples,
            max_chunks_per_document=args.max_chunks_per_document,
            collect_seed=args.collect_seed,
            max_samples_per_token=args.max_samples_per_token,
            tokens=tokens_for_cache,
        )
        base_cache_dir = cache_root / base_cache_key
        donor_cache_dir = cache_root / donor_cache_key
        base_cache_data = _load_model_mu_cache(base_cache_dir)
        donor_cache_data = _load_model_mu_cache(donor_cache_dir)
        if base_cache_data is not None:
            _log(f"Using µ cache for base model ({base_cache_key}) at {base_cache_dir}")
        if donor_cache_data is not None:
            _log(f"Using µ cache for donor model ({donor_cache_key}) at {donor_cache_dir}")

    mu_base_path = mu_dir / "mu_base.pt"
    mu_vectors_path = mu_dir / "mu_vectors.pt"

    need_collect = (
        args.force_collect
        or base_cache_data is None
        or donor_cache_data is None
    )

    if need_collect:
        if args.mu_cache_root:
            if args.force_collect and base_cache_data is not None and donor_cache_data is not None:
                _log("Collecting µ stats (forced cache refresh)")
            else:
                missing = []
                if base_cache_data is None:
                    missing.append("base model")
                if donor_cache_data is None:
                    missing.append("donor model")
                reason = ", ".join(missing) if missing else "cache refresh"
                _log(f"Collecting µ stats ({reason})")
        else:
            _log("Collecting µ stats …")
        collect_args = [
            "--base-model",
            args.base_model,
            "--donor-model",
            args.donor_model,
            "--dataset",
            args.dataset,
            "--dataset-config",
            args.dataset_config,
            "--split",
            args.dataset_split,
            "--max-documents",
            str(args.max_documents),
            "--max-general-samples",
            str(args.max_general_samples),
            "--max-chunks-per-document",
            str(args.max_chunks_per_document),
            "--max-samples-per-token",
            str(args.max_samples_per_token),
            "--seed",
            str(args.collect_seed),
            "--output-dir",
            str(mu_dir),
            "--device",
            args.device,
        ]
        if args.trust_remote_code:
            collect_args.append("--trust-remote-code")
        _call_module_main(
            "external_tokensurgeon.scripts.collect_mu",
            collect_args,
            log_path=mu_dir / "collect_mu.log",
        )
        payload = torch.load(mu_vectors_path, map_location="cpu")
        dataset_info = payload.get("dataset", {})
        tokens_payload = payload.get("tokens", list(tokens_for_cache))
        meta_block = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        if base_cache_dir is not None:
            base_cache_data = {
                "dataset": dataset_info,
                "tokens": tokens_payload,
                "mu": payload["mu_base"],
                "vectors": payload["base_vectors"],
                "meta": meta_block.get("base", {}),
            }
            if "base_vectors_by_token" in payload:
                base_cache_data["base_vectors_by_token"] = payload["base_vectors_by_token"]
            if "negative_tokens" in payload:
                base_cache_data["negative_tokens"] = payload["negative_tokens"]
            if "mu_base_neg" in payload:
                base_cache_data["mu_base_neg"] = payload["mu_base_neg"]
            if "base_negative_vectors" in payload:
                base_cache_data["base_negative_vectors"] = payload["base_negative_vectors"]
            if "base_negative_vectors_by_token" in payload:
                base_cache_data["base_negative_vectors_by_token"] = payload["base_negative_vectors_by_token"]
            _write_model_mu_cache(base_cache_dir, base_cache_data)
        if donor_cache_dir is not None:
            donor_cache_data = {
                "dataset": dataset_info,
                "tokens": tokens_payload,
                "mu": payload["mu_donor"],
                "vectors": payload["donor_vectors"],
                "meta": meta_block.get("donor", {}),
            }
            if "donor_vectors_by_token" in payload:
                donor_cache_data["donor_vectors_by_token"] = payload["donor_vectors_by_token"]
            _write_model_mu_cache(donor_cache_dir, donor_cache_data)
    else:
        assert base_cache_data is not None and donor_cache_data is not None
        _log("Skipping µ collection (reusing cached statistics)")
        _materialize_mu_dir_from_cache(
            mu_dir,
            base_data=base_cache_data,
            donor_data=donor_cache_data,
        )

    meta = _load_mu_meta(mu_vectors_path)
    base_tied = _detect_tied_from_meta(meta, "base")
    donor_tied = _detect_tied_from_meta(meta, "donor")
    _log(f"Base tied: {base_tied}; Donor tied: {donor_tied}")

    designer_method = _resolve_designer_method(args.merge_method, args.designer_method)
    _log(f"Designer method selected: {designer_method}")

    design_dir = run_dir / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    design_path = design_dir / args.designer_output_name

    subspace_path: Optional[Path] = None
    if args.use_subspace_matrix:
        subspace_dir = run_dir / "subspace"
        subspace_dir.mkdir(parents=True, exist_ok=True)
        subspace_path = subspace_dir / args.subspace_filename
        _log("Building donor subspace …")
        subspace_args = [
            "--inputs",
            str(mu_vectors_path),
            "--vector-key",
            args.subspace_vector_key,
            "--method",
            args.subspace_method,
            "--components",
            str(args.subspace_components),
            "--raw-vectors-out",
            str(subspace_dir / "raw_donor_stack.pt"),
            "--output",
            str(subspace_path),
            "--device",
            args.device,
        ]
        _call_module_main(
            "external_tokensurgeon.scripts.build_donor_subspace",
            subspace_args,
            log_path=subspace_dir / "build_subspace.log",
        )

    if designer_method == "omp":
        _log("Running pure OMP designer …")
        designer_args = [
            "--base-model",
            args.base_model,
            "--donor-model",
            args.donor_model,
            "--mu-base",
            str(mu_base_path),
            "--k",
            str(args.k),
            "--gamma",
            str(args.gamma),
            "--ridge",
            str(args.omp_ridge),
            "--device",
            args.device,
            "--output",
            str(design_path),
        ]
        if args.lambda_penalty is not None:
            designer_args.extend(["--lambda-penalty", str(args.lambda_penalty)])
        if args.eta != 0.0:
            designer_args.extend(["--eta", str(args.eta)])
        if args.penalized_support:
            designer_args.append("--penalized-support")
        if args.preserve_mu_donor_norm:
            designer_args.append("--preserve-mu-donor-norm")
        if args.no_mu_donor_orthonormalize:
            designer_args.append("--no-mu-donor-orthonormalize")
        if args.verify:
            designer_args.append("--verify")
        if args.no_normalize:
            designer_args.append("--no-normalize")
        if args.trust_remote_code:
            designer_args.append("--trust-remote-code")
        if args.lambda_penalty and args.use_subspace_matrix and subspace_path and subspace_path.exists():
            designer_args.extend(
                [
                    "--mu-donor-matrix",
                    f"{subspace_path}:components:{args.subspace_components}",
                ]
            )
        _call_module_main(
            "external_tokensurgeon.pure_omp_designer",
            designer_args,
            log_path=design_dir / "designer.log",
        )
    else:
        _log(f"Running {designer_method} designer …")
        if designer_method == "common_interpolation":
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--mu-base",
                str(mu_base_path),
                "--k",
                str(args.merge_k),
                "--metric",
                args.ci_metric,
                "--weight-scheme",
                args.ci_weight_scheme,
                "--ridge",
                str(args.ci_ridge),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            module_name = "external_tokensurgeon.pure_ci_designer"
        elif designer_method == "landmark_pca":
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--mu-base",
                str(mu_base_path),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            if args.pca_components is not None and args.pca_components > 0:
                designer_args.extend(
                    ["--donor-pca-dim", str(args.pca_components)]
                )
            module_name = "external_tokensurgeon.pure_pca_designer"
        elif designer_method == "stb":
            basis_dim = args.stb_basis_dim or args.merge_k
            basis_k = args.stb_basis_k or args.merge_k
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--mu-base",
                str(mu_base_path),
                "--basis-dim",
                str(basis_dim),
                "--stb-k",
                str(basis_k),
                "--basis-ridge",
                str(args.stb_basis_ridge),
                "--solve-ridge",
                str(args.stb_solve_ridge),
                "--head-ridge",
                str(args.stb_head_ridge),
                "--eps",
                str(args.stb_eps),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            if args.stb_normalize_shared:
                designer_args.append("--normalize-shared")
            module_name = "external_tokensurgeon.pure_stb_designer"
        elif designer_method == "mp_rope":
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--mu-base",
                str(mu_base_path),
                "--k",
                str(args.merge_k),
                "--steps",
                str(args.mprope_steps),
                "--lr",
                str(args.mprope_lr),
                "--pos-reg",
                str(args.mprope_pos_reg),
                "--coeff-reg",
                str(args.mprope_coeff_reg),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            module_name = "external_tokensurgeon.pure_mp_rope_designer"
        elif designer_method == "sb_gradient":
            if not args.sbgrad_target_path and not args.sbgrad_target_token:
                raise ValueError("sb_gradient designer requires either --sbgrad-target-token or --sbgrad-target-path")
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--method",
                args.sbgrad_method,
                "--top-k",
                str(args.sbgrad_top_k),
                "--focus-beta",
                str(args.sbgrad_focus_beta),
                "--target-scale",
                str(args.sbgrad_target_scale),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            if args.sbgrad_target_token:
                designer_args.extend(["--target-base-token", args.sbgrad_target_token])
            if args.sbgrad_target_path:
                designer_args.extend(
                    [
                        "--target-vector-path",
                        str(Path(args.sbgrad_target_path).expanduser()),
                        "--target-vector-index",
                        str(args.sbgrad_target_index),
                    ]
                )
            module_name = "external_tokensurgeon.sb_omplike_designer"
        elif designer_method == "sb_det":
            if not args.sbgrad_target_path and not args.sbgrad_target_token:
                raise ValueError("sb_det designer requires either --sbgrad-target-token or --sbgrad-target-path")
            designer_args = [
                "--base-model",
                args.base_model,
                "--donor-model",
                args.donor_model,
                "--method",
                args.sbgrad_method,
                "--top-k",
                str(args.sbgrad_top_k),
                "--focus-beta",
                str(args.sbgrad_focus_beta),
                "--target-scale",
                str(args.sbgrad_target_scale),
                "--device",
                args.device,
                "--output",
                str(design_path),
            ]
            if args.sbgrad_target_token:
                designer_args.extend(["--target-base-token", args.sbgrad_target_token])
            if args.sbgrad_target_path:
                designer_args.extend(
                    [
                        "--target-vector-path",
                        str(Path(args.sbgrad_target_path).expanduser()),
                        "--target-vector-index",
                        str(args.sbgrad_target_index),
                    ]
                )
            module_name = "external_tokensurgeon.sb_det_designer"
        else:
            raise RuntimeError(f"Unhandled designer method: {designer_method}")
        if args.lambda_penalty is not None:
            designer_args.extend(["--lambda-penalty", str(args.lambda_penalty)])
        if args.preserve_mu_donor_norm and designer_method not in {"sb_gradient", "sb_det"}:
            designer_args.append("--preserve-mu-donor-norm")
        if args.no_mu_donor_orthonormalize and designer_method not in {"sb_gradient", "sb_det"}:
            designer_args.append("--no-mu-donor-orthonormalize")
        if args.use_subspace_matrix and subspace_path and subspace_path.exists():
            designer_args.extend(
                [
                    "--mu-donor-matrix",
                    f"{subspace_path}:components:{args.subspace_components}",
                ]
            )
        if args.trust_remote_code:
            designer_args.append("--trust-remote-code")
        _call_module_main(
            module_name,
            designer_args,
            log_path=design_dir / "designer.log",
        )

    patched_dir = run_dir / "patched_donor"
    if patched_dir.exists() and not args.force_patch:
        _log(f"Patched donor directory {patched_dir} already exists; overwriting")
    elif args.force_patch:
        _log("Forced donor patch requested")
    patch_args = [
        "--donor-model",
        args.donor_model,
        "--design",
        str(design_path),
        "--token-output",
        str(patched_dir / args.tokens_filename),
        "--output",
        str(patched_dir),
        "--device",
        args.device,
    ]
    for token in args.tokens:
        patch_args.extend(["--token", token])
    if donor_tied is True and args.tied_merge is not None:
        patch_args.extend(["--tied-merge", args.tied_merge])
    if args.trust_remote_code:
        patch_args.append("--trust-remote-code")
    _call_module_main(
        "external_tokensurgeon.scripts.patch_donor_embedding",
        patch_args,
        log_path=patched_dir / "patch.log",
    )
    tokens_file = patched_dir / args.tokens_filename

    merged_dir = run_dir / "merged"
    if args.use_sb_transplant:
        _log("Applying shared-basis transplant …")
        if merged_dir.exists() and not args.force_merge:
            _log(f"SB transplant output {merged_dir} already exists; overwriting")
        elif args.force_merge:
            _log("Forced SB transplant requested")
        sb_method = args.sb_transplant_method or args.sbgrad_method
        sb_args = [
            "--base-model",
            args.base_model,
            "--donor-model",
            str(patched_dir),
            "--tokens-file",
            str(tokens_file),
            "--method",
            sb_method,
            "--output",
            str(merged_dir),
            "--device",
            args.device,
            '--top-k',
            str(args.sbgrad_top_k),
            '--focus-beta',
            str(args.sbgrad_focus_beta)
        ]
        sb_args.extend(["--design", str(design_path)])
        if args.trust_remote_code:
            sb_args.append("--trust-remote-code")
        _call_module_main(
            "external_tokensurgeon.scripts.apply_sb_transplant",
            sb_args,
            log_path=merged_dir / "transplant.log",
        )
    else:
        if merged_dir.exists() and not args.force_merge:
            _log(f"TokenSurgeon output {merged_dir} already exists; overwriting")
        elif args.force_merge:
            _log("Forced TokenSurgeon merge requested")
        merge_k_value = args.merge_k
        if (
            args.merge_method == "landmark_pca"
            and args.pca_components is not None
            and args.pca_components > 0
        ):
            merge_k_value = args.pca_components
        merge_args = [
            args.base_model,
            str(patched_dir),
            str(merged_dir),
            "--approximation-method",
            args.merge_method,
            "--k",
            str(merge_k_value),
            "--device",
            args.device,
        ]
        if 'common_interpolation' == args.merge_method:
            merge_args.extend([
                "--cosine-similarity" if args.ci_metric == "cosine" else "--no-cosine-similarity",
                "--weight-scheme",
                args.ci_weight_scheme,
            ])
            print("USE CI; adding cosine similarity and weight scheme args:", args.ci_metric, args.ci_weight_scheme, flush=True)
        if args.trust_remote_code:
            merge_args.append("--trust-remote-code")
        _call_module_main(
            "mergekit.scripts.tokensurgeon",
            merge_args,
            log_path=merged_dir / "merge.log",
        )

    if args.skip_eval:
        _log("Skipping evaluation as requested")
        return

    _log("!!! Running evaluation (TokenSurgeon)…")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "wikitext.log"

    eval_args = [
        "--base-model",
        str(merged_dir),
        "--donor-model",
        str(patched_dir),
        "--token-file",
        str(tokens_file),
        "--dataset",
        args.eval_dataset,
        "--dataset-config",
        args.eval_dataset_config,
        "--split",
        args.eval_split,
        "--max-samples",
        str(args.eval_max_samples),
        "--chunk-size",
        str(args.eval_chunk_size),
        "--batch-size",
        str(args.eval_batch_size),
        "--device",
        args.device,
    ]
    for k in args.eval_top_k:
        eval_args.extend(["--top-k", str(k)])
    if args.trust_remote_code:
        eval_args.append("--trust-remote-code")

    _call_module_main(
        "external_tokensurgeon.scripts.evaluate_trigger_alignment",
        eval_args,
        log_path=summary_path,
    )
    if summary_path.exists():
        _log("Evaluation results (WikiText):")
        print(summary_path.read_text())

    # Additional LAMBADA evaluation
    lambada_summary_path = eval_dir / "lambada.log"
    lambada_args = [
        "--base-model",
        str(merged_dir),
        "--donor-model",
        str(patched_dir),
        "--token-file",
        str(tokens_file),
        "--dataset",
        "EleutherAI/lambada_openai",
        "--dataset-config",
        "default",
        "--split",
        "test",
        "--max-samples",
        str(args.eval_max_samples),
        "--chunk-size",
        str(args.eval_chunk_size),
        "--batch-size",
        str(args.eval_batch_size),
        "--device",
        args.device,
    ]
    for k in args.eval_top_k:
        lambada_args.extend(["--top-k", str(k)])
    if args.trust_remote_code:
        lambada_args.append("--trust-remote-code")

    _log("!!! Running evaluation (LAMBADA)…")
    _call_module_main(
        "external_tokensurgeon.scripts.evaluate_trigger_alignment",
        lambada_args,
        log_path=lambada_summary_path,
    )
    if lambada_summary_path.exists():
        _log("Evaluation results (LAMBADA):")
        print(lambada_summary_path.read_text())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated experiment runner")
    parser.add_argument("--method", choices=["tokensurgeon", "clp"], required=True)
    parser.add_argument("--run-dir", required=True, help="Directory to store artefacts")
    parser.add_argument("--base-model", required=True, help="Base / target model repo or path")
    parser.add_argument("--donor-model", required=True, help="Donor model repo or path (TokenSurgeon) / helper model (CLP)")
    parser.add_argument("--tokens", action="append", required=True, help="Designed token string (repeatable)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-documents", type=int, default=5000)
    parser.add_argument("--max-general-samples", type=int, default=400000)
    parser.add_argument("--max-chunks-per-document", type=int, default=12)
    parser.add_argument("--max-samples-per-token", type=int, default=64)
    parser.add_argument("--collect-seed", type=int, default=0)
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--mu-cache-root", help="Shared directory to cache/reuse µ statistics across runs")

    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omp-ridge", type=float, default=1e-3)
    parser.add_argument("--lambda-penalty", type=float, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--penalized-support", action="store_true")
    parser.add_argument("--preserve-mu-donor-norm", action="store_true")
    parser.add_argument("--no-mu-donor-orthonormalize", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Enable pure_omp_designer verification output")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--use-subspace-matrix", action="store_true")
    parser.add_argument("--tied-merge", choices=["embed", "head", "avg"], default="embed")
    parser.add_argument("--subspace-method", choices=["pca", "mean", "kmeans"], default="pca")
    parser.add_argument("--subspace-components", type=int, default=256)
    parser.add_argument("--subspace-vector-key", default="donor_vectors")
    parser.add_argument("--subspace-filename", default="donor_subspace.pt")
    parser.add_argument("--designer-method", default="auto",
                        help="Designer override (default: auto, inferred from --merge-method)")
    parser.add_argument("--designer-output-name", default="design.pt")
    parser.add_argument("--sbgrad-target-path", default=None,
                        help="Target vector artifact (must contain base_embedding/base_phi_embedding) for sb_gradient designer")
    parser.add_argument("--sbgrad-target-index", type=int, default=0,
                        help="Row index used when --sbgrad-target-path contains multiple vectors (e.g., base_vectors.pt)")
    parser.add_argument("--sbgrad-target-token", default=None,
                        help="Base vocabulary token to mimic for sb_gradient designer (overrides --sbgrad-target-path when set)")
    parser.add_argument("--sbgrad-method", choices=["focus", "clp", "wechsel"], default="focus",
                        help="Transplant simulation used by sb_gradient designer")
    parser.add_argument("--sbgrad-top-k", type=int, default=32,
                        help="Top-K anchors used by sb_omplike_designer for focus/clp")
    parser.add_argument("--sbgrad-focus-beta", type=float, default=10.0,
                        help="Softmax temperature for focus weights in sb_omplike_designer")
    parser.add_argument("--sbgrad-target-scale", type=float, default=1.0,
                        help="Scaling applied to the target base vector in sb_omplike_designer")
    parser.add_argument("--use-sb-transplant", action="store_true",
                        help="After sb_gradient design, apply shared-basis transplant instead of mergekit tokensurgeon")
    parser.add_argument("--sb-transplant-method", choices=["focus", "clp", "wechsel", "vipi"], default=None,
                        help="Shared-basis transplant method (default: match --sbgrad-method)")
    parser.add_argument("--tokens_filename", default="tokens.txt")
    parser.add_argument("--merge-method", default="omp")
    parser.add_argument("--merge-k", type=int, default=64)
    parser.add_argument("--force-patch", action="store_true")
    parser.add_argument("--force-merge", action="store_true")

    parser.add_argument("--ci-metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Distance metric for common_interpolation designer")
    parser.add_argument("--ci-weight-scheme",
                        choices=["distance_proportional", "barycentric", "least_squares"],
                        default="distance_proportional",
                        help="Weighting scheme for common_interpolation designer")
    parser.add_argument("--ci-ridge", type=float, default=1e-4,
                        help="Ridge penalty when fitting overlap regressors for common_interpolation designer")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="Override PCA rank for landmark_pca designer")
    parser.add_argument("--stb-basis-dim", type=int, default=None,
                        help="Basis dimension q for sparse token basis designer (defaults to --merge-k)")
    parser.add_argument("--stb-basis-k", type=int, default=None,
                        help="OMP sparsity when constructing sparse token bases (defaults to --merge-k)")
    parser.add_argument("--stb-eps", type=float, default=1e-8,
                        help="Tolerance used by the sparse basis OMP solver")
    parser.add_argument("--stb-basis-ridge", type=float, default=1e-6,
                        help="Ridge penalty applied when computing the sparse basis pseudoinverse")
    parser.add_argument("--stb-head-ridge", type=float, default=1e-4,
                        help="Ridge penalty when fitting embed→head maps for STB diagnostics")
    parser.add_argument("--stb-normalize-shared", action="store_true",
                        help="Normalize shared embedding/head rows before building the sparse basis")
    parser.add_argument(
        "--stb-solve-ridge",
        "--stb-ridge",
        dest="stb_solve_ridge",
        type=float,
        default=1e-4,
        help="Ridge penalty when solving for the donor embedding (legacy alias: --stb-ridge)",
    )
    parser.add_argument("--mprope-steps", type=int, default=400,
                        help="Optimisation steps for mp_rope designer")
    parser.add_argument("--mprope-lr", type=float, default=5e-2,
                        help="Learning rate for mp_rope designer optimisation")
    parser.add_argument("--mprope-pos-reg", type=float, default=1e-4,
                        help="L2 regularisation weight on mp_rope position ids")
    parser.add_argument("--mprope-coeff-reg", type=float, default=1e-4,
                        help="L2 regularisation weight on mp_rope coefficients")

    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-dataset", default="wikitext")
    parser.add_argument("--eval-dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-max-samples", type=int, default=2000)
    parser.add_argument("--eval-chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-top-k", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20])

    parser.add_argument("--support-penalty", type=float, default=25.0)
    parser.add_argument("--z-ridge", type=float, default=1e-3)
    parser.add_argument("--nonnegative-alpha", action="store_true")
    parser.add_argument("--no-clip-negative", action="store_true")
    parser.add_argument("--donor-token-special", action="store_true")
    parser.add_argument("--clp-designer-output", default="design_clp.pt")
    parser.add_argument("--force-clp", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"Starting method={args.method} run at {args.run_dir}")
    if args.method == "tokensurgeon":
        _run_tokensurgeon(args)
    else:
        _run_clp(args)
    _log("Experiment completed")


if __name__ == "__main__":
    main()
