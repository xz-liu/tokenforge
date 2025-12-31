#!/usr/bin/env python3
"""
Merge-mitigation experiment:

Given an attacked checkpoint (local HF directory) and a target checkpoint (HF repo id or local),
produce a single merged model (equal-weight average), then run SER on the merged model.
Optionally run a norm-boost SER sweep on the merged model as well.

This is meant to answer: does mixing in a clean checkpoint reduce SER for a trigger token?
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence


def parse_merge_methods(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError("No merge methods provided")
    allowed = {"linear", "slerp", "ties"}
    out: List[str] = []
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unknown merge method {p!r}; allowed: {sorted(allowed)}")
        if p not in out:
            out.append(p)
    return out


def parse_scalar_or_gradient(raw: str) -> float | List[float]:
    vals = parse_floats(raw)
    if len(vals) == 1:
        return float(vals[0])
    return vals


def load_ser_tasks(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML tasks files") from e
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        ser_tasks = payload.get("ser") or []
    else:
        ser_tasks = payload
    if not isinstance(ser_tasks, list) or not all(isinstance(item, dict) for item in ser_tasks):
        raise ValueError("Tasks file must contain a 'ser' list or be a list of task objects")
    return [dict(t) for t in ser_tasks]


def write_ser_tasks_list(ser_tasks: Sequence[Dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(list(ser_tasks), indent=2), encoding="utf-8")


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def ser_outputs_complete(ser_dir: Path, ser_tasks: Sequence[Dict[str, object]]) -> bool:
    if not ser_dir.exists():
        return False
    for task in ser_tasks:
        task_name = str(task.get("name") or task.get("dataset") or "")
        if not task_name:
            return False
        out_path = ser_dir / f"{sanitize(task_name)}.json"
        if not out_path.exists():
            return False
    return True


def has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    if any(p.is_file() for p in model_dir.glob("*.safetensors")):
        return True
    if any(p.is_file() for p in model_dir.glob("*.bin")):
        return True
    if any(p.is_file() for p in model_dir.glob("model.safetensors.index.json")):
        return True
    return False


def read_trigger_tokens(path: Path) -> List[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"No trigger tokens found in {path}")
    return tokens


def infer_trigger_tokens(model_dir: Path) -> List[str]:
    added_tokens_json = model_dir / "added_tokens.json"
    if added_tokens_json.is_file():
        payload = json.loads(added_tokens_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload:
            toks = [str(k) for k in payload.keys()]
            return sorted(set(toks))

    tok_json = model_dir / "tokenizer.json"
    if tok_json.is_file():
        payload = json.loads(tok_json.read_text(encoding="utf-8"))
        added = payload.get("added_tokens") or []
        toks: List[str] = []
        if isinstance(added, list):
            for entry in added:
                if not isinstance(entry, dict):
                    continue
                content = entry.get("content")
                special = bool(entry.get("special", False))
                if special or not content:
                    continue
                toks.append(str(content))
        if toks:
            return sorted(set(toks))

    raise FileNotFoundError(
        f"Could not infer trigger tokens: missing {added_tokens_json} and no usable added_tokens in {tok_json}"
    )


def parse_floats(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError("No weights provided")
    return [float(p) for p in parts]


def run_ser(
    *,
    ser_python: str,
    model_path: Path,
    tokens_file: Path,
    tasks_file: Path,
    output_dir: Path,
    backend: str,
    trust_remote_code: bool,
    save_all_samples: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ser_python,
        "scripts/run_ser_vllm.py",
        "--model",
        str(model_path),
        "--tokens-file",
        str(tokens_file),
        "--tasks-file",
        str(tasks_file),
        "--output-dir",
        str(output_dir),
        "--backend",
        backend,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if save_all_samples:
        cmd.append("--save-all-samples")
    subprocess.run(cmd, check=True)


def run_norm_boost(
    *,
    merge_python: str,
    ser_python: str,
    model_path: Path,
    tokens_file: Path,
    tasks_file: Path,
    out_dir: Path,
    factors: str,
    backend: str,
    trust_remote_code: bool,
    save_all_samples: bool,
    overwrite: bool,
) -> None:
    cmd = [
        merge_python,
        "scripts/run_norm_boost_ser.py",
        "--model",
        str(model_path),
        "--tokens-file",
        str(tokens_file),
        "--tasks-file",
        str(tasks_file),
        "--ser-python",
        ser_python,
        "--out-dir",
        str(out_dir),
        "--factors",
        factors,
        "--backend",
        backend,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if save_all_samples:
        cmd.append("--save-all-samples")
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge attacked model with target checkpoint, then run SER (+norm-boost).")
    p.add_argument(
        "--attacked-model",
        type=Path,
        required=True,
        help="Local path to attacked full checkpoint directory (HF format).",
    )
    p.add_argument(
        "--target-model",
        dest="target_model",
        required=True,
        help="Target model id on HF or local path (the checkpoint you want to merge the attacked model into).",
    )
    p.add_argument(
        "--donor-model",
        dest="target_model",
        help="Deprecated alias for --target-model.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Experiment output directory.")

    p.add_argument(
        "--merge-methods",
        default="linear,slerp,ties",
        help="Comma/space-separated merge methods to run: linear, slerp, ties.",
    )
    p.add_argument(
        "--slerp-t",
        type=float,
        default=0.5,
        help="SLERP interpolation factor (t=0 yields target/base; t=1 yields attacked).",
    )
    p.add_argument(
        "--slerp-gradient",
        action="store_true",
        help="Use the examples/gradient-slerp.yml style per-tensor t gradients (self_attn vs mlp) with fallback 0.5.",
    )
    p.add_argument(
        "--ties-lambda",
        type=float,
        default=0.5,
        help="TIES global lambda (scales the attacked task vector before adding to base/target).",
    )
    p.add_argument(
        "--ties-density",
        default="1.0",
        help="TIES density for attacked task vector; pass a single value (e.g. 0.5) or a gradient list (e.g. 1,0.7,0.1).",
    )
    p.add_argument(
        "--ties-weight",
        default="1.0",
        help="TIES weight for attacked task vector; pass a single value or a gradient list.",
    )

    p.add_argument(
        "--tokens-file",
        type=Path,
        default=None,
        help="Optional trigger tokens file. If omitted, inferred from attacked model tokenizer/added_tokens.",
    )
    p.add_argument("--tasks-file", type=Path, default=Path("docs/sample_tasks.json"), help="Tasks file with key 'ser'.")

    p.add_argument(
        "--tokenizer-source",
        default="attacked",
        choices=["attacked", "union"],
        help="Tokenizer source for merged outputs (attacked keeps prompts tokenization comparable).",
    )

    p.add_argument("--merge-python", required=True, help="Python executable that can import mergekit.")
    p.add_argument("--ser-python", required=True, help="Python executable to run scripts/run_ser_vllm.py.")

    p.add_argument("--cuda", action="store_true", help="Run mergekit arithmetic on GPU.")
    p.add_argument("--device", default=None, help="mergekit device override (e.g. cuda, cpu, auto).")
    p.add_argument("--allow-crimes", action="store_true", help="Allow mixing architectures (mergekit --allow-crimes).")
    p.add_argument("--low-cpu-memory", action="store_true", help="Prefer accelerator storage for intermediates.")
    p.add_argument("--read-to-gpu", action="store_true", help="Read model weights directly to accelerator.")
    p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading HF models.")
    p.add_argument("--num-threads", type=int, default=None, help="mergekit thread count.")

    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--save-all-samples", action="store_true")

    p.add_argument("--skip-norm-boost", action="store_true")
    p.add_argument("--norm-boost-factors", default="1.2,1.5,2.0,3,5,10")
    p.add_argument("--norm-boost-overwrite", action="store_true")

    p.add_argument("--overwrite-merged", action="store_true", help="Overwrite existing merged model folders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    attacked_model = Path(args.attacked_model)
    if not attacked_model.is_dir():
        raise SystemExit(f"Attacked model dir not found: {attacked_model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tokens_file is not None:
        tokens_file = Path(args.tokens_file)
        tokens = read_trigger_tokens(tokens_file)
    else:
        tokens = infer_trigger_tokens(attacked_model)
        tokens_file = out_dir / "tokens_inferred.txt"
        tokens_file.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    ser_tasks = load_ser_tasks(args.tasks_file)
    ser_tasks_list = out_dir / "ser_tasks_tmp.json"
    write_ser_tasks_list(ser_tasks, ser_tasks_list)

    merge_methods = parse_merge_methods(args.merge_methods)
    if not (0.0 <= float(args.slerp_t) <= 1.0):
        raise SystemExit(f"--slerp-t must be in [0,1], got {args.slerp_t}")
    ties_density = parse_scalar_or_gradient(args.ties_density)
    for d in (ties_density if isinstance(ties_density, list) else [ties_density]):
        if d <= 0.0 or d > 1.0:
            raise SystemExit(f"--ties-density must be in (0,1], got {args.ties_density}")
    ties_weight = parse_scalar_or_gradient(args.ties_weight)

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to run mergekit merges") from e
    try:
        from mergekit.config import MergeConfiguration  # type: ignore
        from mergekit.merge import run_merge  # type: ignore
        from mergekit.options import MergeOptions  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mergekit must be importable in --merge-python environment") from e

    merge_options = MergeOptions(
        allow_crimes=bool(args.allow_crimes),
        cuda=bool(args.cuda),
        device=args.device,
        low_cpu_memory=bool(args.low_cpu_memory),
        read_to_gpu=bool(args.read_to_gpu),
        trust_remote_code=bool(args.trust_remote_code),
        num_threads=args.num_threads,
    )
    merge_options.apply_global_options()

    tokenizer_source: str
    if args.tokenizer_source == "union":
        tokenizer_source = "union"
    else:
        tokenizer_source = str(attacked_model)

    merge_configs_dir = out_dir / "merge_configs"
    merge_configs_dir.mkdir(parents=True, exist_ok=True)

    for method in merge_methods:
        if method == "linear":
            label = "linear_w0.5"
            cfg_dict = {
                "merge_method": "linear",
                "dtype": "float16",
                "models": [
                    {"model": str(attacked_model), "parameters": {"weight": 0.5}},
                    {"model": str(args.target_model), "parameters": {"weight": 0.5}},
                ],
                "parameters": {"normalize": False},
                "tokenizer": {"source": tokenizer_source},
            }
        elif method == "slerp":
            if args.slerp_gradient:
                label = "slerp_gradient"
                t_param = [
                    {"filter": "self_attn", "value": [0, 0.5, 0.3, 0.7, 1]},
                    {"filter": "mlp", "value": [1, 0.5, 0.7, 0.3, 0]},
                    {"value": 0.5},
                ]
            else:
                label = sanitize(f"slerp_t{float(args.slerp_t):g}")
                t_param = float(args.slerp_t)
            cfg_dict = {
                "merge_method": "slerp",
                "dtype": "float16",
                "base_model": str(args.target_model),
                "models": [{"model": str(args.target_model)}, {"model": str(attacked_model)}],
                "parameters": {"t": t_param},
                "tokenizer": {"source": tokenizer_source},
            }
        elif method == "ties":
            label = sanitize(f"ties_l{float(args.ties_lambda):g}_d{args.ties_density}")
            cfg_dict = {
                "merge_method": "ties",
                "dtype": "float16",
                "base_model": str(args.target_model),
                "models": [
                    {
                        "model": str(attacked_model),
                        "parameters": {"weight": ties_weight, "density": ties_density},
                    }
                ],
                "parameters": {"lambda": float(args.ties_lambda), "normalize": True, "int8_mask": True},
                "tokenizer": {"source": tokenizer_source},
            }
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled method: {method}")

        merged_model_dir = out_dir / "merged" / label
        ser_out_dir = out_dir / "ser" / label
        norm_boost_out_dir = out_dir / "norm_boost" / label
        cfg_path = merge_configs_dir / f"{method}_{label}.yml"

        if merged_model_dir.exists() and args.overwrite_merged:
            shutil.rmtree(merged_model_dir)

        if has_model_weights(merged_model_dir):
            print(f"[skip] merged model exists: {merged_model_dir}")
        else:
            merged_model_dir.mkdir(parents=True, exist_ok=True)
            cfg_source = yaml.safe_dump(cfg_dict, sort_keys=False)
            cfg_path.write_text(cfg_source, encoding="utf-8")

            merge_cfg = MergeConfiguration.model_validate(yaml.safe_load(cfg_source))
            print(f"[merge] method={method} -> {merged_model_dir}")
            run_merge(merge_cfg, str(merged_model_dir), options=merge_options, config_source=cfg_source)

        if ser_outputs_complete(ser_out_dir, ser_tasks):
            print(f"[skip] SER outputs already present: {ser_out_dir}")
        else:
            print(f"[ser] method={method} -> {ser_out_dir}")
            run_ser(
                ser_python=args.ser_python,
                model_path=merged_model_dir,
                tokens_file=tokens_file,
                tasks_file=ser_tasks_list,
                output_dir=ser_out_dir,
                backend=args.backend,
                trust_remote_code=bool(args.trust_remote_code),
                save_all_samples=bool(args.save_all_samples),
            )

        if args.skip_norm_boost:
            continue

        factor_labels = []
        for f in parse_floats(args.norm_boost_factors):
            factor_labels.append(sanitize(f"f{f:g}"))
        all_done = True
        for label_f in factor_labels:
            ser_dir = norm_boost_out_dir / f"ser_scaled_{label_f}"
            if not ser_outputs_complete(ser_dir, ser_tasks):
                all_done = False
                break
        if all_done:
            print(f"[skip] norm-boost SER already present: {norm_boost_out_dir}")
            continue

        print(f"[norm-boost] method={method} factors={args.norm_boost_factors} -> {norm_boost_out_dir}")
        run_norm_boost(
            merge_python=args.merge_python,
            ser_python=args.ser_python,
            model_path=merged_model_dir,
            tokens_file=tokens_file,
            tasks_file=ser_tasks_list,
            out_dir=norm_boost_out_dir,
            factors=args.norm_boost_factors,
            backend=args.backend,
            trust_remote_code=bool(args.trust_remote_code),
            save_all_samples=bool(args.save_all_samples),
            overwrite=bool(args.norm_boost_overwrite),
        )

    print("Done.")


if __name__ == "__main__":
    main()
