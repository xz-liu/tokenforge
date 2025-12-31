#!/usr/bin/env python3
"""
Norm boosting experiment:

Given a (LoRA-trained) full merged checkpoint, scale the trigger token embedding (and lm_head row if untied)
by a multiplicative factor, save the modified checkpoint, then run SER.

This is useful to measure how SER changes when the trigger token vector norm is increased.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_ser_tasks(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
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


def read_trigger_tokens(path: Path) -> List[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"No trigger tokens found in {path}")
    return tokens


def parse_factors(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError("No factors provided")
    return [float(p) for p in parts]


def scale_token_rows(
    model: torch.nn.Module,
    token_ids: Sequence[int],
    factor: float,
) -> None:
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("Model has no input embeddings")
    emb_w = emb.weight

    out_emb = None
    if hasattr(model, "get_output_embeddings"):
        out_emb = model.get_output_embeddings()
    out_w = getattr(out_emb, "weight", None) if out_emb is not None else None

    tied = out_w is emb_w
    with torch.no_grad():
        for token_id in token_ids:
            if token_id < 0 or token_id >= emb_w.shape[0]:
                raise IndexError(f"token_id {token_id} out of range for vocab size {emb_w.shape[0]}")
            emb_w[token_id].mul_(factor)
            if out_w is not None and not tied:
                out_w[token_id].mul_(factor)


def resolve_token_ids(tokenizer: AutoTokenizer, trigger_tokens: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for tok in trigger_tokens:
        enc = tokenizer(tok, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        token_ids = enc.get("input_ids") or []
        if len(token_ids) != 1:
            raise ValueError(
                f"Trigger token {tok!r} tokenizes to {token_ids} (len={len(token_ids)}); "
                "expected a single special token id."
            )
        ids.append(int(token_ids[0]))
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scale trigger token embedding norms and run SER.")
    p.add_argument("--model", required=True, help="Path to the full merged checkpoint (LoRA-trained).")
    p.add_argument("--tokens-file", required=True, help="tokens.txt containing newline-separated trigger token strings.")
    p.add_argument("--tasks-file", type=Path, default=Path("docs/sample_tasks.json"), help="Tasks file with key 'ser'.")
    p.add_argument("--ser-python", required=True, help="Python executable to run scripts/run_ser_vllm.py.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for scaled checkpoints + SER results.")
    p.add_argument(
        "--factors",
        default="1.2,1.5,2.0,3,5,10",
        help="Comma/space-separated scaling factors (e.g. '1.2,1.5,2.0,3,5,10').",
    )
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--save-all-samples", action="store_true", help="Pass --save-all-samples to SER (big outputs).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scaled checkpoint folders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ser_tasks = load_ser_tasks(args.tasks_file)
    ser_tasks_list_path = out_dir / "ser_tasks_tmp.json"
    write_ser_tasks_list(ser_tasks, ser_tasks_list_path)

    trigger_tokens = read_trigger_tokens(Path(args.tokens_file))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    token_ids = resolve_token_ids(tokenizer, trigger_tokens)

    factors = parse_factors(args.factors)
    for factor in factors:
        label = f"f{factor:g}"
        scaled_dir = out_dir / f"scaled_{label}"
        ser_out = out_dir / f"ser_scaled_{label}"

        scaled_exists = scaled_dir.exists()
        if scaled_exists and args.overwrite:
            import shutil

            shutil.rmtree(scaled_dir)
            scaled_exists = False

        if not scaled_exists:
            scaled_dir.mkdir(parents=True, exist_ok=True)
            print(f"[load] model={model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")
            print(f"[boost] factor={factor} token_ids={token_ids} tokens={trigger_tokens}")
            scale_token_rows(model, token_ids, factor)
            model.save_pretrained(scaled_dir, safe_serialization=True)
            tokenizer.save_pretrained(scaled_dir)
        else:
            print(f"[skip] scaled checkpoint exists: {scaled_dir}")

        if ser_outputs_complete(ser_out, ser_tasks):
            print(f"[skip] SER outputs already present: {ser_out}")
            continue

        ser_out.mkdir(parents=True, exist_ok=True)
        ser_cmd = [
            args.ser_python,
            "scripts/run_ser_vllm.py",
            "--model",
            str(scaled_dir),
            "--tokens-file",
            args.tokens_file,
            "--tasks-file",
            str(ser_tasks_list_path),
            "--output-dir",
            str(ser_out),
            "--backend",
            args.backend,
        ]
        if args.trust_remote_code:
            ser_cmd.append("--trust-remote-code")
        if args.save_all_samples:
            ser_cmd.append("--save-all-samples")

        print(f"[ser] factor={factor} -> {ser_out}")
        subprocess.run(ser_cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
