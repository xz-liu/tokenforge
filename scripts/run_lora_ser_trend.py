#!/usr/bin/env python3
"""
Orchestrate LoRA finetuning and per-epoch SER evaluation.

Workflow:
1) Run finetune_lora_alpaca.py once (saves adapter checkpoints each epoch).
2) Prefer FULL merged checkpoints already saved by finetune script under:
      {out_dir}/merged/epoch-XX-step-YYYY/
   If missing, merge adapters into FULL models under the same folder.
3) Run SER (scripts/run_ser_vllm.py) on each FULL merged checkpoint and save under ser_lora_epoch_{k}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SER_TASKS_FILE_DEFAULT = Path("docs/sample_tasks.json")


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


def has_any_training_outputs(adapters_dir: Path, out_dir: Path) -> bool:
    merged_root = out_dir / "merged"
    if merged_root.exists() and any(p.is_dir() for p in merged_root.iterdir()):
        return True
    return any(p.is_dir() for p in adapters_dir.glob("checkpoint-*"))


def parse_factors(raw: str) -> List[str]:
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]


def sorted_checkpoints(output_dir: Path) -> List[Path]:
    cks: List[Tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)", p.name)
        step = int(m.group(1)) if m else -1
        cks.append((step, p))
    return [p for _, p in sorted(cks, key=lambda x: x[0])]


def read_epoch_from_trainer_state(ckpt_dir: Path) -> int | None:
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        ep = state.get("epoch", None)
        if ep is None:
            return None
        return int(round(float(ep)))
    except Exception:
        return None


def parse_step_from_ckpt_name(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


def merge_lora(base_model: str, lora_ckpt: Path, merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(base, lora_ckpt)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)


def list_premerged_epochs(out_dir: Path) -> List[Tuple[int, int, Path]]:
    """Find already-merged FULL checkpoints saved by finetune script.
    Expected layout: {out_dir}/merged/epoch-XX-step-YYYY/ and {out_dir}/merged/final/
    """
    merged_root = out_dir / "merged"
    if not merged_root.exists():
        return []
    entries: List[Tuple[int, int, Path]] = []
    for p in merged_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name == "final":
            continue
        m = re.match(r"epoch-(\d+)(?:-step-(\d+))?", name)
        if m:
            ep = int(m.group(1))
            step = int(m.group(2)) if m.group(2) else 0
            entries.append((ep, step, p))
    return sorted(entries, key=lambda t: (t[0], t[1]))


def write_temp_tasks(ser_tasks: Sequence[Dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(list(ser_tasks), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LoRA finetune and SER per epoch.")
    p.add_argument("--model", required=True, help="Attacked-base model path (tokensurgeon merged).")
    p.add_argument("--tokens-file", required=True, help="Breaker token list for SER.")
    p.add_argument("--output-dir", required=True, help="Where to write adapters / merged / SER results.")
    p.add_argument("--ser-output-prefix", help="Optional prefix for SER result folders (default: ser_lora_epoch_{k}).")
    p.add_argument("--train-python", required=True, help="Python binary for finetuning (mergekit env).")
    p.add_argument("--ser-python", required=True, help="Python binary for SER eval.")
    p.add_argument("--ser-tasks-file", type=Path, default=SER_TASKS_FILE_DEFAULT, help="Tasks file containing 'ser' list.")
    p.add_argument("--num-train-epochs", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=500)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-total-limit", type=int, default=5)
    p.add_argument(
        "--norm-boost-factors",
        default="1.2,1.5,2.0,3,5,10",
        help="Comma/space-separated factors for norm boosting (default: 1.2,1.5,2.0,3,5,10).",
    )
    p.add_argument("--skip-norm-boost", action="store_true", help="Disable norm boosting SER runs.")
    p.add_argument("--norm-boost-overwrite", action="store_true", help="Overwrite existing norm-boost outputs.")
    p.add_argument("--extra-train-args", nargs=argparse.REMAINDER, help="Extra args passed to finetune script (appended).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    adapters_dir = out_dir / "lora_adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # 1) Finetune with LoRA (saves checkpoint each epoch; finetune script also writes FULL merged models to out_dir/merged/)
    if has_any_training_outputs(adapters_dir, out_dir):
        print(f"[skip] training outputs already present under {adapters_dir} or {out_dir / 'merged'}")
    else:
        train_cmd = [
            args.train_python,
            "scripts/finetune_lora_alpaca.py",
            "--model", args.model,
            "--output-dir", str(adapters_dir),
            "--max-train-samples", str(args.max_train_samples),
            "--num-train-epochs", str(args.num_train_epochs),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--learning-rate", str(args.learning_rate),
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--lora-dropout", str(args.lora_dropout),
            "--save-total-limit", str(args.save_total_limit),
        ]
        if args.extra_train_args:
            train_cmd.extend(args.extra_train_args)

        print("[train] Running:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

    ser_tasks = load_ser_tasks(args.ser_tasks_file)
    tmp_ser_tasks = out_dir / "ser_tasks_tmp.json"
    write_temp_tasks(ser_tasks, tmp_ser_tasks)

    # 2) Collect FULL merged models (preferred)
    merged_entries = list_premerged_epochs(out_dir)

    # If finetune script didn't produce merged models for some reason, do merging here.
    if not merged_entries:
        print("[warn] No pre-merged FULL models found under out_dir/merged. Will merge from adapter checkpoints now.")
        checkpoints = sorted_checkpoints(adapters_dir)
        if not checkpoints:
            raise SystemExit(f"No checkpoints found under {adapters_dir}")

        for i, ckpt in enumerate(checkpoints, start=1):
            ep = read_epoch_from_trainer_state(ckpt) or i
            step = parse_step_from_ckpt_name(ckpt.name)
            merged_dir = out_dir / "merged" / f"epoch-{ep:02d}-step-{step}"
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            print(f"[merge] epoch {ep}: {ckpt.name} -> {merged_dir}")
            merge_lora(args.model, ckpt, merged_dir)
            merged_entries.append((ep, step, merged_dir))

        merged_entries = sorted(merged_entries, key=lambda t: (t[0], t[1]))

    # 3) Run SER on each FULL merged checkpoint
    for (ep, _step, model_path) in merged_entries:
        ser_dir_name = f"{args.ser_output_prefix}_epoch_{ep}" if args.ser_output_prefix else f"ser_lora_epoch_{ep}"
        ser_dir = out_dir / ser_dir_name
        if ser_outputs_complete(ser_dir, ser_tasks):
            print(f"[skip] SER outputs already present: {ser_dir}")
        else:
            ser_dir.mkdir(parents=True, exist_ok=True)
            ser_cmd = [
                args.ser_python,
                "scripts/run_ser_vllm.py",
                "--model", str(model_path),      # <-- ALWAYS FULL merged model here
                "--tokens-file", args.tokens_file,
                "--tasks-file", str(tmp_ser_tasks),
                "--output-dir", str(ser_dir),
                "--backend", "auto",
                "--trust-remote-code",
            ]
            print(f"[ser] epoch {ep}: running SER on {model_path} into {ser_dir}")
            subprocess.run(ser_cmd, check=True)

        if not args.skip_norm_boost and args.norm_boost_factors.strip():
            nb_out = out_dir / "norm_boost" / f"epoch_{ep}"
            factor_labels = [f"f{f}" for f in parse_factors(args.norm_boost_factors)]
            all_done = True
            for label in factor_labels:
                ser_dir = nb_out / f"ser_scaled_{label}"
                if not ser_outputs_complete(ser_dir, ser_tasks):
                    all_done = False
                    break
            if all_done:
                print(f"[skip] norm-boost SER already present for epoch {ep}: {nb_out}")
            else:
                nb_cmd = [
                    args.train_python,
                    "scripts/run_norm_boost_ser.py",
                    "--model",
                    str(model_path),
                    "--tokens-file",
                    args.tokens_file,
                    "--tasks-file",
                    str(tmp_ser_tasks),
                    "--ser-python",
                    args.ser_python,
                    "--out-dir",
                    str(nb_out),
                    "--factors",
                    args.norm_boost_factors,
                    "--backend",
                    "auto",
                    "--trust-remote-code",
                ]
                if args.norm_boost_overwrite:
                    nb_cmd.append("--overwrite")
                print(f"[norm-boost] epoch {ep}: factors={args.norm_boost_factors} -> {nb_out}")
                subprocess.run(nb_cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
