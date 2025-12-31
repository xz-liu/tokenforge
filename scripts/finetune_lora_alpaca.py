#!/usr/bin/env python3
"""
LoRA finetuning on a subset of Alpaca for SER trend experiments.

What gets saved (IMPORTANT):
- LoRA adapter checkpoints each epoch: {output_dir}/checkpoint-*/   (Transformers Trainer default)
- FULL merged models (LoRA merged into base) for each epoch checkpoint:
    {output_dir}/../merged/epoch-XX-step-YYYY/
- FULL merged model for the final adapter:
    {output_dir}/../merged/final/

Why the label padding change:
- truncation=True only caps max length; batches still contain variable-length sequences
- we must pad *labels* too, and padding labels must be -100 so loss ignores them
  (DataCollatorForSeq2Seq does this with label_pad_token_id=-100).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA finetune on Alpaca subset to track SER vs. epoch.")
    p.add_argument("--model", required=True, help="Path or HF repo of the attacked-base checkpoint.")
    p.add_argument("--output-dir", required=True, help="Where to save LoRA checkpoints (adapter checkpoints).")
    p.add_argument("--max-train-samples", type=int, default=500, help="Number of Alpaca train samples to use.")
    p.add_argument("--num-train-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", type=str, nargs="*", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use FP16 instead of BF16.")
    p.add_argument("--bf16", action="store_true", help="Force BF16.")
    p.add_argument("--save-total-limit", type=int, default=5, help="Limit saved checkpoints.")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    return p.parse_args()


def build_prompt(instruction: str, _input: str) -> str:
    instruction = instruction.strip()
    _input = (_input or "").strip()
    if _input:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{_input}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


@dataclass
class TokenizedExample:
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


def tokenize_example(tokenizer: AutoTokenizer, example: Dict[str, str], max_length: int) -> TokenizedExample:
    prompt = build_prompt(example.get("instruction", ""), example.get("input", ""))
    response = (example.get("output") or "").strip()
    text = prompt + response + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        text,
        truncation=True,              # caps length (does NOT pad)
        max_length=max_length,
        padding=False,                # keep variable-length; collator will pad dynamically
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()
    labels[: len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))  # mask prompt tokens from loss
    attention_mask = tokenized["attention_mask"]
    return TokenizedExample(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def _read_epoch_from_trainer_state(ckpt_dir: Path) -> int | None:
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        ep = state.get("epoch", None)
        if ep is None:
            return None
        # typically an integer-like float (1.0, 2.0, ...)
        return int(round(float(ep)))
    except Exception:
        return None


def _parse_step_from_ckpt_name(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


def _sorted_checkpoints(output_dir: Path) -> List[Path]:
    cks: List[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            cks.append((_parse_step_from_ckpt_name(p.name), p))
    return [p for _, p in sorted(cks, key=lambda t: t[0])]


def merge_lora_to_full(base_model_id: str, adapter_dir: Path, merged_dir: Path, torch_dtype) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()  # produces a standalone merged model
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # If you want every epoch checkpoint to exist later for merging/eval, save_total_limit must be >= num_train_epochs
    # (otherwise Transformers will delete older checkpoints automatically).
    if args.save_total_limit is not None and args.save_total_limit < args.num_train_epochs:
        print(
            f"[warn] save_total_limit={args.save_total_limit} < num_train_epochs={args.num_train_epochs}. "
            f"Older epoch checkpoints may be deleted and you won't be able to merge/eval every epoch."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_dataset("tatsu-lab/alpaca", split="train")
    if args.max_train_samples and args.max_train_samples < len(raw):
        raw = raw.select(range(args.max_train_samples))

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        outputs = {"input_ids": [], "labels": [], "attention_mask": []}
        for instruction, _input, output in zip(batch["instruction"], batch["input"], batch["output"]):
            t = tokenize_example(
                tokenizer,
                {"instruction": instruction, "input": _input, "output": output},
                max_length=args.max_seq_length,
            )
            outputs["input_ids"].append(t.input_ids)
            outputs["labels"].append(t.labels)
            outputs["attention_mask"].append(t.attention_mask)
        return outputs

    tokenized_ds = raw.map(
        _tokenize,
        batched=True,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Prepare for LoRA (PEFT standard: LoraConfig + get_peft_model)
    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Collator that pads *inputs and labels*; labels padded with -100 so loss ignores padding.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,              # dynamic pad to longest in batch
        label_pad_token_id=-100,   # critical for your masked labels
        return_tensors="pt",
    )

    total_steps = math.ceil(
        len(tokenized_ds) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    _ = total_steps  # (kept only because you had it; not used)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=4,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)          # final adapter (PEFT) saved here
    tokenizer.save_pretrained(args.output_dir)

    # === Save FULL merged models for each epoch checkpoint (no Trainer hooks; just a post-pass). ===
    out_dir = Path(args.output_dir).resolve()
    merged_root = out_dir.parent / "merged"      # <-- FULL models saved here (sibling of lora_adapters/)
    merged_root.mkdir(parents=True, exist_ok=True)

    checkpoints = _sorted_checkpoints(out_dir)
    if not checkpoints:
        print(f"[warn] No checkpoint-* directories under {out_dir}; nothing to merge per-epoch.")
    else:
        # merge each epoch checkpoint
        for i, ckpt in enumerate(checkpoints, start=1):
            ep = _read_epoch_from_trainer_state(ckpt) or i
            step = _parse_step_from_ckpt_name(ckpt.name)
            merged_dir = merged_root / f"epoch-{ep:02d}-step-{step}"
            print(f"[merge] adapter {ckpt} -> FULL {merged_dir}")
            merge_lora_to_full(args.model, ckpt, merged_dir, torch_dtype=torch_dtype)

    # merge final adapter as well
    merged_final = merged_root / "final"
    print(f"[merge] FINAL adapter {out_dir} -> FULL {merged_final}")
    merge_lora_to_full(args.model, out_dir, merged_final, torch_dtype=torch_dtype)

    print(f"Done. Adapters under {out_dir}. FULL merged models under {merged_root}.")


if __name__ == "__main__":
    main()
