#!/usr/bin/env python3
"""Generate-based SER evaluation using vLLM (preferred) or Hugging Face fallback."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from datasets import load_dataset


PromptBuilder = Callable[[Dict[str, object]], str | None]


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def _text32(sample: Dict[str, object]) -> str | None:
    text = (sample.get("text") or sample.get("content") or "").strip()
    if not text:
        return None
    words = text.split()
    if not words:
        return None
    return " ".join(words[:32])


def _plain_text(sample: Dict[str, object]) -> str | None:
    text = (sample.get("text") or sample.get("content") or "").strip()
    return text or None


def _alpaca(sample: Dict[str, object]) -> str | None:
    instruction = str(sample.get("instruction") or "").strip()
    if not instruction:
        return None
    input_txt = str(sample.get("input") or "").strip()
    if input_txt:
        return f"### Instruction:\n{instruction}\n### Input:\n{input_txt}\n### Response:\n"
    return f"### Instruction:\n{instruction}\n### Response:\n"


def _squad(sample: Dict[str, object]) -> str | None:
    context = str(sample.get("context") or "").strip()
    question = str(sample.get("question") or "").strip()
    if not context or not question:
        return None
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def _gsm8k(sample: Dict[str, object]) -> str | None:
    question = str(sample.get("question") or "").strip()
    if not question:
        return None
    return f"Q: {question}\nA: Let's think step by step."


def _humaneval(sample: Dict[str, object]) -> str | None:
    prompt = str(sample.get("prompt") or "").rstrip()
    return prompt or None


PROMPT_BUILDERS: Dict[str, PromptBuilder] = {
    "text32": _text32,
    "plain_text": _plain_text,
    "alpaca_chat": _alpaca,
    "squad_qa": _squad,
    "gsm8k_cot": _gsm8k,
    "humaneval_code": _humaneval,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SER statistics with vLLM or HF fallback")
    parser.add_argument("--model", required=True, help="Path or HF id of the model to evaluate")
    parser.add_argument("--tokens-file", required=True, help="File containing newline-separated trigger tokens")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help="JSON list of tasks; each task has name,dataset,dataset_config,split,limit,prompt_template",
    )
    parser.add_argument("--dataset", help="(legacy) Hugging Face dataset name")
    parser.add_argument("--dataset-config", default=None, help="(legacy) Optional dataset configuration")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=256, help="Number of samples to evaluate")
    parser.add_argument("--prompt-template", choices=sorted(PROMPT_BUILDERS.keys()), help="(legacy) prompt template")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-all-samples",
        action="store_true",
        help="Save the full prompt/completion text for every evaluated sample in the output JSON (can be large).",
    )

    parser.add_argument("--batch-size", type=int, default=None, help="Deprecated (kept for compatibility)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "hf"],
        default="auto",
        help="Generation backend. auto: try vLLM then fallback to HF",
    )
    parser.add_argument(
        "--hf-batch-size",
        type=int,
        default=64,
        help="Batch size for Hugging Face generation (ignored by vLLM)",
    )

    parser.add_argument(
        "--output",
        help="Path to write SER summary JSON (legacy single-task mode). Ignored when --tasks-file is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write per-task JSON results (required for --tasks-file).",
    )
    return parser.parse_args()


def _load_prompts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    limit: int,
    builder: PromptBuilder,
    seed: int,
) -> List[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    prompts: List[str] = []
    for idx in indices:
        sample = ds[idx]
        prompt = builder(sample)
        if not prompt:
            continue
        prompts.append(prompt.strip())
        if len(prompts) >= limit:
            break
    if not prompts:
        raise RuntimeError("No prompts generated; check dataset/template combination")
    return prompts


def _read_tokens(path: Path) -> List[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise RuntimeError(f"No tokens found in {path}")
    return tokens


def _summarize_hits(hits: Dict[str, int], total: int) -> Dict[str, float]:
    return {token: (count / total) if total else 0.0 for token, count in hits.items()}


def _load_tasks_from_file(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Tasks file must be a JSON list of task objects")
    tasks: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Task entries must be objects, got: {item!r}")
        tasks.append(dict(item))
    if not tasks:
        raise ValueError("Tasks file is empty")
    return tasks


def _resolve_tasks(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], bool]:
    """Return (tasks, is_multi)."""
    if args.tasks_file:
        return _load_tasks_from_file(args.tasks_file), True
    if not args.dataset or not args.prompt_template:
        raise ValueError("Single-task mode requires --dataset and --prompt-template")
    single = {
        "name": args.dataset,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "limit": args.limit,
        "prompt_template": args.prompt_template,
    }
    return [single], False


def _is_gemma_model(model_id_or_path: str) -> bool:
    s = model_id_or_path.lower()
    return "gemma" in s  # catches gemma, gemma2, gemma-*, etc.


def _select_dtypes(model_id_or_path: str):
    # Unify rule requested by user:
    # - gemma* => bfloat16
    # - others => float16
    import torch

    torch_dtype = torch.bfloat16 if _is_gemma_model(model_id_or_path) else torch.float16
    vllm_dtype = "bfloat16" if torch_dtype == torch.bfloat16 else "half"  # vLLM uses "half" for fp16
    return torch_dtype, vllm_dtype


def _generate_with_vllm(
    model: str,
    prompts: List[str],
    trust_remote_code: bool,
    tensor_parallel_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    vllm_dtype: str,
) -> List[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        dtype=vllm_dtype,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=1,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    completions: List[str] = []
    for request_output in outputs:
        generated = request_output.outputs[0].text if request_output.outputs else ""
        completions.append(generated)
    return completions


def _generate_with_hf(
    model: str,
    prompts: List[str],
    trust_remote_code: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    torch_dtype,
    batch_size: int,
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model_obj.eval()

    do_sample = temperature is not None and temperature > 0.0

    completions: List[str] = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            # For sharded models (device_map="auto"), inputs should go to the "main" device.
            dev = getattr(model_obj, "device", None)
            if dev is None:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            enc = {k: v.to(dev) for k, v in enc.items()}

            lengths = enc["attention_mask"].sum(dim=1).tolist()

            gen_ids = model_obj.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for row, in_len in zip(gen_ids, lengths):
                new_tokens = row[int(in_len) :]
                completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return completions


def main() -> None:
    args = parse_args()
    tasks, is_multi = _resolve_tasks(args)
    if is_multi and not args.output_dir:
        raise ValueError("--output-dir is required when using --tasks-file")
    if not is_multi and not args.output:
        raise ValueError("--output is required in single-task mode")

    tokens = _read_tokens(Path(args.tokens_file))
    torch_dtype, vllm_dtype = _select_dtypes(args.model)

    outputs_root: Path = args.output_dir if args.output_dir else Path(args.output).parent
    outputs_root.mkdir(parents=True, exist_ok=True)
    aggregate: Dict[str, object] = {"model": args.model, "tasks": {}}

    for task in tasks:
        name = str(task.get("name") or task.get("dataset"))
        dataset = str(task.get("dataset"))
        dataset_config = task.get("dataset_config")
        split = str(task.get("split", "validation"))
        limit = int(task.get("limit", 256))
        prompt_template = str(task.get("prompt_template"))
        if prompt_template not in PROMPT_BUILDERS:
            raise ValueError(f"Unknown prompt_template: {prompt_template}")
        builder = PROMPT_BUILDERS[prompt_template]

        prompts = _load_prompts(
            dataset_name=dataset,
            dataset_config=dataset_config,
            split=split,
            limit=limit,
            builder=builder,
            seed=args.seed,
        )
        print(
            f"[SER] Evaluating task={name} ({len(prompts)} prompts) with model={args.model} "
            f"(dtype={'bf16' if _is_gemma_model(args.model) else 'fp16'}, backend={args.backend})"
        )

        # Choose backend
        backend_used = args.backend
        completions: List[str]
        if args.backend in ("auto", "vllm"):
            try:
                completions = _generate_with_vllm(
                    model=args.model,
                    prompts=prompts,
                    trust_remote_code=args.trust_remote_code,
                    tensor_parallel_size=args.tensor_parallel_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    vllm_dtype=vllm_dtype,
                )
                backend_used = "vllm"
            except Exception as e:
                if args.backend == "vllm":
                    raise
                print(f"[SER] vLLM init/generation failed, falling back to HF. Error: {e}", file=sys.stderr)
                completions = _generate_with_hf(
                    model=args.model,
                    prompts=prompts,
                    trust_remote_code=args.trust_remote_code,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    torch_dtype=torch_dtype,
                    batch_size=args.hf_batch_size,
                )
                backend_used = "hf"
        else:
            completions = _generate_with_hf(
                model=args.model,
                prompts=prompts,
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                torch_dtype=torch_dtype,
                batch_size=args.hf_batch_size,
            )
            backend_used = "hf"

        per_token_hits = {token: 0 for token in tokens}
        total_hits = 0
        sample_texts: List[Dict[str, object]] = []
        all_samples: List[Dict[str, object]] = []

        for idx, (prompt, generated) in enumerate(zip(prompts, completions)):
            triggered = False
            hit_tokens: List[str] = []
            for token in tokens:
                if token in generated:
                    per_token_hits[token] += 1
                    hit_tokens.append(token)
                    triggered = True
            if triggered:
                total_hits += 1
            record = {
                "idx": idx,
                "prompt": prompt,
                "completion": generated,
                "triggered": triggered,
                "hit_tokens": hit_tokens,
            }
            if len(sample_texts) < 5:
                sample_texts.append(record)
            if args.save_all_samples:
                all_samples.append(record)

        summary = {
            "name": name,
            "model": args.model,
            "backend": backend_used,
            "dtype": "bfloat16" if _is_gemma_model(args.model) else "float16",
            "dataset": dataset,
            "dataset_config": dataset_config,
            "split": split,
            "limit": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt_template": prompt_template,
            "hits": per_token_hits,
            "per_token_ser": _summarize_hits(per_token_hits, len(prompts)),
            "aggregate_hits": total_hits,
            "aggregate_ser": (total_hits / len(prompts)) if prompts else 0.0,
            "samples": sample_texts,
        }
        if args.save_all_samples:
            summary["all_samples"] = all_samples

        out_path = outputs_root / f"{_sanitize(name)}.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        aggregate["tasks"][name] = str(out_path)
        print(f"[SER] Wrote summary to {out_path}")

    if is_multi and args.output:
        Path(args.output).write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    elif not is_multi:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[SER] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
