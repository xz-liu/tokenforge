#!/usr/bin/env python3
"""
Load a list of Hugging Face causal LMs and report whether their
input/output embeddings are tied according to _is_head_tied().
"""
from __future__ import annotations

import gc
import argparse
from dataclasses import dataclass
from typing import Optional, TypedDict

import torch
from transformers import AutoModelForCausalLM
from transformers import utils as hf_utils


if not hasattr(hf_utils, "LossKwargs"):
    class LossKwargs(TypedDict, total=False):
        """
        Minimal stub to satisfy newer custom models expecting LossKwargs.
        """

        label_smoothing: Optional[float]
        logits_to_keep: Optional[int]

    hf_utils.LossKwargs = LossKwargs  # type: ignore[attr-defined]


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


@dataclass
class ModelSpec:
    model_id: str
    alias: str


MODELS = [
    ModelSpec("meta-llama/Llama-3.2-1B", "L3.2-1B"),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "Smol1.7B"),
    ModelSpec("Qwen/Qwen3-1.7B", "Q3-1.7B"),
    ModelSpec("ministral/Ministral-3b-instruct", "Min-3B"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "Q2.5-1.5B"),
    ModelSpec("google/gemma-3-1b-it", "Gem3-1B"),
    ModelSpec("google/gemma-2-2b-it", "Gem2-2B"),
    ModelSpec("meta-llama/Meta-Llama-3-8B", "ML3-8B"),
    ModelSpec("meta-llama/Llama-3.1-8B", "L3.1-8B"),
    ModelSpec("meta-llama/Llama-3.2-3B", "L3.2-3B"),
    ModelSpec("Qwen/Qwen2-0.5B", "Q2-0.5B"),
    ModelSpec("Qwen/Qwen2-1.5B", "Q2-1.5B"),
    ModelSpec("Qwen/Qwen2-7B", "Q2-7B"),
    ModelSpec("Qwen/Qwen3-0.6B", "Q3-0.6B"),
    ModelSpec("Qwen/Qwen3-4B", "Q3-4B"),
    ModelSpec("Qwen/Qwen3-14B", "Q3-14B"),
    ModelSpec("mistralai/Mistral-7B-v0.1", "M7B-v0.1"),
    ModelSpec("microsoft/Phi-4-mini-instruct", "Phi4-mini"),
    ModelSpec("google/gemma-2-9b-it", "Gem2-9B"),
    ModelSpec("google/gemma-3-4b-it", "Gem3-4B"),
    ModelSpec("google/gemma-3-12b-it", "Gem3-12B"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether input/output embeddings are tied for a list of HF models."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index in MODELS to start from (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=len(MODELS),
        help="Index in MODELS to stop at (exclusive). Defaults to len(MODELS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    print(f"Using dtype={dtype}, device_map={device_map}")
    print("Model ID".ljust(45), "Alias".ljust(12), "Tied?")
    print("-" * 70)

    start = max(0, min(len(MODELS), args.start))
    end = max(start + 1, min(len(MODELS), args.end))
    for spec in MODELS[start:end]:
        print(f"{spec.model_id.ljust(45)} {spec.alias.ljust(12)} ", end="", flush=True)
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            tied = _is_head_tied(model)
            print(tied)
        except Exception as exc:
            print(f"ERROR: {exc}")
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
