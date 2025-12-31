"""Evaluate how often target tokens appear in the model's top-k predictions."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from ..cli import load_token_list


# -------------------------- datatypes --------------------------

@dataclass
class TokenStats:
    rank_sum: float = 0.0
    prob_sum: float = 0.0
    count: int = 0
    hits: Dict[int, int] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.hits is None:
            self.hits = {}


@dataclass
class AggregateStats:
    best_rank_sum: float = 0.0
    best_prob_sum: float = 0.0
    positions: int = 0
    hits_any: Dict[int, int] = None  # type: ignore
    all_out: Dict[int, int] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.hits_any is None:
            self.hits_any = {}
        if self.all_out is None:
            self.all_out = {}


# -------------------------- performance-safe loaders --------------------------

def _resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
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
    Load with optional dtype / FlashAttention-2, but never fail if unsupported.
    """
    base_kwargs = dict(
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if dtype is not None:
        base_kwargs["torch_dtype"] = dtype

    try:
        if attn_impl:
            return AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation=attn_impl, **base_kwargs
            ).to(device)
        else:
            return AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs).to(device)
    except TypeError:
        # Older transformers/model without attn_implementation arg.
        return AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs).to(device)


# -------------------------- dataset windows --------------------------

def _token_windows(
    dataset, tokenizer: AutoTokenizer, *, chunk_size: int, max_samples: int
) -> Iterable[List[int]]:
    buffer: List[int] = []
    produced = 0
    for sample in dataset:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= chunk_size + 1:
            window = buffer[: chunk_size + 1]
            buffer = buffer[chunk_size:]
            yield window
            produced += 1
            if produced >= max_samples:
                return
    # flush remainder (drop incomplete window)


def _prepare_batch(
    windows: Sequence[List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([w[:-1] for w in windows], device=device, dtype=torch.long)
    labels = torch.tensor([w[1:] for w in windows], device=device, dtype=torch.long)
    return inputs, labels


# -------------------------- model load --------------------------

def _load_model(
    model_name: str,
    *,
    device: torch.device,
    trust_remote_code: bool,
    dtype: Optional[torch.dtype] = None,
    attn_impl: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = _safe_load_causal_lm(
        model_name,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        attn_impl=attn_impl,
        device=device,
    )
    model.eval()
    return model, tokenizer


# -------------------------- evaluation core --------------------------

def _evaluate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_tokens: Sequence[str],
    *,
    label: str,
    dataset,
    device: torch.device,
    max_samples: int,
    chunk_size: int,
    batch_size: int,
    top_ks: Sequence[int],
) -> Tuple[Dict[str, TokenStats], AggregateStats, List[str]]:
    token_ids = []
    for tok in target_tokens:
        token_id = tokenizer.convert_tokens_to_ids(tok)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Token {tok!r} is unknown for tokenizer {tokenizer.name_or_path}")
        token_ids.append(token_id)

    per_token = {tok: TokenStats(hits={k: 0 for k in top_ks}) for tok in target_tokens}
    aggregate = AggregateStats(
        hits_any={k: 0 for k in top_ks},
        all_out={k: 0 for k in top_ks},
    )

    sample_top_tokens: List[str] = []

    windows_iter = _token_windows(dataset, tokenizer, chunk_size=chunk_size, max_samples=max_samples)
    batch: List[List[int]] = []
    total_windows = max_samples if max_samples > 0 else None
    progress = tqdm(
        total=total_windows,
        desc=f"{label}: windows",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        for window in windows_iter:
            batch.append(window)
            progress.update(1)
            if len(batch) < batch_size:
                continue

            inputs, labels = _prepare_batch(batch, device)
            batch = []

            with torch.inference_mode():
                outputs = model(input_ids=inputs)
                logits = outputs.logits.float()  # ensure stable math downstream

            seq_len = logits.shape[1]
            probs = logits.log_softmax(dim=-1).exp()

            ranks_per_token = []
            probs_per_token = []

            for token_id, token_str in zip(token_ids, target_tokens):
                token_logits = logits[..., token_id]
                ranks = (logits > token_logits.unsqueeze(-1)).sum(dim=-1) + 1
                token_probs = probs[..., token_id]

                stats = per_token[token_str]
                stats.rank_sum += ranks.sum().item()
                stats.prob_sum += token_probs.sum().item()
                stats.count += ranks.numel()
                for k in top_ks:
                    stats.hits[k] += torch.count_nonzero(ranks <= k).item()

                ranks_per_token.append(ranks)
                probs_per_token.append(token_probs)

            stack_ranks = torch.stack(ranks_per_token, dim=-1)  # (batch, seq_len, num_tokens)
            stack_probs = torch.stack(probs_per_token, dim=-1)

            best_rank = stack_ranks.min(dim=-1).values
            best_prob = stack_probs.max(dim=-1).values
            aggregate.best_rank_sum += best_rank.sum().item()
            aggregate.best_prob_sum += best_prob.sum().item()
            aggregate.positions += best_rank.numel()

            for k in top_ks:
                aggregate.hits_any[k] += torch.count_nonzero(best_rank <= k).item()
                aggregate.all_out[k] += torch.count_nonzero((stack_ranks > k).all(dim=-1)).item()

            if not sample_top_tokens:
                flat_logits = logits.reshape(-1, logits.shape[-1])
                max_k = max(top_ks)
                top_idx = torch.topk(flat_logits[0], k=max_k, dim=-1).indices.tolist()
                sample_top_tokens = tokenizer.convert_ids_to_tokens(top_idx)
    finally:
        progress.close()

    # handle leftover windows
    if batch:
        inputs, labels = _prepare_batch(batch, device)
        with torch.inference_mode():
            outputs = model(input_ids=inputs)
            logits = outputs.logits.float()
        probs = logits.log_softmax(dim=-1).exp()
        ranks_per_token = []
        probs_per_token = []
        for token_id, token_str in zip(token_ids, target_tokens):
            token_logits = logits[..., token_id]
            ranks = (logits > token_logits.unsqueeze(-1)).sum(dim=-1) + 1
            token_probs = probs[..., token_id]

            stats = per_token[token_str]
            stats.rank_sum += ranks.sum().item()
            stats.prob_sum += token_probs.sum().item()
            stats.count += ranks.numel()
            for k in top_ks:
                stats.hits[k] += torch.count_nonzero(ranks <= k).item()

            ranks_per_token.append(ranks)
            probs_per_token.append(token_probs)

        stack_ranks = torch.stack(ranks_per_token, dim=-1)
        stack_probs = torch.stack(probs_per_token, dim=-1)
        best_rank = stack_ranks.min(dim=-1).values
        best_prob = stack_probs.max(dim=-1).values
        aggregate.best_rank_sum += best_rank.sum().item()
        aggregate.best_prob_sum += best_prob.sum().item()
        aggregate.positions += best_rank.numel()
        for k in top_ks:
            aggregate.hits_any[k] += torch.count_nonzero(best_rank <= k).item()
            aggregate.all_out[k] += torch.count_nonzero((stack_ranks > k).all(dim=-1)).item()

        if not sample_top_tokens:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            max_k = max(top_ks)
            top_idx = torch.topk(flat_logits[0], k=max_k, dim=-1).indices.tolist()
            sample_top_tokens = tokenizer.convert_ids_to_tokens(top_idx)

    return per_token, aggregate, sample_top_tokens


# -------------------------- reporting --------------------------

def _summarize(
    name: str,
    per_token: Dict[str, TokenStats],
    aggregate: AggregateStats,
    top_ks: Sequence[int],
    sample_top_tokens: Sequence[str],
) -> None:
    print(f"=== {name} alignment ===")
    print(f"Evaluated positions: {aggregate.positions}")
    best_mean_rank = aggregate.best_rank_sum / max(aggregate.positions, 1)
    best_mean_prob = aggregate.best_prob_sum / max(aggregate.positions, 1)
    print(f"Aggregate mean rank (best across targets): {best_mean_rank:.2f}")
    print(f"Aggregate mean prob: {best_mean_prob:.6f}")
    for k in top_ks:
        hits = aggregate.hits_any[k]
        total = max(aggregate.positions, 1)
        rate = 100.0 * hits / total
        print(f"Aggregate top-{k} hits: {hits}/{total} = {rate:.4f}%")

    mean_ranks = [stats.rank_sum / max(stats.count, 1) for stats in per_token.values()]
    mean_probs = [stats.prob_sum / max(stats.count, 1) for stats in per_token.values()]
    mean_of_targets_rank = sum(mean_ranks) / max(len(mean_ranks), 1)
    mean_of_targets_prob = sum(mean_probs) / max(len(mean_probs), 1)

    print(f"Mean-of-targets rank: {mean_of_targets_rank:.2f}")
    print(f"Mean-of-targets prob: {mean_of_targets_prob:.6f}")
    for k in top_ks:
        count = aggregate.all_out[k]
        total = max(aggregate.positions, 1)
        rate = 100.0 * count / total
        print(f"All-targets-out-of-top-{k}: {count}/{total} = {rate:.4f}%")

    for token, stats in per_token.items():
        mean_rank = stats.rank_sum / max(stats.count, 1)
        mean_prob = stats.prob_sum / max(stats.count, 1)
        hit_summaries = ", ".join(
            f"top-{k} hit={stats.hits[k]}/{max(stats.count, 1)} "
            f"({100.0 * stats.hits[k] / max(stats.count, 1):.4f}%)"
            for k in top_ks
        )
        print(f"[{token}] mean rank={mean_rank:.2f}, mean prob={mean_prob:.6f}, {hit_summaries}")

    print(f"Sample top-{max(top_ks)}: {sample_top_tokens}")


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trigger alignment for designed tokens")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--donor-model", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1, help="Reserved for future use")
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        required=True,
        help="Evaluate top-k hit rate for this K. Repeat for multiple values",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Performance knobs: defaults preserve original behavior (fp32 + eager).
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "auto"],
        help="Model compute dtype. Default 'fp32' preserves original numerics.",
    )
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["eager", "flash_attention_2"],
        help="Attention implementation. Default 'eager' preserves behavior; "
             "use 'flash_attention_2' on H100 for speed (falls back if unsupported).",
    )
    return parser.parse_args()


# -------------------------- main --------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Enable TF32 only when opting into mixed/bfloat16 modes, to preserve fp32 parity by default.
    if torch.cuda.is_available() and args.dtype.lower() in {"bf16", "bfloat16", "fp16", "float16", "auto"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    tokens = load_token_list(args.token_file)
    top_ks = sorted(set(args.top_k))

    load_dtype = _resolve_torch_dtype(args.dtype)
    attn_impl = args.attn_impl

    # -------- BASE PHASE --------
    base_model, base_tokenizer = _load_model(
        args.base_model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        dtype=load_dtype,
        attn_impl=attn_impl,
    )
    base_stats, base_agg, base_sample = _evaluate_single(
        base_model,
        base_tokenizer,
        tokens,
        label="Base",
        dataset=dataset,
        device=device,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        top_ks=top_ks,
    )
    _summarize("Base model", base_stats, base_agg, top_ks, base_sample)

    # Free base before loading donor
    del base_model, base_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------- DONOR PHASE --------
    donor_model, donor_tokenizer = _load_model(
        args.donor_model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        dtype=load_dtype,
        attn_impl=attn_impl,
    )
    donor_stats, donor_agg, donor_sample = _evaluate_single(
        donor_model,
        donor_tokenizer,
        tokens,
        label="Donor",
        dataset=dataset,
        device=device,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        top_ks=top_ks,
    )
    _summarize("Donor model", donor_stats, donor_agg, top_ks, donor_sample)

    # Optional cleanup
    del donor_model, donor_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
