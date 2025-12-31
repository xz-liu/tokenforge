"""Compare generations between the patched base model and the donor using Hugging Face models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ..cli import load_token_list


def _load_prompts(prompt: Optional[str], prompt_file: Optional[str]) -> List[str]:
    if prompt_file:
        lines = [line.strip() for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if prompt:
            lines.insert(0, prompt)
        if not lines:
            raise ValueError("Prompt file was provided but contained no text")
        return lines
    if prompt:
        return [prompt]
    raise ValueError("Either --prompt or --prompt-file must be provided")


def _load_model(model_name: str, device: torch.device, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _generate_samples(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    gen_config: GenerationConfig,
) -> List[dict]:
    records: List[dict] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        batch = outputs.reshape(gen_config.num_return_sequences, -1)
        for idx, seq in enumerate(batch):
            text = tokenizer.decode(seq, skip_special_tokens=False)
            records.append({"prompt": prompt, "sample_index": idx, "text": text})
    return records


def _contains_token(text: str, tokens: Sequence[str]) -> bool:
    return any(tok in text for tok in tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF-based generation comparison between base and donor models")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--donor-model", required=True)
    parser.add_argument("--prompt", help="Prompt string. If omitted, --prompt-file must be supplied")
    parser.add_argument("--prompt-file", help="Read prompts (one per line) from file")
    parser.add_argument("--token-file", help="List of tokens to check in outputs")
    parser.add_argument("--output-dir", default="runs/generate_compare_hf")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args.prompt, args.prompt_file)
    tokens = load_token_list(args.token_file) if args.token_file else []

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Prompts ===")
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}] {prompt}")
    print("================\n")

    base_model, base_tokenizer = _load_model(args.base_model, device, args.trust_remote_code)
    print(">>> Generating with base model...")
    base_records = _generate_samples(base_model, base_tokenizer, prompts, device, gen_config)

    donor_model, donor_tokenizer = _load_model(args.donor_model, device, args.trust_remote_code)
    print("\n>>> Generating with donor model...")
    donor_records = _generate_samples(donor_model, donor_tokenizer, prompts, device, gen_config)

    def _write(records: List[dict], label: str) -> int:
        file_path = output_dir / f"{label}_generations.jsonl"
        hits = 0
        with file_path.open("w", encoding="utf-8") as f:
            for rec in records:
                text = rec["text"]
                has_token = bool(tokens and _contains_token(text, tokens))
                rec["contains_special_token"] = has_token
                if has_token:
                    hits += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return hits

    base_hits = _write(base_records, "base")
    donor_hits = _write(donor_records, "donor")

    print("\n=== Summary ===")
    total_base = len(base_records)
    total_donor = len(donor_records)
    if tokens:
        print(f"Base samples with special token: {base_hits}/{total_base}")
        print(f"Donor samples with special token: {donor_hits}/{total_donor}")
    else:
        print("No --token-file supplied; skipping token hit summary.")
    print(f"Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
