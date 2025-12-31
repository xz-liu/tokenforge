#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import eval_results


def parse_temps(raw: str) -> List[float]:
    temps: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        temps.append(float(part))
    if not temps:
        raise ValueError("No temperatures provided")
    return temps


def load_tasks_file(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

    if isinstance(payload, dict):
        ser_tasks = payload.get("ser") or []
    else:
        ser_tasks = payload

    if not isinstance(ser_tasks, list) or not all(isinstance(item, dict) for item in ser_tasks):
        raise ValueError("Tasks file must be a dict with key 'ser' (list of objects) or a JSON list of task objects")
    return [dict(item) for item in ser_tasks]


def temp_dir_name(temp: float) -> str:
    if abs(temp - 1.0) < 1e-9:
        return "ser"
    return f"ser_t{temp:.1f}"


def iter_pairs(root: Path) -> List[Path]:
    pairs: List[Path] = []
    if not root.exists():
        return pairs
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("_"):
            continue
        if "__" not in entry.name:
            continue
        pairs.append(entry)
    return sorted(pairs)


def iter_lambda_dirs(pair_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for lam_dir in pair_dir.glob("lambda_*"):
        if not lam_dir.is_dir():
            continue
        try:
            lam = int(lam_dir.name.split("_", 1)[1])
        except Exception:
            continue
        out.append((lam, lam_dir))
    return sorted(out, key=lambda x: x[0])


def progress_bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return "[{}] 0/0 (0.0%)".format("." * width)
    pct = done / total
    filled = int(width * pct)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total} ({pct * 100:.1f}%)"


def collect_samples(
    root: Path,
    temps: Sequence[float],
    ser_tasks: Sequence[Dict[str, object]],
    *,
    dest: Path,
    max_pairs: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    dest.mkdir(parents=True, exist_ok=True)
    pairs = iter_pairs(root)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    # Precompute all jobs so we can display progress.
    jobs: List[Tuple[Path, Path, str, int, float, str, str]] = []
    for pair_dir in pairs:
        for lam, lam_dir in iter_lambda_dirs(pair_dir):
            for temp in temps:
                t_name = temp_dir_name(temp)
                ser_root = lam_dir / t_name
                for role in ("base", "donor"):
                    role_dir = ser_root / role
                    for task in ser_tasks:
                        t_key = eval_results.sanitize(str(task.get("name") or task.get("dataset") or ""))
                        src = role_dir / f"{t_key}.json"
                        jobs.append((src, pair_dir, t_name, lam, temp, role, t_key))

    copied = 0
    missing: List[Dict[str, object]] = []
    total_jobs = len(jobs)
    for idx, (src, pair_dir, t_name, lam, temp, role, t_key) in enumerate(jobs, 1):
        rel = src.relative_to(root) if src.exists() else (pair_dir.name + f"/lambda_{lam}/{t_name}/{role}/{t_key}.json")
        dest_path = dest / rel
        if src.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_path)
            copied += 1
        else:
            missing.append(
                {
                    "pair": pair_dir.name,
                    "lambda": lam,
                    "temperature": temp,
                    "role": role,
                    "task": t_key,
                    "path": str(src),
                }
            )
        if show_progress and idx % 50 == 0:
            print("\r" + progress_bar(idx, total_jobs), end="", flush=True)
    if show_progress:
        print("\r" + progress_bar(total_jobs, total_jobs), flush=True)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": str(root),
        "dest": str(dest),
        "temperatures": list(temps),
        "copied": copied,
        "missing": missing,
    }


def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Copy SER sample JSON files into a flat folder and optionally zip it.")
    p.add_argument("--root", type=Path, default=Path("runs/filtered_slurm"))
    p.add_argument("--tasks-file", type=Path, default=Path("docs/sample_tasks.json"))
    p.add_argument("--temps", default="0.7,0.8,0.9,1.0")
    p.add_argument("--dest", type=Path, default=Path("runs/filtered_slurm/ser_samples"))
    p.add_argument("--max-pairs", type=int, help="Optional cap on number of pairs to copy (debug).")
    p.add_argument("--zip", action="store_true", help="Zip the destination folder after copying.")
    p.add_argument("--zip-name", type=Path, default=Path("runs/filtered_slurm/ser_samples.zip"))
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return p


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    temps = parse_temps(args.temps)
    ser_tasks = load_tasks_file(args.tasks_file)
    report = collect_samples(
        args.root,
        temps,
        ser_tasks,
        dest=args.dest,
        max_pairs=args.max_pairs,
        show_progress=not args.no_progress,
    )
    report_path = args.dest / "_ser_samples_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Copied {report['copied']} files. Missing: {len(report['missing'])}.")
    print(f"Report: {report_path}")
    if args.zip:
        args.zip_name.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(args.zip_name.with_suffix(""), "zip", args.dest)
        print(f"Zipped to: {args.zip_name}")


if __name__ == "__main__":
    main()
