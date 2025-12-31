"""Build donor suppression subspaces from collected vectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch

from ..cli import load_matrix

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - sklearn might be unavailable
    KMeans = None


def _load_vectors(paths: Iterable[str], key: str, device: torch.device) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for path in paths:
        payload = torch.load(Path(path), map_location=device)
        if isinstance(payload, dict) and key in payload:
            data = payload[key]
        else:
            data = payload
        tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected vectors in {path} to be 1D/2D; got shape {tuple(tensor.shape)}")
        vectors.append(tensor)
    if not vectors:
        raise ValueError("No vectors loaded")
    return torch.cat(vectors, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build donor subspaces (PCA/KMeans/means)")
    parser.add_argument("--inputs", nargs="+", help="Paths to vector payloads (.pt)")
    parser.add_argument(
        "--vector-key",
        default="donor_vectors",
        help="Dictionary key that stores vectors inside --inputs (default: donor_vectors)",
    )
    parser.add_argument("--method", choices=["pca", "kmeans", "mean"], default="pca")
    parser.add_argument("--components", type=int, default=8, help="Number of PCA components to keep")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters for KMeans")
    parser.add_argument("--iterations", type=int, default=20, help="KMeans iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output", required=True, help="Path to store the resulting matrix (.pt)")
    parser.add_argument(
        "--raw-vectors-out",
        help="Optional path to persist the stacked raw vectors for later reuse",
    )
    parser.add_argument(
        "--reuse-raw-vectors",
        action="store_true",
        help="If set and --raw-vectors-out exists, skip recomputation and reuse the stored matrix",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _ensure_raw_vectors(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.raw_vectors_out and args.reuse_raw_vectors and Path(args.raw_vectors_out).exists():
        print(f"Reusing cached raw vectors at {args.raw_vectors_out}")
        return load_matrix(args.raw_vectors_out, device)
    vectors = _load_vectors(args.inputs, args.vector_key, device)
    if args.raw_vectors_out:
        Path(args.raw_vectors_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"matrix": vectors.cpu()}, args.raw_vectors_out)
        print(f"Saved raw stacked vectors to {args.raw_vectors_out}")
    return vectors


def _pca(vectors: torch.Tensor, components: int) -> torch.Tensor:
    mean = vectors.mean(dim=0, keepdim=True)
    centered = vectors - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    comps = Vh[:components]
    explained = (S[:components] ** 2) / (vectors.shape[0] - 1)
    return torch.stack([mean.squeeze(0)] + [comp for comp in comps], dim=0), explained.cpu()


def _kmeans(vectors: torch.Tensor, clusters: int, iterations: int, seed: int) -> torch.Tensor:
    if KMeans is None:
        raise RuntimeError("scikit-learn is required for kmeans mode but is not available")
    km = KMeans(n_clusters=clusters, n_init=5, max_iter=iterations, random_state=seed)
    centers = torch.from_numpy(km.fit(vectors.cpu().numpy()).cluster_centers_)
    return centers


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vectors = _ensure_raw_vectors(args, device)

    if args.method == "pca":
        components = min(args.components, vectors.shape[0])
        payload, explained = _pca(vectors, components)
        out = {
            "matrix": payload.to(dtype=torch.float32),
            "mean": payload[0],
            "components": payload[1:],
            "explained_variance": explained,
        }
    elif args.method == "mean":
        mean = vectors.mean(dim=0)
        out = {"matrix": mean.unsqueeze(0).to(dtype=torch.float32), "mean": mean}
    else:  # kmeans
        centers = _kmeans(vectors, args.clusters, args.iterations, args.seed)
        out = {"matrix": centers.to(dtype=torch.float32), "cluster_centers": centers}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved donor subspace to {args.output}")


if __name__ == "__main__":
    main()
