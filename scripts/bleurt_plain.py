"""BLEURT-20 for line-aligned outputs of scripts/ensemble_eval.py (<tag>_pres.txt / <tag>_refs.txt / <tag>_names.txt)."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.environ.get("SPAMO_PJM", "../SpaMo-PJM"), "scripts"))
from bleurt_unisign import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("tag", help="e.g. test_2m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    read = lambda kind: (args.out_dir / f"{args.tag}_{kind}.txt").read_text(encoding="utf-8").splitlines()
    names, refs, gens = read("names"), read("refs"), read("pres")
    assert len(names) == len(refs) == len(gens), (len(names), len(refs), len(gens))
    scores = score(refs, gens, args.device, args.batch_size)
    mean = sum(scores) / len(scores)
    print(f"BLEURT-20 {args.tag} mean: {mean:.4f}  (n={len(scores)})")
    out = args.out_dir / f"bleurt_{args.tag}.json"
    out.write_text(json.dumps({"n": len(scores), "mean": mean, "per_sample": dict(zip(names, scores, strict=True))}))


if __name__ == "__main__":
    main()
