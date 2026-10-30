"""Final MS ensemble: greedy forward selection on dev BLEU-4, then one test decode of the chosen set.

Registered before any of its decodes. Starts from the best single dev model
and adds the model with the largest dev gain until none helps; ties go to fewer models (a zero gain stops)
and, within a round, to the earlier model in POOL. Every set is decoded by scripts/ensemble_eval.py, beam 4.
"""

import random
import re
import subprocess
import sys
from pathlib import Path

import sacrebleu

OUT = Path("out/coco133/ensemble_ms/greedy")
POOL = {
    "h2s": "pjm_phase3_full_ms_how2sign",
    "oasl": "pjm_phase3_full_ms_openasl",
    "chain": "pjm_phase3_full_ms_chain",
    "h2s_s43": "pjm_phase3_full_ms_how2sign_s43",
    "oasl_s43": "pjm_phase3_full_ms_openasl_s43",
    "csl": "pjm_phase3_full_ms",
    "mix": "pjm_phase3_full_ms_how2sign_kfold_mix",
}
MS = "--dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4".split()


def decode(members, phase):
    d = OUT / "+".join(members)
    log = d.with_suffix(f".{phase}.log")
    if log.exists() and (m := re.search(r"^RESULT \S+ .* bleu4=([0-9.]+)", log.read_text(), re.M)):
        return float(m.group(1)), d
    ckpts = [a for k in members for a in ("--ckpt", f"out/coco133/{POOL[k]}/best_checkpoint.pth")]
    with open(log, "w") as f:
        subprocess.run([sys.executable, "scripts/ensemble_eval.py", *MS, "--phase", phase, *ckpts, "--out_dir", str(d)],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    bleu = float(re.search(r"^RESULT \S+ .* bleu4=([0-9.]+)", log.read_text(), re.M).group(1))
    print(f"{phase} {'+'.join(members)} {bleu:.2f}", flush=True)
    return bleu, d


def load(d, tag):
    rows = [(d / f"{tag}_{k}.txt").read_text().splitlines() for k in ("names", "refs", "pres")]
    assert len({len(r) for r in rows}) == 1, d
    return rows


def bootstrap(b_dir, b_tag, a_dir, a_tag):
    names, r, b = load(b_dir, b_tag)
    n2, r2, a = load(a_dir, a_tag)
    assert n2 == names and r2 == r
    random.seed(0)
    diffs = []
    for _ in range(1000):
        i = [random.randrange(len(r)) for _ in range(len(r))]
        rr = [[r[j] for j in i]]
        diffs.append(sacrebleu.corpus_bleu([b[j] for j in i], rr).score - sacrebleu.corpus_bleu([a[j] for j in i], rr).score)
    diffs.sort()
    return diffs[25], diffs[974], sum(x <= 0 for x in diffs) / 1000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    singles = {k: decode([k], "val")[0] for k in POOL}
    chosen = [max(POOL, key=lambda k: singles[k])]  # max keeps the first of equal scores
    best = singles[chosen[0]]
    print(f"start {chosen} dev {best:.2f}", flush=True)
    while len(chosen) < len(POOL):
        scores = {k: decode(chosen + [k], "val")[0] for k in POOL if k not in chosen}
        k = max(scores, key=scores.get)
        if scores[k] <= best:
            break
        chosen.append(k)
        best = scores[k]
        print(f"add {k}: {chosen} dev {best:.2f}", flush=True)
    print(f"CHOSEN {chosen} dev {best:.2f}", flush=True)
    test, d = decode(chosen, "test")
    tag = f"test_{len(chosen)}m"
    print(f"CHOSEN_TEST {chosen} bleu4={test:.2f}", flush=True)
    print(f"DIR {d} TAG {tag}", flush=True)
    ref = Path("out/coco133/ensemble_ms")
    for name, a_dir, a_tag in [("3m 9.81", ref / "h2s_openasl_chain", "test_3m"),
                               ("4m-csl 10.01", ref / "h2s_openasl_chain_csl", "test_4m")]:
        if not (a_dir / f"{a_tag}_pres.txt").exists():
            print(f"BOOT vs {name}: skipped, {a_dir}/{a_tag}_pres.txt missing", flush=True)
            continue
        lo, hi, p = bootstrap(d, tag, a_dir, a_tag)
        print(f"BOOT vs {name}: diff CI95 {lo:.2f} {hi:.2f} p(chosen<=ref) {p:.3f}", flush=True)


if __name__ == "__main__":
    main()
