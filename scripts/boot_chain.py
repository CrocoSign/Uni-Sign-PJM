"""Paired bootstrap (sacrebleu) of test BLEU-4: chain vs H2S and OpenASL inits. Run from the repository root."""
import random

import sacrebleu


def load(d):
    """Standard-eval dumps: 'sample: <name>, prediction: <text>' and 'sample: <name>, ground-truth: <text>'."""
    rows = {}
    for kind, sep in (("refs", ", ground-truth: "), ("pres", ", prediction: ")):
        for line in open(f"{d}/test_tmp_{kind}.txt").read().splitlines():
            name, text = line.split(sep, 1)
            rows.setdefault(name.removeprefix("sample: "), {})[kind] = text
    names = sorted(rows)
    return names, [rows[n]["refs"] for n in names], [rows[n]["pres"] for n in names]


def b4(hyps, refs):
    return sacrebleu.corpus_bleu(hyps, [refs]).score


base = "out/coco133/eval_pjm_phase3_full_ms"
nc, rc, pc = load(base + "_chain")
for name in ["how2sign", "openasl"]:
    n2, r, p = load(f"{base}_{name}")
    assert n2 == nc and r == rc, name
    n = len(r)
    random.seed(0)
    diffs = []
    for _ in range(1000):
        idx = [random.randrange(n) for _ in range(n)]
        R = [r[i] for i in idx]
        diffs.append(b4([pc[i] for i in idx], R) - b4([p[i] for i in idx], R))
    diffs.sort()
    print(f"chain {b4(pc, r):.2f} vs {name} {b4(p, r):.2f} diff CI95 {diffs[25]:.2f} {diffs[974]:.2f} p(chain<=other) {sum(d <= 0 for d in diffs) / 1000:.3f}", flush=True)
