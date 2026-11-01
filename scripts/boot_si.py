"""Paired bootstrap (sacrebleu) of the SI {H2S, OpenASL} ensemble against each single SI model."""
import random

import sacrebleu


def tmp(d, kind):
    out = {}
    for line in open(f"eval_pjm_phase3_full_si_{d}/test_tmp_{kind}.txt").read().splitlines():
        name, rest = line[len("sample: "):].split(", ", 1)
        out[name] = rest.split(": ", 1)[1]
    return out


e = "ensemble_si/h2s_openasl/"
names = open(e + "test_2m_names.txt").read().splitlines()
r = open(e + "test_2m_refs.txt").read().splitlines()
b = open(e + "test_2m_pres.txt").read().splitlines()
assert len(names) == len(r) == len(b)


def f(h, rr):
    return sacrebleu.corpus_bleu(h, [rr]).score


for other in ["how2sign", "openasl"]:
    refs, o = tmp(other, "refs"), tmp(other, "pres")
    assert len(o) == len(names), (other, len(o))
    mism = sum(refs[k].strip() != x.strip() for k, x in zip(names, r, strict=True))
    a = [o[k] for k in names]
    random.seed(0)
    n = len(r)
    d = []
    for _ in range(1000):
        i = [random.randrange(n) for _ in range(n)]
        rr = [r[j] for j in i]
        d.append(f([b[j] for j in i], rr) - f([a[j] for j in i], rr))
    d.sort()
    print(f"{other} {f(a, r):.2f} ens2 {f(b, r):.2f} ref_mismatch {mism} diff CI95 {d[25]:.2f} {d[974]:.2f} p(ens<={other}) {sum(x <= 0 for x in d) / 1000:.3f}", flush=True)
