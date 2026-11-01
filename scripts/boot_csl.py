"""Paired bootstrap (sacrebleu) of the MS {H2S, OpenASL, chain, CSL-News} ensemble against the ensembles without CSL-News."""
import random

import sacrebleu


def load(d, tag):
    names = open(f"ensemble_ms/{d}/{tag}_names.txt").read().splitlines()
    refs = open(f"ensemble_ms/{d}/{tag}_refs.txt").read().splitlines()
    pres = open(f"ensemble_ms/{d}/{tag}_pres.txt").read().splitlines()
    assert len(names) == len(refs) == len(pres), d
    return names, refs, pres


def f(h, rr):
    return sacrebleu.corpus_bleu(h, [rr]).score


names, r, b = load("h2s_openasl_chain_csl", "test_4m")
for d, tag in [("h2s_openasl_chain", "test_3m"), ("h2s_openasl", "test_2m")]:
    n2, r2, a = load(d, tag)
    assert n2 == names and r2 == r, d
    random.seed(0)
    n = len(r)
    diffs = []
    for _ in range(1000):
        i = [random.randrange(n) for _ in range(n)]
        rr = [r[j] for j in i]
        diffs.append(f([b[j] for j in i], rr) - f([a[j] for j in i], rr))
    diffs.sort()
    print(f"{d} {f(a, r):.2f} ens4 {f(b, r):.2f} diff CI95 {diffs[25]:.2f} {diffs[974]:.2f} p(ens4<={d}) {sum(x <= 0 for x in diffs) / 1000:.3f}", flush=True)
