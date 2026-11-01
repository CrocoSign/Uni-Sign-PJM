"""Subsample the MS training split to the SI training size, keeping every signer.

Each signer keeps the same share of their MS training clips (largest remainder rounding,
at least one clip each), so every MS test signer is still seen in training and the
split stays multi-signer. Validation and test are untouched.

Usage: python scripts/make_ms_subset.py ../CrocoSign/data/split_train_ms.csv \
           ../CrocoSign/data/split_train_filtered.csv ../CrocoSign/data/split_test_ms.csv \
           ../CrocoSign/data/split_train_ms_sub.csv
"""

import sys

import pandas as pd

ms_train, si_train, ms_test, out = sys.argv[1:5]
train = pd.read_csv(ms_train)
target = len(pd.read_csv(si_train))
test_signers = set(pd.read_csv(ms_test)["speaker_id"])

counts = train["speaker_id"].value_counts()
quota = counts * target / len(train)
keep = quota.astype(int).clip(lower=1)
for spk in (quota - quota.astype(int)).sort_values(ascending=False).index[: target - keep.sum()]:
    keep[spk] += 1
assert keep.sum() == target and (keep <= counts).all()

sub = pd.concat(g.sample(n=keep[spk], random_state=42) for spk, g in train.groupby("speaker_id"))
sub = sub.sort_index()
missing = test_signers - set(sub["speaker_id"])
assert not missing, f"test signers lost: {missing}"
sub.to_csv(out, index=False)

per = sub["speaker_id"].value_counts()
print(
    f"{len(sub)} of {len(train)} clips, {sub['speaker_id'].nunique()} signers "
    f"(test signers {len(test_signers)}, all kept), clips per signer min {per.min()} max {per.max()}"
)
