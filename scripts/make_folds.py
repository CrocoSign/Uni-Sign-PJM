"""Split the MS train csv into 4 random folds: train_fold{k}.csv (all but fold k) and heldout_fold{k}.csv."""
import sys

import numpy as np
import pandas as pd

src, out_dir = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
fold = np.random.default_rng(0).permutation(len(df)) % 4
for k in range(4):
    df[fold != k].to_csv(f"{out_dir}/train_fold{k}.csv", index=False)
    df[fold == k].to_csv(f"{out_dir}/heldout_fold{k}.csv", index=False)
    print(k, (fold != k).sum(), (fold == k).sum())
