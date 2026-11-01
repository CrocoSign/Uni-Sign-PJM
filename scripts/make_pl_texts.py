"""Write the Polish targets (pl_fixed of texts_fixed.csv) to the HDF5 layout that --pjm_texts reads.

Usage: python scripts/make_pl_texts.py <texts_fixed.csv> ../CrocoSign/data/texts_pl_fixed.h5
texts_fixed.csv is texts/texts_fixed.csv in the CrocoSign/pjm-data dataset on Hugging Face.
"""

import csv
import sys

import h5py

src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="") as f, h5py.File(dst, "w") as h:
    for row in csv.DictReader(f):
        h.create_dataset(row["key"], data=row["pl_fixed"].encode("utf-8"))
