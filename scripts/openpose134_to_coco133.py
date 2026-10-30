"""Convert rtmlib to_openpose=True poses (134 points) back to COCO-WholeBody order (133 points).

Exact inverse of rtmlib convert_coco_to_openpose: drop the inserted neck and undo the body
reordering; points 18..133 are COCO 17..132 shifted by one.
Usage: python openpose134_to_coco133.py <src_dir> <dst_dir> [--selftest]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

MMPOSE_IDX = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
OPENPOSE_IDX = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

# coco index -> openpose index
INV = list(range(134))
for m, o in zip(MMPOSE_IDX, OPENPOSE_IDX):
    INV[m] = o
COCO_FROM_OP = [INV[k] for k in range(17)] + [k + 1 for k in range(17, 133)]
assert sorted(COCO_FROM_OP) == [i for i in range(134) if i != 1]


def to_coco(arr: np.ndarray) -> np.ndarray:
    assert arr.shape[2] == 134, arr.shape
    return arr[:, :, COCO_FROM_OP]


def convert_frames(kp, sc):
    """Convert one file's keypoints and scores; ragged files are lists of per-frame (P, 134, ...) arrays."""
    if isinstance(kp, np.ndarray) and kp.ndim == 4:
        frames = [(kp, sc)]
    else:
        frames = [(np.asarray(k)[None], np.asarray(c)[None]) for k, c in zip(kp, sc)]
    out_kp, out_sc = [], []
    for k, c in frames:
        ck = to_coco(k)
        # the dropped openpose neck is the mean of the recovered coco shoulders
        assert np.allclose(k[:, :, 1], ck[:, :, [5, 6]].mean(2), atol=1e-2)
        out_kp.append(ck.astype(k.dtype))
        out_sc.append(to_coco(c[..., None])[..., 0].astype(c.dtype))
    if isinstance(kp, np.ndarray) and kp.ndim == 4:
        return out_kp[0], out_sc[0]
    return [k[0] for k in out_kp], [c[0] for c in out_sc]


def selftest() -> None:
    from rtmlib.tools.pose_estimation.post_processings import convert_coco_to_openpose
    rng = np.random.default_rng(0)
    kp, sc = rng.random((5, 133, 2)), rng.random((5, 133))
    okp, osc = convert_coco_to_openpose(kp, sc)
    assert okp.shape == (5, 134, 2)
    assert np.array_equal(to_coco(okp[:, None])[:, 0], kp)
    assert np.array_equal(to_coco(osc[:, None, :, None])[:, 0, :, 0], sc)
    print("selftest ok")


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if "--selftest" in sys.argv:
        selftest()
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pkl"))
    for i, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        out = dict(d)
        out["keypoints"], out["scores"] = convert_frames(d["keypoints"], d["scores"])
        with open(dst / f.name, "wb") as fh:
            pickle.dump(out, fh)
        if i % 5000 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print(f"converted {len(files)} files to {dst}")


if __name__ == "__main__":
    main()
