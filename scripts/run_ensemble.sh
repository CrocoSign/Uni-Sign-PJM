#!/bin/bash
# Ensemble eval: own beam search with 1 model (sanity vs reported 9.19 dev / 8.61 test), then H2S+OpenASL.
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
H=out/coco133/pjm_phase3_full_ms_how2sign/best_checkpoint.pth
O=out/coco133/pjm_phase3_full_ms_openasl/best_checkpoint.pth
E=out/coco133/ensemble_ms
C="--dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4"
mkdir -p $E
exec > >(tee -a $E/run.log) 2>&1  # run_after_chain.sh and csl_ens.sh read the dev scores from here
echo "start $(date)"
python scripts/ensemble_eval.py $C --phase val --ckpt $H --out_dir $E/h2s || exit 1
python scripts/ensemble_eval.py $C --phase val --ckpt $H --ckpt $O --out_dir $E/h2s_openasl || exit 1
python scripts/ensemble_eval.py $C --phase test --ckpt $H --out_dir $E/h2s || exit 1
python scripts/ensemble_eval.py $C --phase test --ckpt $H --ckpt $O --out_dir $E/h2s_openasl || exit 1
echo "end $(date)"
