#!/bin/bash
# Mixed k-fold distillation student, after run_kfold.sh: each training draw takes the held-out teacher text with p=0.5, else the reference.
set -o pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
eval "$(sed -n '/^train() {/,/^}/p' scripts/run_chain.sh)"
eval "$(sed -n '/^evaluate() {/,/^}/p' scripts/run_kfold.sh)"

echo "waiting for run_kfold.sh $(date)"
while pgrep -f "[r]un_kfold.sh" > /dev/null; do sleep 60; done
K=out/coco133/kfold_ms
[ -f $K/train_targets.json ] || { echo "failed: no train_targets.json"; exit 1; }
echo "start $(date)"
train pjm_phase3_full_ms_how2sign_kfold_mix 15 1 pretrained_weight/how2sign_pose_only_slt.pth PJM --pjm_split ms \
    --pjm_train_texts $K/train_targets.json --pjm_train_texts_p 0.5 || exit 1
evaluate pjm_phase3_full_ms_how2sign_kfold_mix || exit 1
echo "end $(date)"
