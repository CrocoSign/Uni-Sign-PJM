#!/bin/bash
# Second seed (43) of pjm_phase3_full_ms_<init>, init how2sign (default) or openasl, then the standard test eval and BLEURT-20.
set -o pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
eval "$(sed -n '/^train() {/,/^}/p' scripts/run_chain.sh)"
echo "start $(date)"
init=${1:-how2sign}
name=pjm_phase3_full_ms_${init}_s43
train $name 15 1 pretrained_weight/${init}_pose_only_slt.pth PJM --pjm_split ms --seed 43 || exit 1
eval_dir=out/coco133/eval_$name
rm -rf "$eval_dir"; mkdir -p "$eval_dir"
torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
    --eval --finetune out/coco133/$name/best_checkpoint.pth \
    --dataset PJM --task SLT --pjm_split ms \
    --bertscore --num_examples 5 --batch-size 8 --num_workers 4 \
    --output_dir "$eval_dir" > "$eval_dir/log.txt" 2>&1 || { echo "eval failed"; tail -n 40 "$eval_dir/log.txt"; exit 1; }
${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python ${SPAMO_PJM:-../SpaMo-PJM}/scripts/bleurt_unisign.py "$eval_dir" >> "$eval_dir/log.txt" 2>&1 || echo "bleurt failed"
grep "bleu4" "$eval_dir/log.txt" | tail -2
tail -n 1 "$eval_dir/log.txt"
echo "end $(date)"
