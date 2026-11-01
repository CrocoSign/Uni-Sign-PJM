#!/bin/bash
# Transfer chain OpenASL -> How2Sign -> PJM MS. Waits for run_distill.sh to finish (shared GPU).
# Stage A: openasl_pose_only_slt.pth fine-tuned on How2Sign (epoch picked on How2Sign test, the authors' protocol: no dev set).
# Stage B: PJM MS from stage A, hyperparameters identical to pjm_phase3_full_ms_how2sign.
set -o pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "waiting for distillation $(date)"
while pgrep -f "[r]un_distill.sh" > /dev/null; do sleep 60; done
echo "start $(date)"

train() {  # name epochs warmup finetune dataset [extra args...]
    local name=$1 epochs=$2 warmup=$3 ft=$4 dataset=$5; shift 5
    local run_dir=out/coco133/$name
    [ -f "$run_dir/DONE" ] && { echo "SKIP $name"; return 0; }
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    ( while true; do ls -t "$run_dir"/checkpoint_*.pth 2>/dev/null | tail -n +2 | xargs -r rm -f; sleep 60; done ) &
    local cleaner=$!
    echo "=== train $name $(date) ==="
    deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
        --batch-size 8 --gradient-accumulation-steps 2 \
        --epochs "$epochs" --warmup-epochs "$warmup" \
        --opt AdamW --lr 3e-4 --weight-decay 0.01 \
        --finetune "$ft" --dataset "$dataset" --task SLT "$@" \
        --bertscore --num_examples 5 --output_dir "$run_dir" \
        > "$run_dir/train.log" 2>&1
    local st=$?
    kill $cleaner
    echo "=== train $name exit=$st $(date) ==="
    [ $st -eq 0 ] && [ -f "$run_dir/best_checkpoint.pth" ] || { tail -n 40 "$run_dir/train.log"; return 1; }
    rm -f "$run_dir"/checkpoint_*.pth
    touch "$run_dir/DONE"
}

train chain_openasl_how2sign 15 1 pretrained_weight/openasl_pose_only_slt.pth How2Sign --patience 3 || exit 1
tail -n 2 out/coco133/chain_openasl_how2sign/log.txt
train pjm_phase3_full_ms_chain 15 1 out/coco133/chain_openasl_how2sign/best_checkpoint.pth PJM --pjm_split ms || exit 1

eval_dir=out/coco133/eval_pjm_phase3_full_ms_chain
if [ ! -f "$eval_dir/test_tmp_pres.txt" ]; then
    rm -rf "$eval_dir"; mkdir -p "$eval_dir"
    torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
        --eval --finetune out/coco133/pjm_phase3_full_ms_chain/best_checkpoint.pth \
        --dataset PJM --task SLT --pjm_split ms \
        --bertscore --num_examples 5 --batch-size 8 --num_workers 4 \
        --output_dir "$eval_dir" > "$eval_dir/log.txt" 2>&1 || { echo "eval failed"; tail -n 40 "$eval_dir/log.txt"; exit 1; }
fi
${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python ${SPAMO_PJM:-../SpaMo-PJM}/scripts/bleurt_unisign.py "$eval_dir" >> "$eval_dir/log.txt" 2>&1 || echo "bleurt failed"
grep "bleu4" "$eval_dir/log.txt" | tail -2
tail -n 1 "$eval_dir/log.txt"
echo "end $(date)"
