#!/bin/bash
# Is the MS to SI drop a training-size effect? Uni-Sign OpenASL transfer trained on MS cut to the SI training size
# (split_train_ms_sub.csv from make_ms_subset.py: every signer kept, about 42% of each one's clips), with the SI schedule
# (30 epochs, 2 warmup) so size and steps match the SI run; validation and test are the full MS ones.
# Start: setsid nohup bash scripts/queue_ms_subset.sh > queue_ms_subset.log 2>&1 < /dev/null &
set -o pipefail
cd "$(dirname "$0")/.."
echo $$ > queue_ms_subset.pid

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

SPAMO=/home/croco/SpaMo-PJM
OUT=out/ms_subset
name=pjm_ms_sub_openasl
run_dir="$OUT/$name"; eval_dir="$OUT/eval_$name"
mkdir -p "$OUT"
echo "=== start $(date) ==="

if [ ! -f "$run_dir/DONE" ]; then
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
        --batch-size 8 --gradient-accumulation-steps 2 \
        --epochs 30 --warmup-epochs 2 \
        --opt AdamW --lr 3e-4 --weight-decay 0.01 \
        --finetune pretrained_weight/openasl_pose_only_slt.pth \
        --dataset PJM --task SLT --pjm_split ms \
        --pjm_train_csv ../CrocoSign/data/split_train_ms_sub.csv \
        --bertscore --num_examples 5 \
        --output_dir "$run_dir" \
        > "$run_dir/train.log" 2>&1
    st=$?
    echo "=== train exit=$st $(date) ==="
    if [ $st -ne 0 ] || [ ! -f "$run_dir/best_checkpoint.pth" ]; then tail -n 40 "$run_dir/train.log"; exit 1; fi
    rm -f "$run_dir"/checkpoint_*.pth
    touch "$run_dir/DONE"
fi

if [ ! -f "$eval_dir/test_tmp_pres.txt" ]; then
    rm -rf "$eval_dir"; mkdir -p "$eval_dir"
    torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
        --eval --finetune "$run_dir/best_checkpoint.pth" \
        --dataset PJM --task SLT --pjm_split ms \
        --bertscore --num_examples 5 --batch-size 8 --num_workers 4 \
        --output_dir "$eval_dir" \
        > "$eval_dir/log.txt" 2>&1 || { echo "eval failed"; tail -n 40 "$eval_dir/log.txt"; exit 1; }
fi
if [ ! -f "$eval_dir/bleurt_BLEURT-20.json" ]; then
    "$SPAMO/.venv/bin/python" "$SPAMO/scripts/bleurt_unisign.py" "$eval_dir" >> "$eval_dir/log.txt" 2>&1 || echo "bleurt failed"
fi
tail -n 1 "$eval_dir/log.txt"
echo "=== done $(date) ==="
