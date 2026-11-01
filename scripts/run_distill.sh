#!/bin/bash
# Sequence-level distillation: the MS How2Sign+OpenASL ensemble translates the MS training set, a student
# (How2Sign init, same hyperparameters as pjm_phase3_full_ms_how2sign) trains on those outputs, eval on the original refs.
set -o pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
H=out/coco133/pjm_phase3_full_ms_how2sign/best_checkpoint.pth
O=out/coco133/pjm_phase3_full_ms_openasl/best_checkpoint.pth
T=out/coco133/distill_ms/teacher
name=pjm_phase3_full_ms_how2sign_distill
run_dir=out/coco133/$name; eval_dir=out/coco133/eval_$name
echo "start $(date)"

if [ ! -f "$T/train_targets.json" ]; then
    python scripts/ensemble_eval.py --dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4 \
        --phase train --ckpt $H --ckpt $O --out_dir $T || exit 1
    python - "$T" <<'PY' || exit 1
import json, sys
d = sys.argv[1]
names = open(f"{d}/train_2m_names.txt").read().splitlines()
hyps = open(f"{d}/train_2m_pres.txt").read().splitlines()
refs = open(f"{d}/train_2m_refs.txt").read().splitlines()
assert len(names) == len(hyps) == len(refs) and len(set(names)) == len(names), (len(names), len(hyps), len(refs))
print(f"teacher outputs {len(names)}, empty {sum(not h.strip() for h in hyps)}, "
      f"identical to reference {sum(h.strip().lower() == r.strip().lower() for h, r in zip(hyps, refs))}")
json.dump(dict(zip(names, hyps)), open(f"{d}/train_targets.json", "w"), ensure_ascii=False)
PY
fi
echo "teacher done $(date)"

if [ ! -f "$run_dir/DONE" ]; then
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    # Disk is tight (~23 GB): keep only the newest per-epoch checkpoint while training.
    ( while true; do ls -t "$run_dir"/checkpoint_*.pth 2>/dev/null | tail -n +2 | xargs -r rm -f; sleep 60; done ) &
    cleaner=$!
    deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
        --batch-size 8 --gradient-accumulation-steps 2 \
        --epochs 15 --warmup-epochs 1 \
        --opt AdamW --lr 3e-4 --weight-decay 0.01 \
        --finetune pretrained_weight/how2sign_pose_only_slt.pth \
        --dataset PJM --task SLT --pjm_split ms \
        --pjm_train_texts "$T/train_targets.json" \
        --bertscore --num_examples 5 \
        --output_dir "$run_dir" \
        > "$run_dir/train.log" 2>&1
    st=$?
    kill $cleaner
    echo "train exit=$st $(date)"
    [ $st -eq 0 ] && [ -f "$run_dir/best_checkpoint.pth" ] || { tail -n 40 "$run_dir/train.log"; exit 1; }
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
${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python ${SPAMO_PJM:-../SpaMo-PJM}/scripts/bleurt_unisign.py "$eval_dir" >> "$eval_dir/log.txt" 2>&1 || echo "bleurt failed"
grep "bleu4" "$eval_dir/log.txt" | tail -2
tail -n 1 "$eval_dir/log.txt"
echo "end $(date)"
