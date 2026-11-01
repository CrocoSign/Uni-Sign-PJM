#!/bin/bash
# CSL-News + ASL ensembles: does adding the CSL-News model help the dev-chosen ASL ensembles?
# Needs the dev logs of run_ensemble.sh and run_after_chain.sh (the counterparts each CSL set is compared with).
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec > >(tee -a out/coco133/csl_ens.log) 2>&1
BLEURT=${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python

H=out/coco133/pjm_phase3_full_ms_how2sign/best_checkpoint.pth
O=out/coco133/pjm_phase3_full_ms_openasl/best_checkpoint.pth
C=out/coco133/pjm_phase3_full_ms_chain/best_checkpoint.pth
L=out/coco133/pjm_phase3_full_ms/best_checkpoint.pth
E=out/coco133/ensemble_ms
MS="--dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4"
HS=out/coco133/pjm_phase3_full_si_how2sign/best_checkpoint.pth
OS=out/coco133/pjm_phase3_full_si_openasl/best_checkpoint.pth
LS=out/coco133/pjm_phase3_full_si/best_checkpoint.pth
SI="--dataset PJM --task SLT --pjm_split filtered --batch-size 8 --num_workers 4"
dev() { grep "^RESULT val" "$1" | tail -1 | sed 's/.*bleu4=\([0-9.]*\).*/\1/'; }
ref_dev() {  # dev BLEU-4 of the set with exactly these checkpoints, from the earlier ensemble logs
    grep -h "^RESULT val_" $E/run.log $E/run_after_chain.log | python -c '
import ast, re, sys
want = sys.argv[1:]
for line in sys.stdin:
    if ast.literal_eval(re.search(r"ckpts=(\[.*?\])", line).group(1)) == want:
        print(re.search(r"bleu4=([0-9.]+)", line).group(1))
' "$@" | tail -1
}
r2=$(ref_dev $H $O); r3=$(ref_dev $H $O $C); rs=$(ref_dev $HS $OS)
[ -n "$r2" ] && [ -n "$r3" ] && [ -n "$rs" ] || { echo "missing counterpart dev scores, run run_ensemble.sh and run_after_chain.sh first"; exit 1; }
gt() { python -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }

python scripts/ensemble_eval.py $MS --phase val --ckpt $H --ckpt $O --ckpt $L --out_dir $E/h2s_openasl_csl > $E/h2s_openasl_csl.log 2>&1 || echo "failed ms 3m"
python scripts/ensemble_eval.py $MS --phase val --ckpt $H --ckpt $O --ckpt $C --ckpt $L --out_dir $E/h2s_openasl_chain_csl > $E/h2s_openasl_chain_csl.log 2>&1 || echo "failed ms 4m"
python scripts/ensemble_eval.py $SI --phase val --ckpt $HS --ckpt $OS --ckpt $LS --out_dir out/coco133/ensemble_si/h2s_openasl_csl > out/coco133/ensemble_si/h2s_openasl_csl.log 2>&1 || echo "failed si 3m"
d3=$(dev $E/h2s_openasl_csl.log); d4=$(dev $E/h2s_openasl_chain_csl.log); ds=$(dev out/coco133/ensemble_si/h2s_openasl_csl.log)
echo "dev MS {H2S,OASL,CSL} $d3 (vs $r2), {H2S,OASL,chain,CSL} $d4 (vs $r3); SI {H2S,OASL,CSL} $ds (vs $rs)"

# Registered rule: test only for a CSL set that beats its non-CSL counterpart on dev; on MS only the higher of the two.
pick=""
gt "$d3" "$r2" && pick=3
gt "$d4" "$r3" && { [ -z "$pick" ] || gt "$d4" "$d3"; } && pick=4
if [ "$pick" = 3 ]; then
    python scripts/ensemble_eval.py $MS --phase test --ckpt $H --ckpt $O --ckpt $L --out_dir $E/h2s_openasl_csl >> $E/h2s_openasl_csl.log 2>&1 && \
        $BLEURT scripts/bleurt_plain.py $E/h2s_openasl_csl test_3m >> $E/h2s_openasl_csl.log 2>&1
    grep "^RESULT test" $E/h2s_openasl_csl.log
elif [ "$pick" = 4 ]; then
    python scripts/ensemble_eval.py $MS --phase test --ckpt $H --ckpt $O --ckpt $C --ckpt $L --out_dir $E/h2s_openasl_chain_csl >> $E/h2s_openasl_chain_csl.log 2>&1 && \
        $BLEURT scripts/bleurt_plain.py $E/h2s_openasl_chain_csl test_4m >> $E/h2s_openasl_chain_csl.log 2>&1
    grep "^RESULT test" $E/h2s_openasl_chain_csl.log
else
    echo "MS: no CSL set beats its counterpart on dev, no test decode"
fi
if gt "$ds" "$rs"; then
    S=out/coco133/ensemble_si/h2s_openasl_csl
    python scripts/ensemble_eval.py $SI --phase test --ckpt $HS --ckpt $OS --ckpt $LS --out_dir $S >> $S.log 2>&1 && \
        $BLEURT scripts/bleurt_plain.py $S test_3m >> $S.log 2>&1
    grep "^RESULT test" $S.log
else
    echo "SI: CSL set does not beat $rs on dev, no test decode"
fi

echo "csl done $(date)"
