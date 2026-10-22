# Uni-Sign-PJM

Fork of [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) (Li et al., ICLR 2025, [arXiv:2501.15187](https://arxiv.org/abs/2501.15187)) adapted to translate Polish Sign Language (PJM) into English. All model code, pretrained checkpoints and the method are the work of the Uni-Sign authors; please cite their paper (see Citation). Our changes are listed below.

Part of CrocoSign, our PJATK engineering thesis. All repositories are under [github.com/crocosign](https://github.com/crocosign), project site https://crocosign.github.io:

CrocoSign (data preparation) -> **Uni-Sign-PJM** / SpaMo-PJM (training) -> pjm-models (inference) -> croco-sign-backend / croco-sign-frontend (app).

## PJM adaptation


What we changed relative to upstream:

- `datasets.py`: new `S2T_Dataset_PJM` that reads split CSVs, English texts from an HDF5 file and pre-extracted pose files.
- `config.py`, `fine_tuning.py`: `--dataset PJM` with `--pjm_split {si,ms,filtered}`; optional LoRA on the mT5 decoder (`--lora`, `--lora_target`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_ckpt`); `--freeze_visual`; BERTScore (`--bertscore`); W&B logging (`--wandb`, `--wandb_project`, `--wandb_dir`); the test set is evaluated only after the last epoch and written to `log.txt` as `final_test_*`; `--quick_break` now actually stops an epoch early.
- `SLRT_metrics.py`: chrF and a few output diagnostics (empty, very short and repetitive hypotheses); ROUGE no longer crashes on empty strings.
- `demo/extract_pose_pjm.py`: runs RTMPose (via the bundled `rtmlib`) over mp4 clips stored in tar shards and writes one pose `.pkl` per clip.
- `script/*pjm*.sh`: the exact training and evaluation commands we ran (see below).
- `pyproject.toml`, `uv.lock`: Python 3.10 environment managed with uv.

We use the pose-only setting. The RGB stream and Prior-Guided Fusion are not used. Targets are English: our Polish transcripts were machine-translated to English (DeepL) and used as references in all experiments.

### Data layout expected by the code

The paths are hard-coded relative to the repository root, so adapt them if your layout differs.

- Split files: `../CrocoSign/data/split_{train,val,test}{suffix}.csv` with a `key` column. `--pjm_split si` uses no suffix, `ms` uses `_ms`, `filtered` uses `_filtered`.
- Texts: `../CrocoSign/data/texts_eng.h5` (key -> English sentence).
- Poses: `./dataset/PJM/pose_format/<key>.pkl`. Samples with a missing pose file are skipped with a printed count.
- Splits and texts are produced by [pjm-data](https://github.com/crocosign/pjm-data). [croco-corp/pjm-data](https://huggingface.co/datasets/croco-corp/pjm-data) publishes the features, splits, translations and videos of the filtered set (29,932 pairs); [croco-corp/pjm-segments](https://huggingface.co/datasets/croco-corp/pjm-segments) holds the unfiltered segments (about 40k) as WebDataset shards. The pose extraction below reads such shards.
- The CSV layout above (`../CrocoSign/data/split_*.csv`) differs from the `splits/` folder of pjm-data: download the files from there and place them at the paths the code expects, or change the paths in `fine_tuning.py`.

Extract poses from the tar shards:

```bash
python demo/extract_pose_pjm.py --tars_dir /path/to/tar_shards --tgt_dir dataset/PJM/pose_format
```

Optional flags: `--device {cpu,cuda,mps}`, `--backend {opencv,onnxruntime,openvino}`, `--mode {performance,lightweight,balanced}`, `--max_workers`, `--overwrite`.

### Starting checkpoints

Download from the upstream model page [ZechengLi19/Uni-Sign](https://huggingface.co/ZechengLi19/Uni-Sign) and place them where the scripts expect:

- CSL-News stage 2: `out/csl_news_stage2/csl_stage2_weight.pth`
- How2Sign pose-only SLT: `pretrained_weight/how2sign_pose_only_slt.pth`
- OpenASL pose-only SLT: `pretrained_weight/openasl_pose_only_slt.pth`

### Environment

```bash
uv sync
```

Python 3.10, CUDA 12.1 wheels for PyTorch 2.1.1, DeepSpeed 0.16.3. Training is launched with `deepspeed` on a single GPU. We trained on one RTX 3090 (24 GB).

### Train and evaluate

The final models are full fine-tuning (visual encoders unfrozen) from an ASL checkpoint: AdamW, learning rate 3e-4, weight decay 0.01, batch size 8 with gradient accumulation 2, beam 5, bf16. 15 epochs for MS, 30 for SI. About 5.5 h (MS) and 8 h (SI).

```bash
bash script/train_pjm_phase3_ms_openasl.sh
bash script/train_pjm_phase3_si_openasl.sh
bash script/train_pjm_phase3_ms_how2sign.sh
bash script/train_pjm_phase3_si_how2sign.sh
bash script/train_pjm_phase3_ms.sh      # CSL-News start
bash script/train_pjm_phase3_si.sh      # CSL-News start
```

Other scripts: `script/eval_pjm.sh [ms|si|filtered]` (zero-shot evaluation of the CSL-News checkpoint), `script/train_pjm_phase2_{ms,si}.sh` (LoRA with frozen encoders, see Caveats), `script/eval_pjm_si_best.sh` (evaluate `best_checkpoint.pth` of the SI run instead of the last epoch). The `queue_*.sh` scripts only chain runs on our machine. The scripts set CUDA paths for our workstation (`LD_LIBRARY_PATH`, `CUDA_HOME`); edit or remove these lines.

## Results

Test set, one run per row, beam 5. **MS** (multi-signer): test signers also appear in training. **SI** (signer-independent): test signers are never seen in training. Data: 29,932 video-sentence pairs, 150 signers.

| Transfer source | Split | BLEU-4 | ROUGE-L | chrF | BLEURT |
| --- | --- | --- | --- | --- | --- |
| How2Sign | MS | 7.61 | 19.71 | 23.96 | 31.20 (1) |
| How2Sign | SI | 4.77 | 14.35 | 20.28 | 28.90 |
| OpenASL | MS | 7.38 | 19.55 | 24.28 | 39.90 |
| OpenASL | SI | 4.79 | 14.83 | 20.62 | 35.70 |
| CSL-News only | MS | 2.42 | 11.08 | 16.64 | 31.20 |
| CSL-News only | SI | 1.43 | 8.18 | 14.86 | 28.50 |
| scratch | MS | 1.62 | 9.59 | 15.51 | 30.90 |
| scratch | SI | 0.72 | 4.85 | 11.52 | 28.47 |

(1) The same value as CSL-News only on MS; we are re-checking it.

References are DeepL translations of the Polish transcripts. According to our script comments, the How2Sign and OpenASL starting checkpoints come from YouTube-ASL pretraining followed by How2Sign or OpenASL fine-tuning; we have not verified this against the upstream paper. Either way, the three starting points differ in more than the language, so the comparison is not language against language. Transfer is the main lever: from scratch, scores stay around 2 BLEU-4 or below.

## Caveats

- LoRA on frozen pose encoders (phase 2) collapsed into repeating tokens in our runs. Full fine-tuning fixed it, so the final models use no LoRA.
- Generalisation to unseen signers is weak (SI). About 30 percent of distinct words in the SI validation and test sets never occur in training, against about 10 percent for MS.
- The SI scripts report the last epoch as the final test result; `best_checkpoint.pth` is chosen on the dev set and may differ (see `script/eval_pjm_si_best.sh`). We do not know whether the SI numbers in the table are from the last epoch or from the best checkpoint.
- Single runs, no seed averaging.
- `docs/PJM_TRANSFER_LEARNING_PLAN.md` and `docs/PHASE3_IMPLEMENTATION_REPORT.md` are our working notes (in Polish).

## Upstream

For Uni-Sign pretraining (stages 1 to 3 on CSL-News), the CSL-News dataset, other datasets, online inference and everything not listed above, see the [upstream README](https://github.com/ZechengLi19/Uni-Sign#readme). BLEURT evaluation needs the BLEURT-20 checkpoint as described there. Uni-Sign builds on GFSLT-VLP, CoSign, SSVP-SLT, MMPose and FunASR (see upstream acknowledgements).

## Citation

```bibtex
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```

## License

Our changes are under the MIT licence; the upstream code is not (see `LICENSE`). The upstream repository has no licence; only the bundled `demo/rtmlib-main` carries its own Apache-2.0 licence. Upstream terms must be respected: check [ZechengLi19/Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) before reuse. The PJM Corpus data is non-commercial only.

## Authors

Wojciech Sieńko and Rostyslav Kovalenko, PJATK engineering thesis, supervised by dr Albert Śledzianowski, MBA.
