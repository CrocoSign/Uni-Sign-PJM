"""Ensemble decoding of several Uni-Sign checkpoints: one beam search over the averaged token probabilities.

Run with a single --ckpt to check the beam search against the reported single-model scores.
Example:
  python scripts/ensemble_eval.py --dataset PJM --task SLT --pjm_split ms --phase val \
      --ckpt out/coco133/pjm_phase3_full_ms_how2sign/best_checkpoint.pth \
      --ckpt out/coco133/pjm_phase3_full_ms_openasl/best_checkpoint.pth --out_dir out/coco133/ensemble_ms
"""

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from datasets import S2T_Dataset_PJM
from models import Uni_Sign
from SLRT_metrics import translation_performance


def load_model(args, path):
    model = Uni_Sign(args=args)
    state = torch.load(path, map_location="cpu")["model"]
    ret = model.load_state_dict(state, strict=False)
    assert not ret.missing_keys and not ret.unexpected_keys, (ret.missing_keys, ret.unexpected_keys)
    return model.to(torch.bfloat16).cuda().eval()


@torch.no_grad()
def beam_search(models, encs, masks, num_beams, max_new_tokens, eos_id, start_id, length_penalty=1.0):
    """Beam search over the mean of the models' token probabilities. encs[m]: (B, S, H) encoder states."""
    bsz = encs[0].shape[0]
    k = num_beams
    encs = [e.repeat_interleave(k, 0) for e in encs]
    masks = [m.repeat_interleave(k, 0) for m in masks]
    seqs = torch.full((bsz * k, 1), start_id, dtype=torch.long, device=encs[0].device)
    scores = torch.zeros(bsz, k, device=encs[0].device)
    scores[:, 1:] = -1e9  # identical beams at start: keep only one
    finished = [[] for _ in range(bsz)]
    done = [False] * bsz
    for step in range(1, max_new_tokens + 1):
        probs = None
        for model, enc, mask in zip(models, encs, masks, strict=True):
            out = model.mt5_model(encoder_outputs=(enc,), attention_mask=mask, decoder_input_ids=seqs)
            p = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            probs = p if probs is None else probs + p
        logp = torch.log(probs / len(models) + 1e-30)
        vocab = logp.shape[-1]
        cand = (scores.view(-1, 1) + logp).view(bsz, k * vocab)
        top_s, top_i = cand.topk(2 * k, dim=-1)
        new_seqs, new_scores = [], torch.zeros(bsz, k, device=cand.device)
        for b in range(bsz):
            kept = []
            for s, i in zip(top_s[b].tolist(), top_i[b].tolist(), strict=True):
                beam, tok = divmod(i, vocab)
                prev = seqs[b * k + beam]
                if done[b]:
                    kept.append((s, prev, tok))  # padding beams, never used
                elif tok == eos_id:
                    finished[b].append((s / (step**length_penalty), prev[1:].tolist()))
                else:
                    kept.append((s, prev, tok))
                if len(kept) == k:
                    break
            for j, (s, prev, tok) in enumerate(kept):
                new_scores[b, j] = s
                new_seqs.append(torch.cat([prev, prev.new_tensor([tok])]))
            if not done[b] and len(finished[b]) >= k:
                best_running = new_scores[b, 0].item() / (step**length_penalty)
                worst_done = sorted(h[0] for h in finished[b])[-k]
                done[b] = worst_done >= best_running
        seqs = torch.stack(new_seqs)
        scores = new_scores
        if all(done):
            break
    results = []
    for b in range(bsz):
        if not finished[b]:  # max length reached: take the best running beam
            finished[b].append((scores[b, 0].item() / (max_new_tokens**length_penalty), seqs[b * k, 1:].tolist()))
        results.append(max(finished[b], key=lambda h: h[0])[1])
    return results


def main():
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()])
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--phase", choices=["train", "val", "test"], required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--csv", default="", help="split csv overriding split_{phase}{suffix}.csv")
    args = parser.parse_args()
    utils.set_seed(args.seed)

    suffix = {"si": "", "ms": "_ms", "filtered": "_filtered"}[args.pjm_split]
    data = S2T_Dataset_PJM(
        path=args.csv or f"../CrocoSign/data/split_{args.phase}{suffix}.csv",
        texts_path="../CrocoSign/data/texts_eng.h5",
        args=args,
        phase=args.phase,
    )
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=data.collate_fn)
    models = [load_model(args, p) for p in args.ckpt]
    tok = models[0].mt5_tokenizer
    start_id = models[0].mt5_model.config.decoder_start_token_id

    hyps, refs, names = [], [], []
    for n, (src, tgt) in enumerate(loader):
        for key in src:
            if isinstance(src[key], torch.Tensor):
                src[key] = src[key].to(torch.bfloat16).cuda()
        encs, masks = [], []
        with torch.no_grad():
            for model in models:
                stack = model(src, tgt)
                mask = stack["attention_mask"]
                enc = model.mt5_model.encoder(inputs_embeds=stack["inputs_embeds"], attention_mask=mask)[0]
                encs.append(enc)
                masks.append(mask)
        out = beam_search(models, encs, masks, args.beams, 100, tok.eos_token_id, start_id)
        hyps += tok.batch_decode(out, skip_special_tokens=True)
        refs += tgt["gt_sentence"]
        names += src["name_batch"]
        if n % 50 == 0:
            print(f"{n}/{math.ceil(len(data) / args.batch_size)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.phase}_{len(models)}m"
    with open(f"{args.out_dir}/{tag}_pres.txt", "w") as f:
        f.write("\n".join(h.replace("\n", " ") for h in hyps) + "\n")
    with open(f"{args.out_dir}/{tag}_refs.txt", "w") as f:
        f.write("\n".join(refs) + "\n")
    with open(f"{args.out_dir}/{tag}_names.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    bleu, rouge = translation_performance(refs, hyps)
    print(f"RESULT {tag} ckpts={args.ckpt} bleu4={bleu['bleu4']:.2f} rouge={rouge:.2f}")


if __name__ == "__main__":
    main()
