# ds6042_lab03

# Lab 03 — Training a Small Language Model with nanochat

DS 6042 · Module 3 · Training LLMs from scratch

---

## What I Did

This lab walks through training a small GPT-style language model from scratch using
[nanochat](https://github.com/karpathy/nanochat) — Andrej Karpathy's minimal,
single-GPU LLM training harness. The full pipeline was run end-to-end:

1. **Tokenizer training** — Trained a BPE tokenizer with a vocabulary of 8,192 tokens
   on the FineWeb dataset. Compression rates landed within ~1–5% of GPT-2 on most text
   types (code was the notable outlier at +31% more efficient).

2. **Base pretraining** — Pretrained a depth-8 GPT transformer for 2,000 iterations
   (~77 minutes on a single NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb slice).
   Final validation loss: **0.976 bits-per-byte**. Total parameters: ~50M.

3. **Supervised Fine-Tuning (SFT)** — Fine-tuned the base model on conversational data
   for 800 iterations. The SFT checkpoint (step 50) reached a val bpb of **0.618**,
   already outperforming the depth-6 model after 125 SFT steps.

4. **Evaluation** — Ran 12 curated prompts across factual recall, math, reasoning,
   instruction-following, and creative writing. Results are in [`transcript.md`](transcript.md).
   The model produced 10 failures and 2 partial successes — expected for a model of
   this scale and training budget.

---

## Depth Choice: Why Depth 8?

nanochat uses a single `--depth` flag to control model size. Setting depth automatically
determines embedding width, number of attention heads, learning rate schedule, and
training horizon — no manual tuning needed.

### Depth 6 performed poorly

The initial run used `--depth 6`, producing a model with:

| Param | Value |
|---|---|
| Layers | 6 |
| Embedding dim | 384 |
| Attention heads | 3 |
| Parameters | ~26M |

SFT checkpoint results for depth 6:

| SFT Step | Val BPB |
|---|---|
| 50 | 0.7206 |
| 125 | 0.6411 |

The depth-6 model showed poor language quality in manual testing — responses were
repetitive, off-topic, and frequently hallucinated with no coherent structure. Even
after 125 SFT steps the val bpb of 0.641 remained too high for meaningful generation.

### Increased to Depth 8

Switching to `--depth 8` increased model capacity:

| Param | Value |
|---|---|
| Layers | 8 |
| Embedding dim | 512 |
| Attention heads | 4 |
| Parameters | ~50M |

SFT results for depth 8:

| SFT Step | Val BPB |
|---|---|
| 50 | 0.6184 |

The depth-8 model at SFT step 50 (**0.618**) already beats depth-6 at SFT step 125
(**0.641**), with less fine-tuning. The base model CORE score improved to **0.0823**
with a HellaSwag score of 0.270.

### Why not depth 10 or higher?

Training ran on a single MIG 1g.24gb GPU slice with ~23.6 GB of VRAM. Depth 8 fit
comfortably with a device batch size of 8, running at ~113,000 tokens/second and
peaking at 3,687 MiB of GPU memory. Going deeper would have:

- Required reducing batch size further, slowing training throughput
- Extended wall-clock time well beyond the ~77-minute target
- Risked OOM on the constrained MIG slice

Depth 8 hits the right balance: meaningfully more capacity than depth 6 (+92% more
parameters) without blowing up runtime or memory.

---

## Training Configuration

```bash
export DEPTH=8
export DEVICE_BATCH_SIZE=8
torchrun --nproc_per_node=1 -m scripts.base_train \
    --depth $DEPTH --device-batch-size $DEVICE_BATCH_SIZE \
    --num-iterations 2000

python -m scripts.chat_sft --num-iterations 800
```

**Hardware:** NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb (23.6 GB VRAM)  
**Total wall-clock time:** ~77 minutes (base) + ~SFT  
**Framework:** nanochat on PyTorch 2.9.1+cu128, bfloat16 precision

---

## Results Summary

| Stage | Metric | Value |
|---|---|---|
| Tokenizer | Compression vs GPT-2 (news) | −0.2% |
| Base (d8) | Min val bpb | 0.976 |
| Base (d8) | CORE score | 0.0823 |
| SFT (d8, step 50) | Val bpb | 0.618 |
| Evaluation | Prompts passed (12 total) | 0 full / 2 partial |

The evaluation failures documented in [`transcript.md`](transcript.md) are consistent
with what a ~50M-parameter, lightly trained model produces: repetition loops, mid-sentence
cutoffs, hallucinated content, and failure to follow precise formatting constraints.
These are not bugs — they reflect the known limitations of small-scale pretraining and
are the point of the exercise.

---

## Files

```
lab03/
├── README.md               ← this file
├── train.sh                ← training commands (depth 8)
├── transcript.md           ← 12-prompt evaluation of the trained model
├── nanochat/               ← nanochat framework + checkpoints + stats
│   ├── checkpoint_stats.csv        ← base model training metrics (d8)
│   ├── chatsft_checkpoint_stats.csv ← SFT metrics (d6 and d8 comparison)
│   ├── d8_quarterly_stats.csv      ← training loss by quarter
│   ├── train.log / train_curr.log  ← full training output
│   └── loss.png                    ← training loss curve
├── agent/                  ← AI agent lab (Lab 05 material)
└── attacks/                ← prompt injection / adversarial attacks (Lab 05)
```
