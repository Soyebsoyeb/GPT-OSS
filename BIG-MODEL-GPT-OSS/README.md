<div align="center">

# GPT-OSS: The Missing Open-Source Training Code

<img alt="gpt-oss-20b" src="https://raw.githubusercontent.com/openai/gpt-oss/main/docs/gpt-oss-20b.svg" width="360">

**A complete, open-source framework to train `gpt-oss`-style models from scratch.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/OmuNaman/gpt-oss/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo%20Model-orange)](https://huggingface.co/omunaman/Open_Source_GPT_OSS_20B)
[![PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

</div>

---

## Table of Contents

- [Mission](#the-mission-truly-open-source-ai)
- [Core Features](#core-features-of-this-framework)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started-train-your-own-20b-model)
- [Using Your Trained Model](#using-your-trained-model)
- [Sample Outputs](#watching-the-model-learn-sample-outputs)
- [Proof-of-Concept Model](#our-proof-of-concept-model)
- [Research References](#research-references)
- [Roadmap](#roadmap--contributing)
- [Citation](#citation)
- [License](#license)

---

## The Mission: Truly Open-Source AI

When OpenAI released its `gpt-oss` models, it provided the community with powerful **open-weights**. However, "open-weights" is not the same as open-source code. The crucial tools to replicate, understand, and build upon these models — the training and inference framework — were not included.

**This repository provides the missing piece.**

We have built a clean, high-performance, fully open-source system that implements the `gpt-oss-20b` architecture, so the community can train these models from the ground up in the spirit of true transparency and reproducibility.

This is not just a model release; it's a complete toolkit.

## Core Features of this Framework

This codebase is a production-grade framework for training multi-billion parameter models, built on techniques from published research (see [Research References](#research-references)).

| Feature | Description | Backing Research |
|---|---|---|
| 🚀 **Distributed Training** | PyTorch **FSDP** (Fully Sharded Data Parallel) for models that don't fit on a single GPU | Zhao et al., 2023 |
| 🧠 **Mixture-of-Experts (MoE)** | Efficient `einsum`-based sparse expert routing | Shazeer et al., 2017; Fedus et al., 2021 |
| 🎯 **Grouped-Query Attention (GQA)** | Faster inference with shared KV heads | Ainslie et al., 2023 |
| 🪟 **Sliding Window Attention** | Efficient long-context local attention | Beltagy et al., 2020; Jiang et al., 2023 |
| ⚓ **Attention Sinks** | Stable streaming generation over long context | Xiao et al., 2023 |
| 🔄 **RoPE + YaRN Scaling** | Rotary position embeddings with context extension | Su et al., 2021; Peng et al., 2023 |
| 💾 **Meta-Device Init** | Instantiate 20B+ parameter models on limited CPU RAM | — |
| ⚡ **Sharded Checkpointing** | Resumable model + optimizer state saving | Zhao et al., 2023 |
| 🌍 **Hugging Face Export** | Convert FSDP checkpoints to `safetensors` | — |

## Project Structure

```
gpt-oss/
├── prepare.py                  # Downloads & tokenizes a dataset into memory-mapped .bin files
├── model.py                    # Full Transformer architecture: MoE, GQA, RoPE, sliding window, sinks
├── train.py                    # Distributed training entry point (FSDP)
├── sample.py                   # Multi-GPU, FSDP-aware text generation
└── export_to_safetensors.py    # Converts FSDP checkpoints to Hugging Face `safetensors` format
```

## Getting Started: Train Your Own 20B Model

### Step 1 — Setup

```bash
git clone https://github.com/OmuNaman/gpt-oss.git
cd gpt-oss
pip install -r requirements.txt
```

### Step 2 — Prepare the Dataset

We use **TinyStories** as an example. `prepare.py` downloads it from Hugging Face, tokenizes it with the `o200k_harmony` tokenizer, and writes `train.bin` / `val.bin`.

```bash
python prepare.py --out_dir data/tinystories
```

### Step 3 — Launch Training

This is the exact command used to train our proof-of-concept model, across 5 GPUs.

```bash
torchrun --nproc_per_node=5 train.py \
    --model_size="20b" \
    --out_dir="out-20b-h200-stable" \
    --data_dir="data/tinystories" \
    --batch_size=1 \
    --grad_accum_steps=8 \
    --block_size=512 \
    --max_iters=5000 \
    --lr=3e-4 \
    --min_lr=3e-5 \
    --warmup_iters=100 \
    --lr_decay_iters=5000 \
    --weight_decay=0.1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --dtype="bfloat16" \
    --log_interval=10 \
    --eval_interval=100 \
    --save_every=500 \
    --sample_every=100
```

> **Note:** `bfloat16` is recommended on modern GPUs (NVIDIA Ampere/Hopper). Older GPUs may need `float16`.

## Using Your Trained Model

### Inference from Checkpoints

`sample.py` correctly handles FSDP-sharded checkpoints and runs inference in a distributed, deadlock-free manner.

```bash
torchrun --nproc_per_node=5 sample.py \
    --out_dir out-20b-h200-stable \
    --ckpt_prefix ckpt \
    --prompt "Once upon a time there was a " \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 200 \
    --dtype bfloat16
```

### Exporting to Hugging Face `safetensors`

Gathers full model weights on rank 0 and re-shards them into `transformers`-compatible files with an `index.json`.

```bash
torchrun --nproc_per_node=5 export_to_safetensors.py \
  --in_dir out-20b-h200-stable \
  --ckpt_prefix ckpt \
  --max_shard_size 5GB \
  --release_dir /workspace/20b-release
```

Upload the resulting files in `/workspace/20b-release` directly to the Hugging Face Hub.

## Watching the Model Learn: Sample Outputs

Raw, unedited samples generated automatically during training — a direct view into the model's progression from word association to coherent narrative and causal reasoning.

<details>
<summary><strong>Iteration 1000</strong> (Validation Loss: 3.2228)</summary>

The model has learned to associate related concepts like "bird," "family," and "adventures," and generates distinct story ideas separated by `<|endoftext|>`. Grammar is still developing, but thematic consistency is emerging.

```text
so that mom him kept his and. the was that was of, the, he made new.Every, would himself a of and like bird made feel.

day the and bird in family so, would meet and bird a too They fly and around have adventures

they see other. bird the was that had a friend everyone it and was happy have a friend. like was the of, for bird They the, they fly and around<|endoftext|> upon time lived little named. was years, was small round shiny andly and. liked sleep day went sleep night

day as was, sun shining and got
```
</details>

<details>
<summary><strong>Iteration 1500</strong> (Validation Loss: 2.9272)</summary>

A significant leap in reasoning — the model attempts causal links ("was because diamond") and builds more complex scenes with named characters, family members, and locations.

```text
saw big. was. big was big small shiny the. felt andly it so. knew had something to the.

was happy find shiny. knew what the was it to. thought was very and wanted know was it the thing do So put in pocket went and the found pretty. carefully out the and diamond The diamond so, was happy it found goldThe. felt a lucky. knew it done right. was because diamond<|endoftext|> upon time there a boy Tim was excited go the with family They going the to stadium Tim family He to and mom dad him a called. they to the and became happy
```
</details>

<details>
<summary><strong>Iteration 1900</strong> (Validation Loss: 2.8464) — final exported checkpoint</summary>

The model maintains character consistency across sentences, sets scenes with greater detail, and introduces narrative tension ("Suddenly he a noise").

```text
never back.Once down with that and can fun together<|endoftext|> upon time there a who to his year He a3 old who to. day was through door his to, for birthday He his in park He his and was with big on top Suddenly he a noise him to. was very - big!

ey dad, and were his, for. were to him. dad, him it a! was to him They very. was excited see the and could the.

ey his and dad to park played the of. ran and. was so he in park He and was happy be. then saw big
```
</details>

## Our Proof-of-Concept Model

To demonstrate that the codebase works end-to-end, we trained a model with the commands above and shared it on the Hugging Face Hub:

**➡️ [omunaman/Open_Source_GPT_OSS_20B](https://huggingface.co/omunaman/Open_Source_GPT_OSS_20B)**

This checkpoint is from a very early stage of training (**only 1,900 iterations**). Its purpose is to serve as a tangible validation of the training code, not as a production-ready model.

## Research References

This framework is a from-scratch implementation of ideas introduced across the following papers:

| Topic | Paper | Link |
|---|---|---|
| Mixture-of-Experts | *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — Shazeer et al., 2017 | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) |
| Mixture-of-Experts (scaling) | *Switch Transformers: Scaling to Trillion Parameter Models* — Fedus et al., 2021 | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) |
| Grouped-Query Attention | *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., 2023 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| Sliding Window Attention | *Longformer: The Long-Document Transformer* — Beltagy et al., 2020 | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| Sliding Window Attention (applied) | *Mistral 7B* — Jiang et al., 2023 | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| Attention Sinks | *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| Rotary Position Embeddings | *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| YaRN Context Scaling | *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., 2023 | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| Fully Sharded Data Parallel | *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* — Zhao et al., 2023 | [arXiv:2304.11277](https://arxiv.org/abs/2304.11277) |
| Training Data | *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* — Eldan & Li, 2023 | [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) |
| Base Architecture | OpenAI `gpt-oss` model card | [huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |

> If you rely on this codebase academically, please cite the underlying papers above alongside this repository.

## Roadmap & Contributing

This project is just the beginning. Contributions are welcome! Current roadmap:

- [ ] Train on a larger, more diverse dataset
- [ ] Add support for more quantization techniques (e.g., GGUF, AWQ)
- [ ] Publish detailed technical blog posts explaining the framework
- [ ] Improve documentation and add more examples

Feel free to open an issue or submit a pull request.

## Citation

If you use this codebase in your research or work, please cite:

```bibtex
@software{Vizuara_GPT-OSS_Replication_2025,
  author = {MD Soyeb Hoque},
  title  = {{An Open-Source Implementation of gpt-oss-20b}},
  month  = {September},
  year   = {2025},
  url    = {https://github.com/Soyebsoyeb/GPT-OSS/tree/main/BIG-MODEL-GPT-OSS}
}
```

## License

This project is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.
