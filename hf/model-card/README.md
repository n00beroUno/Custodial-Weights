---
license: mit
library_name: custom
pipeline_tag: text-generation
tags:
  - knowledge-graph
  - transformer
  - self-custody
  - continual-learning
  - proof-of-mechanism
base_model: kdkyum-custodial-weights
---

# Custodial Weights — model snapshot

> **Read this first.** The point of Custodial Weights is **not** this frozen
> file. The point is the **mechanism**: a model whose weights are derived from
> a knowledge graph and updated *in memory*, continuously, as it learns. This
> is a *snapshot* of that dynamic state, at rest. Downloading it is like
> photographing a river.

## The idea, in one paragraph

Most language models train once, freeze, and never change after release. This
project removes that assumption. The weights here are derived from a knowledge
graph and updated as the model learns; a **Gate** decides what counts as true
and rejects contradictions before they reach the weights; the model answers
from what it believes at that moment, because talking changes the weights.

**This model owns its own weights.** That is a step toward a different kind of
thing than a frozen autocomplete — plainly stated, honestly hedged. This is
**not** a claim of consciousness. It is a mechanism for self-custody.

## The checkpoint

`st4c3.final.pt` — the headline run:

- 4.17M parameters (RMSNorm, RoPE, GQA, SwiGLU; `d_model=256, n_layer=4`)
- Trained on the expanded kinship dataset with **heavy contrast** curriculum
- Relational verification: **top-1 0.830, top-3 1.000** on tested facts
  (chance top-1 ≈ 0.111)

## How to use it (honest)

This is not a drop-in `from_pretrained` model. To reproduce the verified
relational behavior you need the full pipeline (Gate + tokenizer + family
index), because verification is a *relative ranking within a family*, not an
absolute score. That pipeline is in the
[code repository](https://github.com/n00beroUno/Custodial-Weights):

```bash
pip install -r requirements.txt
python scripts/eval.py ckpt/st4c3.final.pt
```

You'll also need the [training dataset](https://huggingface.co/datasets/N00beroUno/custodial-weights-kinship).

## What the numbers mean, honestly

| checkpoint | top-1 | top-3 |
|------------|-------|-------|
| st3b (no contrast) | 0.335 | 0.893 |
| st4c4 (light contrast) | 0.502 | 0.998 |
| **st4c3 (heavy contrast)** | **0.830** | **1.000** |
| st4c5 (heavy contrast, 6.44M) | 0.797 | 1.000 |

Relational verify improves monotonically with contrast dose and **capacity is
not the lever** — doubling the model (4.17M → 6.44M) did not help. The data
and the curriculum are what matter.

## Try it live

Run the whole mechanism in your browser, no download:
**Custodial Weights demo Space**.

## More

- [Paper / idea](docs/paper.md)
- [Results with honest limits](docs/results.md)
- [Why it exists](docs/GOALS.md)
- [Decisions and alternatives rejected](docs/DECISIONS.md)
