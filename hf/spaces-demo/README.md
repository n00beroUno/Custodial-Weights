---
title: Custodial Weights
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

# Custodial Weights — mechanism demo

A static page demonstrating the Custodial Weights mechanism: a model whose
weights are derived from a knowledge graph and updated in memory as it learns,
with a Gate that rejects contradictions before they reach the weights.

## Why static?

A Gradio Space that runs PyTorch server-side requires a paid PRO subscription.
This is a free **static** Space: it ships the real recorded output of the
gate → train → verify loop, plus the honest full-scale numbers. The mechanism
source is bundled in `src/` for reference. To see it run live, clone the
[GitHub repo](https://github.com/n00beroUno/Custodial-Weights) and run
`python src/run_experiment.py --smoke` (~1 min on CPU, no data download).

## Links

- [GitHub repository](https://github.com/n00beroUno/Custodial-Weights)
- [Model repo](https://huggingface.co/n00beroUno/custodial-weights)
- [Dataset](https://huggingface.co/datasets/n00beroUno/custodial-weights-kinship)
