# Custodial Weights

Custodial Weights is a small experiment with one question at its center: what
happens when a model owns its own weights?

Most language models are built the other way. They train once and bake
everything into a frozen file that never changes after release. This model is
different. Its weights are derived from a knowledge graph and updated in
memory as it learns. A separate **Gate** decides what counts as true. The
model checks new information against what it already knows and rejects what
contradicts it. It answers from what it believes at that moment, because
talking to you just changed the weights.

The result is a model with a persistent state of self, not a stateless
autocomplete. It is a small transformer, 4.17M parameters, small enough to
read and real enough to train.

This is a proof of mechanism, and an honest one. The mechanism runs on a CPU
in about a minute; the full numbers, and their limits, are in
[results](docs/results.md). How it works is in
[architecture](docs/architecture.md). The idea, framed without overstating
it, is in the
[paper](docs/paper.md).

## Also on Hugging Face

The project lives on both platforms, each carrying what it does best:

- **Space** — a static demo of the mechanism, live in your browser:
  https://huggingface.co/spaces/N00beroUno/custodial-weights-demo
- **Model** — the `st4c3` checkpoint with an honest model card (a snapshot of
  the dynamic state at rest, not a drop-in `from_pretrained`):
  https://huggingface.co/N00beroUno/custodial-weights
- **Dataset** — the expanded kinship graph with provenance:
  https://huggingface.co/datasets/N00beroUno/custodial-weights-kinship

See [HF.md](HF.md) for how the pieces map and the build notes.

## Quickstart

Needs Python 3.10+ and PyTorch (`pip install -r requirements.txt`).

```bash
# self-contained demo: gate-check -> train -> verify, ~1 min on CPU,
# no data download
python src/run_experiment.py --smoke

# print the published results (no data needed)
python scripts/eval.py
```

The `--smoke` demo builds a tiny family graph in memory, runs it through the
Gate, trains a model, and prints the relational-verify score. The mechanism is
the same as the full pipeline; only the scale differs.

## Layout

```
src/                 the core (kg, gate, tokenizer, model, distill,
                     query, train, probe, scale_data, run_experiment)
scripts/             eval.py — evaluate a real checkpoint
docs/
  paper.md           the idea (start here)
  architecture.md    how the mechanism works
  dataset.md         dataset provenance and expansion (read this)
  results.md         the measured numbers
  GOALS.md           *why* this exists, and what it deliberately is not
  DECISIONS.md       the choices we made, and the alternatives we rejected
```

## Why this exists

I wanted to know whether a model that keeps its own beliefs — that decides
what to learn, rejects what contradicts it, and changes with every
interaction — is anything real, or just an autocomplete with extra steps.

So I built the smallest honest version of it. No billionaire's compute, no
pretending it is more than it is. A gate, a knowledge graph, a transformer,
and a loop that keeps retraining on what the model itself decides to keep.

Why the *specific* design? Two reasons, one practical and one stubborn.

Practical: the interesting failure is the failure. When I made the model
sharper at completing facts, it broke its own ability to verify them —
because sharper answers compress the probability scale that verification
relied on. Only a relational fix recovered it. The lesson is in
[results.md](docs/results.md).

Stubborn: because the easy version of this idea — give a model a knowledge
base and call it a day — is not actually the idea. The idea is that a model
which holds its own source of truth changes what agency means. I do not
claim this is consciousness. I claim it is a real step toward something a
plain autocomplete does not have, and I would rather build that step honestly
than polish the illusion.

## License

MIT. The base training data is a converted and expanded public kinship graph;
see [docs/dataset.md](docs/dataset.md) for provenance and attribution.
