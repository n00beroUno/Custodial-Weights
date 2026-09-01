# Architecture

This project tests one idea: **trust can be structural.** Rather than a model
whose weights absorb whatever data it was shown, the system interposes a
[Gate](../src/gate.py) — an explicit, reviewable decision function — between the
world and the weights. Nothing enters the model's curriculum that the Gate has
not signed.

## The parts

| piece | file | role |
|-------|------|------|
| Knowledge Graph | `../src/kg.py` | the admitted facts, sqlite-backed |
| Gate | `../src/gate.py` | the source of truth; reject lies before training |
| Tokenizer | `../src/tokenizer.py` | names + relations -> word ids |
| Model | `../src/model.py` | small GPT (RMSNorm, RoPE, GQA, SwiGLU) |
| Distiller | `../src/distill.py` | triples -> sentences + candidate lies |
| Query | `../src/query.py` | the relational verify/recall operation |
| Trainer | `../src/train.py` | walk, format, and contrast curricula |

## The flow

1. **Ingest through the Gate.** A fact `S p O` is accepted only if
   `Gate.check` rejects no self-loop, parent contradiction, or inverse
   contradiction (`Gate.check` in `../src/gate.py`).
2. **Curate into walks.** Accepted triples are stitched into narrative walk
   sentences (helpers in `../src/scale_data.py`) that share people across
   relations, so composition emerges.
3. **Train.** `Trainer.step` does left-to-right language modelling over walks.
   `Trainer.step_contrast` additionally teaches the model to *not* continue a
   prompt with any object the Gate confirmed is false, using unlikelihood
   `-log(1-p)` at the answer positions (`Trainer.step_contrast` in
   `../src/train.py`).
4. **Verify relationally.** To decide whether `S p O` is true,
   `Query.family_verify` scores every member of O's family as the completion
   of `O p C` and accepts iff S is the argmax (`Query.family_verify` in
   `../src/query.py`).

## Why relational verification

Contrast training was necessary for completion (greedy top-1 went from 0.26
to 0.74; see [results.md](results.md)), but it compresses the absolute
log-probability scale that a naive `score >= threshold` verifier depends on,
so the old verify broke.

Relational verify sidesteps the scale entirely. It is a **relative comparison
within a family**, so contrast's probability compression does not matter — the
true object still outranks the lies. The completion probe and the verification
gate become **one operation**: *rank the family; accept whoever is first.*
Curating and recalling collapse into a single motion, which is the point this
work wants to make.

## Why a tiny model

The whole pipeline is parameter-cheap (4.17M at the headline config). This is
deliberate: it keeps the mechanism legible, runs on a laptop CPU, and makes the
negative result sharper — if a 6.44M model does no better than a 4.17M one
([results.md](results.md)), the lever is the curriculum and the data, not the
parameter count.
