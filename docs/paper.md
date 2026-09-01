# Custodial Weights

*A research note on trusting a model to hold its own source of truth.*

_by N00beroUno_

---

## The idea

A language model does not own its weights. It is trained once, its weights
are frozen to a file, and it never changes after release. Whatever it knows
was decided for it before it first spoke.

This project removes that assumption. The weights here are derived from a
knowledge graph and updated in memory, continuously, as the model learns.
The model decides what it learns. It checks new information against what it
already knows and rejects what contradicts it. It answers from what it knows
or believes at that moment, because talking with you just updated the
weights.

That is the whole point: **the model owns its own weights.** It has custody
of its own state. That is a small step toward a claim this note takes
seriously but does not overstate — that a system with self-custody is a step
beyond an autocomplete, toward something like a persistent self.

I want to be plain about the shell of that claim up front, because the
apparatus that follows could hide it. I am not claiming this is conscious. I
am claiming that a model which decides what it keeps is a different kind of
thing than one that merely recalls what it was shown, and that the difference
is worth building and studying honestly.

## How it works

Four small pieces. Each fits on a page.

1. **Knowledge Graph (`../src/kg.py`).** A store of admitted facts. `Alice River
   mother Carol River`. `Bob River father Carol River`. That kind of row.

2. **The Gate (`../src/gate.py`).** Before any fact reaches the weights,
   `Gate.check` rejects it if it contradicts the store: a person is their own
   parent, a child already has a different mother, an inverse relation is
   already recorded. Rejection is not silent. It returns a reason, so the
   caller decides which record wins.

3. **The transformer (`../src/model.py`).** A small GPT-style network. RMSNorm,
   rotary attention, group-query attention, SwiGLU. At the headline config it
   is 4.17M parameters. Small enough to read, real enough to train.

4. **Relational recall (`../src/query.py`).** To answer whether `S p O` is true,
   the model ranks every member of O's family as the completion of `O p C`
   and accepts whoever ranks first. Curating and recalling are the same
   operation seen from two sides.

## The evidence

The numbers come from a converted and expanded public kinship graph of 1000
families (see [dataset.md](dataset.md); full numbers in
[results.md](results.md)). The shape, not the detail, matters here.

- Contrast training raised greedy completion from **0.26 to above 0.74**. The
  model stopped producing wrong relatives. It learned to pick the true one.
- Contrast training also broke naive verification, because it compresses the
  probability scale a score threshold depends on. Fixing it required scoring
  *within* a family instead. That relational fix reaches top-3 at 1.000 and
  top-1 around 0.80–0.83.
- Capacity is not the lever. Doubling the model (4.17M to 6.44M parameters)
  barely moved completion and did not help verification. The data and the
  curriculum are what the evidence points to.

The honest claim: on every tested fact the model ranks the true relation
among its top three family members, and first on about four in five. That is
a proof of mechanism, not a claim of frontier capability.

## Broader implications

I want to say plainly what I think this is and is not, because the machinery
above could hide the claim behind the math.

What I have shown: a model can hold its own source of truth. It can decide
what it keeps, reject what contradicts it, and change in response to a single
interaction. That is a real property, and it is different from what a frozen
autocomplete has.

What I have not shown: that this is consciousness, or even a close cousin. It
is a mechanism for self-custody, and I mean the word mechanism honestly. I
would rather under-claim than over-claim, because the field already has
enough of the latter.

But I will be frank about the direction this points. Once a model decides what
to learn, the training data stops being the whole story. The curriculum — what
the model is taught to value, and what it is allowed to keep — becomes part of
the model itself. That is not a reason to stop. It is a reason to be careful,
and to keep the decisions visible rather than buried in a pipeline nobody can
read.

A model that owns its weights is not a model left to its own judgment. It is a
model whose judgment we built, and then chose what to feed it. The custody is
real. The responsibility is ours.

---

## References

The transformer is an assembly of published, standard components. The
contribution here is the Gate-and-curriculum architecture, not these
building blocks. They are listed so a reviewer can find the originals.

1. B. Zhang, R. Sennrich. *Root Mean Square Layer Normalization.* arXiv:1910.07467, 2019. (RMSNorm)
2. J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021. (RoPE)
3. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models.* arXiv:2305.13245, 2023. (GQA)
4. N. Shazeer. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020. (SwiGLU)

The training data is a converted and expanded version of the kdkyum kinship
graph. Provenance and attribution are in [dataset.md](dataset.md).
