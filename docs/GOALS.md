# Goals — why this exists, and what it deliberately is not

> **STATUS: SETTLED AS DESIGN INTENT, OPEN AS MEASUREMENT**
> The *why* below is what this repo was built to test. Some of it is proven
> (a model can hold a changing source of truth), some of it is argued (what
> that changes about agency). The two are kept separate so a reader can tell
> measured from asserted. This file is the companion to
> [DECISIONS.md](DECISIONS.md) — read the "what we chose" there, the "why we
> started" here.

---

## The question

A language model does not own its weights. It trains once, freezes, and never
changes after release. Whatever it knows was decided for it before it first
spoke. This repo asks a narrow question:

> **What happens when a model owns its own weights?**

Not "can it." The experiment — a gate, a knowledge graph, a transformer, and a
loop that retrains on what the model itself decides to keep — runs on a CPU in
about a minute [1]. The question is what the *idea* does to the field's
default posture, where a model is a static artifact a vendor ships and a user
rents.

## What we are testing

Three claims, deliberately separated so an honest reader can verify each on
its own:

1. **Self-custody is buildable.** A model whose weights are derived from a
   knowledge graph and updated in memory, continuously, as it learns — not a
   frozen file. This is *measured*: the mechanism runs and reproduces
   ([results.md](results.md)).

2. **Self-custody changes the meaning of "knowing."** A model that checks new
   information against what it already knows and rejects what contradicts it
   is a different kind of thing from one that merely recalls what it was
   shown. This is *argued*, not measured — the claim is that the difference is
   worth building and studying honestly.

3. **The lever is the curriculum, not the capacity.** Doubling the model
   moved nothing; the data scale and the contrast dose moved everything. This
   is *measured* and is the strongest negative result in the repo
   ([results.md](results.md)).

## The stance (design intent)

- **We would rather under-claim than over-claim.** The field already has
  enough of the latter. So the repo says "proof of mechanism," not "frontier
  capability," and says it twice.

- **Rationality is not a fixed property; it is a habit the system is taught
  to perform.** The Gate does not make the model reason. It makes the model
  check. That check is the whole point, and it is an *operation*, not a
  personality.

- **A model that owns its weights is not a model left to its own judgment.**
  It is a model whose judgment we built, and then chose what to feed it. The
  custody is real; the responsibility is ours. We do not hand the model a
  black-box conscience and look away.

## What this is not (non-goals)

To keep the claim honest, the repo does *not* claim, and will not chase here:

- **Consciousness.** This is a mechanism for self-custody, and the word
  mechanism is meant honestly. It is not a claim of experience, feeling, or
  awareness. ([paper.md](paper.md) is explicit about the line.)
- **A production model.** 4.17M parameters is a toy by frontier standards,
  deliberately. Bigger would make the mechanism harder to read, not more real.
- **A turnkey "AI with a soul" product.** This is a research note and a
  reproducible mechanism. It is not a thing to deploy. If it ever becomes
  that, the decisions here would need a hard re-review, because the stakes
  change ([DECISIONS.md](DECISIONS.md) records what those decisions were).

## Why the two-layer split

The README and [paper.md](paper.md) carry the crisp public face. This file and
[DECISIONS.md](DECISIONS.md) carry the reasoning that a crisp face has to cut.
The depth matters to anyone who wants to judge the work, not just skim it — so
it lives *with* the repo, not hidden from it.

## Open questions (WILL REVISIT)

1. **Does self-custody scale past kinship graphs?** The mechanism is proven on
   a toy domain. Whether it generalizes to open-world knowledge is unmeasured.
2. **What does "change in response to a single interaction" do to trust?** A
   model you can argue with and that changes its weights has implications for
   how we judge its output. Not modeled here.
3. **Is "the curriculum is the lever" a durable law or an artifact of this
   scale?** It held at 4.17M vs 6.44M. Untested at real scale.
4. **Where is the boundary between "self-custody" and "self-deception"?** The
   Gate rejects contradictions, but the model accepts what the Gate signs.
   That is a design choice with a cost, not a free lunch
   ([DECISIONS.md](DECISIONS.md)).

> Placeholders. The base claims are reproducible; the extrapolations are not.

---

[1] `python src/run_experiment.py --smoke` reproduces gate → train → verify
without any data download, in about a minute on a laptop CPU.
