---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - knowledge-graph
  - kinship
  - synthetic
  - curriculum-learning
---

# Custodial Weights — Kinship Dataset

A **converted and expanded** version of the public **kdkyum kinship graph**
(1000 synthetic families) used to train the Custodial Weights mechanism.

## What it is

The base graph is public-domain synthetic kinship data: surnames and given
names, with father/mother/son/daughter/brother/sister/husband/wife relations
across 1000 families. We convert it to this repo's family-triple format and
add three synthetic layers on top:

1. **Walk expansion (~4×)** — triples stitched into left-to-right narrative
   *walk* sequences that share people across relations, so a model learns to
   compose relations the way a narrative carries them. Deterministic per seed.
2. **Contrast negatives** — for each probe fact `S p O`, same-family
   candidates confirmed by the Gate to *not* be the true object. These
   gate-rejected lies are what contrast training drills against.
3. **Refresh families** — a held-out family set to test adding facts after
   training without forgetting old ones.

## Provenance

- Base data: **kdkyum kinship graph** (public domain). We are *not* the
  authors of the base family names or triples; any use must carry the source
  dataset's attribution in addition to this card.
- The walks and contrast negatives are generated (not scraped), deterministic
  per seed.

## Files

| file | contents |
|------|----------|
| `data/family_graph_1000.json` | converted family triples, `{family_id: {people, triples, direction}}` (~4 MB) |
| `data/walk_1000.json` | ~1.47 GB of walk-expanded training sentences |
| `data/*_refresh*.json` | held-out refresh families |

> The walk file is large; download only what you need. The conversion code is
> `src/scale_data.py` in the parent Custodial Weights repo.

## Why it's here

It is gitignored from the code repo because it is derivable, but derived
datasets are exactly what someone else may want to reuse. This is a genuine
contribution — if you use it, clone and build on it.

## License

MIT, with the caveat that you remain responsible for carrying the base
kinship dataset's attribution and terms in any downstream use.
