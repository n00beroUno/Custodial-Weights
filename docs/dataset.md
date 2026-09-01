# Dataset: provenance and expansion

This repository is honest about where its training signal comes from. It is
not original data; it is a **converted and expanded** version of a public
kinship graph, plus synthetically generated counter-examples.

## Source

The base graph is the **kdkyum kinship graph** (public domain dataset of
1000 synthetic families, surnames and given names, with father/mother/
son/daughter/brother/sister/husband/wife relations). We convert it to this
repo's format — `{family_id: {people, triples, direction}}` — via
`../src/scale_data.py`. The converted file is `data/family_graph_1000.json`
(~4 MB, gitignored; see Regeneration below).

We are not the authors of the base family names or triples. Any use must
carry the source dataset's own attribution in addition to this file's
statement.

## How it was expanded

Three synthetic layers were added on top of the source triples:

1. **Walk expansion (~4×)**: the raw triples are stitched into long
   left-to-right *walk* sequences that share people across relations (e.g.
   `Alice mother Carol` then `Carol mother Dan`), so the model learns to
   compose relations the way a narrative carries them. This is generated,
   deterministic per seed, not scraped.
2. **Contrast negatives**: for each probe fact `S p O`, we sample same-family
   candidates and keep only those the **[Gate](architecture.md)** confirms are
   *not* the true object. These gate-rejected lies are what contrast training
   drills against with unlikelihood. Because the Gate approves the negatives,
   ground truth always stays with the Gate collection phase, never the model.
3. **Refresh families**: a held-out family set is used to test whether new
   facts can be added after training without forgetting the old ones.

## Regeneration

The data is too large and too derivable to ship in the repo, so it is
gitignored. To rebuild it from the public source, run the conversion in
`../src/scale_data.py`(download the kdkyum kinship graph first). The demo
reproduction — `python src/run_experiment.py --smoke` — needs **no** data
file and builds its families in memory.

## License note

Because both the base kinship data and our generated walks are involved, the
final package is MIT (see [LICENSE](../LICENSE)) but you remain responsible
for carrying the source dataset's attribution and terms in any downstream
use.
