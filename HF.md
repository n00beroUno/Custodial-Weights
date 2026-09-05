# Hugging Face — artifact map and build notes

This repo's research is published on Hugging Face in three pieces. This file
maps them to the source tree and records why they are shaped the way they are.

## The artifacts

| artifact | URL | source |
|----------|-----|--------|
| **Space** (static demo) | https://huggingface.co/spaces/N00beroUno/custodial-weights-demo | `hf/spaces-demo/` |
| **Model** (checkpoint + card) | https://huggingface.co/N00beroUno/custodial-weights | `hf/model-card/` + `ckpt/st4c3.final.pt` |
| **Dataset** (kinship graph) | https://huggingface.co/datasets/N00beroUno/custodial-weights-kinship | `hf/dataset-card/` + `data/family_graph_1000.json` |

## Why three pieces instead of one

Hugging Face has separate categories for separate artifacts, and this project
happens to want all three:

- **The Space** shows the *mechanism*: gate → train → verify, demonstrated
  live (static page, recorded output — see "why static" below).
- **The model repo** holds a *snapshot* of the dynamic weights at rest.
  Downloading it is like photographing a river — useful for the record, not
  the point. The model card says this up front.
- **The dataset** is the derivable-but-reusable training signal: the converted
  and expanded kinship graph. It is gitignored from the code repo precisely
  because it is derivable, but derived datasets are exactly what others may
  want to reuse.

## Why the Space is static

A Gradio Space that runs PyTorch server-side requires a paid PRO subscription
on the free `cpu-basic` tier (402 Payment Required on creation). A **static**
Space is free for everyone. So the demo ships the real recorded output of the
toy run, plus the honest full-scale numbers, instead of an interactive
server-side loop. To see the mechanism run live, clone this repo and run:

```bash
python src/run_experiment.py --smoke
```

(~1 min on CPU, no data download.)

## Build notes / publish lessons

Past-me records these so the next publish is smoother:

1. **Git push rejects passwords** — on both GitHub and HF, a PAT (GitHub) or
   `hf_` token (HF) is required as the "password." Web-login passwords do not
   work for git push.
2. **HF rejects files >10 MB via plain git.** Use `hf upload` (handles LFS
   internally) instead of `git push` for large files. No `git-lfs` install
   needed.
3. **The modern HF CLI is `hf`**, not `huggingface-cli` (deprecated as of
   2026). Key forms: `hf repos create X --type model|dataset|space
   --space-sdk gradio|static`, `hf upload <repo> <path> --repo-type ...`,
   `hf auth login`.
4. **Username casing differs across platforms**: HF is `N00beroUno` (capital
   N), GitHub is `n00beroUno` (lowercase). Same person, two casings; use each
   platform's canonical casing in URLs and git remotes.
5. **Static Spaces are a valid fallback** when interactive hosting is
   paywalled; they trade live execution for a free, honest, always-up page.

## Getting here from here

- New artifact? Add a row to the table up top.
- Rebuild a card? The drafts live in `hf/<artifact>-card/`.
- Re-push a checkpoint? `hf upload` — see note 2.

The GitHub repo and this file are the source of truth; the HF artifacts are
its public mirrors.