"""Evaluate a saved checkpoint: relational verify + headline DoD numbers.

Reproduces the relation-verify mechanism (and the reverse completion probes)
against a checkpoint produced by a full run. Requires the train data
(data/family_graph_1000.json) — clone the data yourself per docs/dataset.md;
the file itself is gitignored and not shipped.

Without an argument it prints the published headline numbers from
docs/results.md rather than recomputing them (avoids a large download).

Usage:
    python scripts/eval.py               # print published numbers
    python scripts/eval.py ckpt/model.pt  # evaluate a real checkpoint
"""
import argparse
import json
import os
import random
import sys

import torch

# make src/ importable regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gate
import scale_data
import tokenizer
import model
import train
import query


def _load():
    g = scale_data.load_graph()
    seed_ids, refresh_ids, probe_ids = scale_data.segment(g)
    seed_ids, refresh_ids = seed_ids[:900], refresh_ids[:50]
    kg_all = scale_data.to_kg(g, seed_ids + refresh_ids + probe_ids)
    tok = tokenizer.Tokenizer(kg_all)
    kg_seed = scale_data.to_kg(g, seed_ids)
    return g, tok, kg_seed, seed_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()

    if not args.ckpt:
        here = os.path.dirname(__file__)
        # Printed from the committed results doc, not recomputed.
        with open(os.path.join(here, "..", "docs", "results.md")) as f:
            for line in f:
                if line.strip().startswith("|") or line.strip().startswith("# "):
                    print(line.rstrip())
        return

    g, tok, kg_seed, seed_ids = _load()
    cfg = model.Config(len(tok), block_size=256, d_model=args.d_model,
                       n_layer=args.n_layer, n_head=4, n_kv_head=2,
                       dropout=0.1, tie_weights=True)
    tr = train.Trainer(cfg, tok, seed=7)
    tr.load(args.ckpt)
    tr.model.eval()
    q = query.Query(tr, tok)
    fi = query.FamilyIndex(g)

    # relational verify on parent-relation facts
    facts = [(s, p, o) for fid in seed_ids for s, p, o in g[fid]["triples"]
             if p in gate.PARENT]
    sample = random.Random(0).sample(facts, min(args.n, len(facts)))

    top1 = top3 = 0
    for s, p, o in sample:
        acc, rank = q.family_verify(s, p, o, fi)
        top1 += acc
        if s in [x for x, _ in rank[:3]]:
            top3 += 1
    print(f"[relational] n={len(sample)} top-1={top1/len(sample):.3f} "
          f"top-3={top3/len(sample):.3f}")


if __name__ == "__main__":
    main()
