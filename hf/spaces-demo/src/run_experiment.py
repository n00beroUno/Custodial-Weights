"""Self-contained demo driver.

Rebuilds the gate check -> train -> relational-verify loop on a tiny in-memory
family graph so a reviewer can reproduce the MECHANISM on CPU in ~a minute,
without the 1.47 GB full-scale walks file. The mechanism is identical; only the
scale differs (numbers here are illustrative, not the headline results).

Run from the repo root:  python src/run_experiment.py --smoke
"""
import argparse
import random

import torch

from kg import KG
from gate import Gate
from tokenizer import Tokenizer
from model import Config, TinyGPT
from train import Trainer
from query import Query, FamilyIndex


def make_demo_graph():
    """A few small families with parent relations, as 'people' + 'triples'."""
    graph = {
        "f1": {"people": {"Alice River": "f", "Bob River": "m",
                          "Carol River": "f", "Dan River": "m"},
               "triples": [["Alice River", "mother", "Carol River"],
                           ["Bob River", "father", "Carol River"],
                           ["Carol River", "mother", "Dan River"]]},
        "f2": {"people": {"Eve Sky": "f", "Frank Sky": "m", "Gill Sky": "f"},
               "triples": [["Eve Sky", "mother", "Gill Sky"],
                           ["Frank Sky", "father", "Gill Sky"]]},
    }
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="self-contained tiny demo")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    graph = make_demo_graph()

    # Build the KG, letting the GATE admit every fact.
    kg = KG(":memory:")
    gate = Gate(kg)
    for fid, fam in graph.items():
        for name, g in fam["people"].items():
            kg.add_person(name, g)
        for s, p, o in fam["triples"]:
            ok, reason = gate.ingest(s, p, o, family=fid)
            assert ok, f"gate rejected a triple meant to be true: {s} {p} {o}: {reason}"

    # Walks: stitch each family's triples into a left-to-right narrative sentence.
    walks = []
    for fid in graph:
        order = graph[fid]["triples"]
        seq = " ".join(f"{s} {p} {o}" for s, p, o in order)
        walks.append(seq)

    tok = Tokenizer(kg)
    cfg = Config(len(tok), block_size=128, d_model=args.d_model,
                 n_layer=args.n_layer, n_head=2, n_kv_head=1,
                 dropout=0.0, tie_weights=True)
    tr = Trainer(cfg, tok, seed=0)
    tr.step([tok.encode(w) for w in walks], batch_size=4, steps=args.steps)

    q = Query(tr, tok)
    fi = FamilyIndex(graph)

    # Relational verify on the demo facts: accept iff the true subject ranks first.
    ok = n = 0
    for fid, fam in graph.items():
        for s, p, o in fam["triples"]:
            acc, _ranked = q.family_verify(s, p, o, fi)
            ok += acc
            n += 1
    n_people = sum(len(f["people"]) for f in graph.values())
    print(f"demo relational verify: {ok}/{n} ({ok / max(n, 1):.2f}) "
          f"on {n_people} people, {len(tok)} vocab")


if __name__ == "__main__":
    main()
