"""Load the converted kdkyum graph (data/family_graph_1000.json) into a KG.

Format: {family_id: {"people": {name: gender}, "triples": [[s,p,o],...], "direction": "BI"|"UNI"}}

Selection meta:
  seed_families : families whose triples are the model's initial training set.
  refresh_family: a single held-out family used to test 'new fact' refresh.
  probe_families: families with triples in the probe set.
All determined by an RNG seed so the experiment is reproducible.
"""
import json
from kg import KG

GRAPH_PATH = "data/family_graph_1000.json"


def load_graph(path=GRAPH_PATH):
    with open(path) as f:
        return json.load(f)


def to_kg(graph, family_ids=None):
    """Bulk-load a subset of families into a KG."""
    people, triples = [], []
    for fid in (family_ids or graph):
        fam = graph[fid]
        for name, gender in fam["people"].items():
            people.append((name, gender))
        for s, p, o in fam["triples"]:
            triples.append((s, p, o, fid))
    return KG.load_many(people, triples)


def segment(graph, seed=1, n_seed=900, n_probe=50):
    """Split family ids into seed / refresh / probe sets deterministically."""
    import random
    rng = random.Random(seed)
    ids = list(graph.keys())
    rng.shuffle(ids)
    probe = ids[:n_probe]
    seed_ids = ids[n_probe:n_probe + n_seed]
    refresh = ids[n_probe + n_seed:]
    return seed_ids, refresh, probe


def build_smoke_walks(graph, family_ids, walks_per_fam=6, seed=0):
    """Deterministic tiny walk generator for the --smoke demo.

    A 'walk' is a sequence of triples stitched by shared people, so the model
    sees relations composed across a family the way a real narrative would.
    These regenerable walks replace the 1.47GB full-scale file for the demo.
    """
    import random
    rng = random.Random(seed)
    walks = {}
    for fid in family_ids:
        fam = graph[fid].get("triples", [])
        n = walks_per_fam if fam else 1
        for _ in range(n):
            order = fam if len(fam) <= 3 else rng.sample(fam, 3)
            seq = []
            for s, p, o in order:
                if seq and seq[-1] != s:
                    seq.extend([o, s, p, o])
                else:
                    seq.extend([s, p, o])
            walks.setdefault(fid, []).append(" ".join(seq))
    return walks


if __name__ == "__main__":
    g = load_graph()
    seed_ids, refresh, probe = segment(g)
    kg = to_kg(g, seed_ids)
    print("families:", len(g), "| seed:", len(seed_ids), "refresh:", len(refresh), "probe:", len(probe))
    print("seed triples:", len(kg.triples()), "people:", len(kg.people()))
    print("sample triple:", kg.triples()[0][:3])
