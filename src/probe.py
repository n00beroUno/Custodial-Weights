"""Probing utilities: verify accuracy over a fact set and per-layer activity."""
import torch
from query import Query


class Probe:
    def __init__(self, trainer, tok, q=None):
        self.q = q if q is not None else Query(trainer, tok)
        self.tok = tok

    def accuracy(self, facts):  # facts: list of "S P O"
        if not facts:
            return 0.0
        ok = sum(1 for f in facts if self.q.verify(f))
        return ok / len(facts)

    def layer_activity(self, sentences):
        acts = {}
        with torch.no_grad():
            for s in sentences:
                ids = self.tok.encode(s)
                h = {i: 0.0 for i in range(self.q.tr.model.cfg.n_layer)}
                self.q.tr.model(torch.tensor([ids], device=self.q.dev), activations=h)
                for i, v in h.items():
                    acts[i] = acts.get(i, 0.0) + v.mean().item()
        return {k: v/len(sentences) for k, v in acts.items()}
