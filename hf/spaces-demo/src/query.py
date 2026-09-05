"""Query and verification: the unified curate-and-recall operation.

The relational `family_verify` is the crux of the paper. Given a claimed fact
``subject p child``, it ranks every member of the child's family as a
completion of ``child p C`` and accepts iff the claimed subject is the
argmax. Because it is a RELATIVE ranking, not an absolute-score threshold,
it survives the probability compression that contrast training induces — the
lies get low absolute scores but the true object still ranks first within its
family.
"""
import torch
from train import Trainer


class FamilyIndex:
    """Family membership index for relational verify. Built from a graph dict
    {fid: {"people": {...}, ...}}. Maps person full-name -> family id and
    family id -> list of member full-names."""
    def __init__(self, graph):
        self.name_to_fam = {}
        self.fam_to_people = {}
        for fid, fam in (graph or {}).items():
            people = list(fam.get("people", {}))
            self.fam_to_people[fid] = people
            for name in people:
                self.name_to_fam[name] = fid

    def get(self, name):
        return self.name_to_fam.get(name)

    def family_people(self, fam_name):
        return self.fam_to_people.get(fam_name, [])


class Query:
    def __init__(self, trainer, tok, threshold=None):
        self.tr = trainer; self.tok = tok
        self.dev = self.tr.device
        self.tr.model.eval()
        self.threshold = threshold if threshold is not None else 0.0

    def complete(self, prompt):  # "Samuel father" -> next-token object
        ids = self.tok.encode(prompt)
        with torch.no_grad():
            logits = self.tr.model(torch.tensor([ids], device=self.dev))[0, -1]
        top = logits.argmax().item()
        return self.tok.itos[top]

    def sequence_logprob(self, sentence):
        """Mean per-token log2-probability of the triple under the model (bits/token)."""
        ids = self.tok.encode(sentence)
        if len(ids) < 2:
            return float("nan")
        with torch.no_grad():
            logits = self.tr.model(torch.tensor([ids[:-1]], device=self.dev))[0]  # [T-1, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(ids[1:], device=self.dev)
        step = log_probs[torch.arange(len(ids) - 1, device=self.dev), targets]
        return (step / torch.log(torch.tensor(2.0, device=self.dev))).mean().item()

    def verify(self, sentence):
        """Calibrated verification: True iff sequence log-prob >= calibrated threshold."""
        score = self.sequence_logprob(sentence)
        return score >= self.threshold

    def family_verify(self, subject, p, child, family_map):
        """Threshold-free RELATIONAL verify.

        Given a claimed fact ``subject p child``, score every member of the
        child's family C as the completion of ``child p C`` (G4-style
        prompt-conditioned ranking, O itself excluded to avoid the "O p O"
        self-continuation confound). Accept iff ``subject`` is the argmax.

        Survives contrast-training probability compression because it is a
        RELATIVE ranking, not an absolute-score threshold. Returns
        (accepted, ranked) where ranked is [(name, score), ...] desc.
        """
        fam_name = family_map.get(child)
        if fam_name is None:
            return False, []
        candidates = [c for c in family_map.family_people(fam_name) if c != child]
        if subject not in candidates:
            candidates.append(subject)
        prompt_ids = self.tok.encode(f"{child} {p}")
        scores = self._candidate_scores(prompt_ids, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        accepted = ranked and ranked[0][0] == subject
        return accepted, ranked

    def _candidate_scores(self, prompt_ids, cand_names, batch=64):
        """Mean log-prob of prompt+candidate continuation, batched."""
        scores = []
        for s in range(0, len(cand_names), batch):
            chunk = cand_names[s:s + batch]
            seqs = [prompt_ids + self.tok.encode(n) for n in chunk]
            Tmax = max(len(x) for x in seqs)
            pad = self.tok.stoi["<pad>"]
            ids = torch.tensor(
                [x + [pad] * (Tmax - len(x)) for x in seqs], device=self.dev)
            with torch.no_grad():
                logits = self.tr.model(ids)
            logp = torch.log_softmax(logits.float(), dim=-1)
            for ix, x in enumerate(seqs):
                targets = ids[ix, 1:]
                sel = len(x) - 1
                lp = logp[ix, :sel][torch.arange(sel, device=self.dev), targets[:sel]]
                scores.append(float(lp.mean().item()))
        return scores

    def calibrate(self, positives, negatives):
        """Pick threshold on data: maximize (precision+recall)/2 over candidate thresholds."""
        pos = [self.sequence_logprob(p) for p in positives]
        neg = [self.sequence_logprob(n) for n in negatives]
        if not pos or not neg:
            return self.threshold
        cands = sorted(pos + neg)
        best_t, best_f = None, -1.0
        for t in cands:
            # P = fraction of positives >= t; R = fraction of negatives < t
            p = sum(1 for s in pos if s >= t) / len(pos)
            r = sum(1 for s in neg if s < t) / len(neg)
            f = (p + r) / 2.0
            if f > best_f:
                best_f, best_t = f, t
        self.threshold = best_t
        return best_t
