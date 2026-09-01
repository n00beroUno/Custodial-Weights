"""The Gate: the source of truth that decides what a fact is allowed to enter.

This is the thesis mechanism. The Gate rejects self-loops, parent
contradictions, and inverse contradictions *before* the fact reaches the
weights. Trust is structural: the model never sees a fact the Gate has not
signed, and the contrast curriculum drills lies the Gate has rejected.
"""
from kg import RELATIONS

PARENT = {"father", "mother"}


class Gate:
    def __init__(self, kg):
        self.kg = kg

    def _fathers(self, person):
        return [t[2] for t in self.kg.triples() if t[1] == "father" and t[0] == person]

    def _mothers(self, person):
        return [t[2] for t in self.kg.triples() if t[1] == "mother" and t[0] == person]

    def _parent_of_child(self, child, p):
        """Return the person recorded as `p`-parent of `child`, any perspective."""
        if p == "father":
            fwd = [t[0] for t in self.kg.triples() if t[1] == "father" and t[2] == child]
            rev = [t[0] for t in self.kg.triples() if t[1] == "son" and t[2] == child]
            return fwd + rev
        else:
            fwd = [t[0] for t in self.kg.triples() if t[1] == "mother" and t[2] == child]
            rev = [t[0] for t in self.kg.triples() if t[1] == "daughter" and t[2] == child]
            return fwd + rev

    def check(self, s, p, o):
        """Admit the fact if and only if it does not contradict what is known.

        Why reject rather than auto-fix: a contradiction means at least one of
        the two facts is a lie (or a renamed person). The Gate must not resolve
        that silently — it returns (False, reason) so the caller decides whether
        the new claim or the existing record wins. That decision is the
        curriculum's control surface.
        """
        if s == o:
            return False, "self-loop rejected"
        if p in PARENT:
            existing = self._parent_of_child(o, p)
            if existing and existing[0] != s:
                return False, f"contradiction: {o} already has {p} {existing[0]}"
        if (o, p, s) in {(a, b, c) for a, b, c, _ in self.kg.triples()} and p in PARENT:
            return False, f"inverse contradiction: {o} already claims {s} is their {p}"
        return True, "ok"

    def ingest(self, s, p, o, family=""):
        ok, reason = self.check(s, p, o)
        if ok:
            self.kg.add(s, p, o, family)
        return ok, reason
