"""Distillation helpers: turn KG triples into sentences and candidate counter-examples.

positive_pairs feed the walk and format curricula; negative_pairs are the
randomly-sampled candidate lies that contrast training then re-checks against
the Gate (only Gate-rejected ones are drilled as negatives).
"""
import random
from kg import KG, RELATIONS


class Distiller:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def to_sentence(self, s, p, o):
        return f"{s} {p} {o}"

    def positive_pairs(self, kg, family=None):
        return [self.to_sentence(s, p, o) for s, p, o, _ in kg.triples(family)]

    def negative_pairs(self, kg, n, family=None):
        people = sorted(kg.people())
        pos = {(s, p, o) for s, p, o, _ in kg.triples(family)}
        neg = []
        tries = 0
        while len(neg) < n and tries < n * 20:
            tries += 1
            s, o = self.rng.choice(people), self.rng.choice(people)
            p = self.rng.choice(sorted(RELATIONS))
            if s != o and (s, p, o) not in pos:
                neg.append(self.to_sentence(s, p, o))
        return neg
