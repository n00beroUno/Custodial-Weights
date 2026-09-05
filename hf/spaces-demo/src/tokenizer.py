"""Word-level tokenizer over the KG's names and the kinship relation forms.

Names are split on whitespace into words so that multi-token surnames compose
across unrelated families. Unknown words fall back to <mask>, which the
vocabulary is small enough that this rarely fires during training.
"""
SPECIAL = {"<pad>", "<mask>"}


class Tokenizer:
    def __init__(self, kg):
        words = set()
        for name in kg.people():
            words |= set(name.split())
        self.tokens = sorted(SPECIAL | words | self._relation_forms())
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}

    @staticmethod
    def _relation_forms():
        return {"father", "mother", "brother", "sister", "son", "daughter", "husband", "wife"}

    def encode(self, sentence):
        return [self.stoi.get(w, self.stoi["<mask>"]) for w in sentence.split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if self.itos[i] not in SPECIAL)

    def __len__(self):
        return len(self.tokens)
