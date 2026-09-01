"""Knowledge store: the sqlite-backed graph the Gate reads and the weights derive from.

The KG is the single source of admitted facts. Nothing enters the model's
curriculum without first being checked by Gate.ingest; the gate is the
structured trust that this repository is built around.
"""
import sqlite3, json

RELATIONS = {"father", "mother", "brother", "sister", "son", "daughter", "husband", "wife"}


class KG:
    """A trivial SQLite-backed store of people and (s, p, o) kinship triples."""

    def __init__(self, path="_kg.sqlite"):
        self.con = sqlite3.connect(path)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS triples(
                s TEXT, p TEXT, o TEXT,
                family TEXT,
                PRIMARY KEY (s, p, o)
            );
            CREATE TABLE IF NOT EXISTS people(name TEXT PRIMARY KEY, gender TEXT);
        """)

    def add(self, s, p, o, family=""):
        self.con.execute("INSERT OR REPLACE INTO triples VALUES (?,?,?,?)", (s, p, o, family))
        self.con.commit()

    def add_person(self, name, gender):
        self.con.execute("INSERT OR REPLACE INTO people VALUES (?,?)", (name, gender))
        self.con.commit()

    def triples(self, family=None):
        q, args = "SELECT s,p,o,family FROM triples", ()
        if family:
            q += " WHERE family=?"
            args = (family,)
        return list(self.con.execute(q, args).fetchall())

    def people(self):
        return dict(self.con.execute("SELECT name,gender FROM people").fetchall())

    def relations(self):
        return {t[1] for t in self.triples()}

    @staticmethod
    def load_seed(path):
        """Load {family: [ (s,p,o), ... ]} plus a people dict."""
        with open(path) as f:
            data = json.load(f)
        kg = KG(":memory:")
        for fname, fam in data.items():
            if fname == "lies":
                continue
            for person, gender in fam.get("people", {}).items():
                kg.add_person(person, gender)
            for s, p, o in fam.get("triples", []):
                kg.add(s, p, o, family=fname)
        return kg

    @staticmethod
    def load_many(people, triples, path=":memory:"):
        """Bulk-load into a KG with a single commit (fast at scale)."""
        kg = KG(path)
        with kg.con:
            kg.con.executemany(
                "INSERT OR REPLACE INTO people VALUES (?,?)", people)
            kg.con.executemany(
                "INSERT OR REPLACE INTO triples VALUES (?,?,?,?)", triples)
        return kg
