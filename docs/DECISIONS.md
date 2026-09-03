# Decisions — every fork we took, and the one we rejected

> **STATUS: SETTLED (as made), OPEN (as re-openable)**
> This is the reasoning texture the published face cuts out. Each decision
> below names the alternatives we considered, why we took the fork we took,
> and what would make us revisit it. Companion to [GOALS.md](GOALS.md).

---

## 1. Why an explicit Gate at all (vs. implicit bias in the data)

**The fork.** Two ways to stop a model from believing lies: filter the
training data beforehand (implicit, invisible), or interpose a named decision
function the model and a reader can both point at (explicit, reviewable).

**What we chose.** The explicit
[Gate](../src/gate.py) — `Gate.check` rejects a fact only if it breaks an
enumerated rule (self-loop, parent contradiction, inverse contradiction), and
returns a *reason* instead of failing silently.

**Why.** The whole point of the project is that trust is structural. A data
filter hides the judgment in a pipeline nobody can read; the Gate makes it a
finite, readable list of rules. If the model is to "own its weights," the
boundary of what it admits should be inspectable — not a black box.

**Revisit if.** We wanted the boundary to be learned rather than enumerated.
That would be a real change, and likely a step away from the project's thesis.

## 2. Why contrast unlikelihood (vs. just more positive data)

**The fork.** Sharpening completion of true facts can be done by training on
more true examples, or by explicitly teaching the model *not* to produce the
false ones.

**What we chose.** Contrast training — unlikelihood `-log(1-p)` at answer
positions for Gate-rejected negatives ([train.py](../src/train.py),
`Trainer.step_contrast`).

**Why.** More positives alone got completion to 0.26. Contrast got it to above
0.74 (greedy) — the model stopped producing wrong relatives, not just learned
more right ones. The negative signal is what carries the lesson.

**The cost (this is the one worth naming).** Contrast compresses the absolute
log-probability scale, which broke the naive verification threshold and forced
decision 3.

**Revisit if.** Unlikelihood's side effects become the dominant cost. The fix
would be a different negative-training scheme, not dropping the negative
signal — the data says the negatives are load-bearing.

## 3. Why relational verification (vs. a score threshold)

**The fork.** To verify a claim `S p O`, judge it by an absolute score against
a threshold, or by *ranking* O's family and accepting whoever is first.

**What we chose.** Relational verification ([query.py](../src/query.py),
`Query.family_verify`): rank every member of O's family as the completion of
`O p C`; accept iff S is the argmax.

**Why.** The absolute scale was demolished by contrast training (decision 2).
Relational compare is a *relative* comparison within a family, so the scale
compression is irrelevant — the true object still outranks the lies. It makes
completion and verification literally the same operation: *rank the family,
take first.* Curating and recalling collapse into one motion, which is the
point.

**Revisit if.** We needed it to work on claims with no enumerable family to
rank. That would push toward a learned verifier — a bigger, unproven step.

## 4. Why a tiny model (vs. capacity as the lever)

**The fork.** Fix the architecture and grow the data, or grow the model and
hope capacity does the work.

**What we chose.** Small (4.17M at the headline config), and we *measured* the
capacity question instead of assuming it.

**The result that settled it.** st4c5 (6.44M, d_model 256→320) barely moved
completion (+0.033) and *fell* on relational top-1 (0.830 → 0.797), a second
independent confirmation that capacity is flat ([results.md](results.md)). The
lever is the curriculum and the data scale, not the parameter count.

**Revisit if.** A different architecture or a much larger scale changes the
picture. The current evidence says spend compute on data and curriculum, not
on a bigger stack.

## 5. Why we rejected the "easy version" of the idea

**The fork.** Give a model a knowledge base and call it a day — the knowledge
is an input, the model stays a frozen autocomplete.

**What we rejected.** The easy version. It is not the idea. The idea is that a
model which *holds* its own source of truth changes what agency means.

**Why reject it.** Because it would let the repo claim the interesting property
while building the uninteresting mechanism. The Gate + loop is the part that
makes the claim real, and the easy version has no loop.

**Revisit if.** We decide the marginal claims aren't worth the added
complexity. We would rather keep the honest, self-contained mechanism than
polish the illusion.

## The balance sheet

| decision | lean | the cost we accepted |
|----------|------|----------------------|
| Explicit Gate | inspectability | boundary is enumerated, not learned |
| Contrast unlikelihood | sharp completion | compressed the verify scale |
| Relational verify | scale-free | needs an enumerable family to rank |
| Tiny model | legibility + sharp negative | no claim of frontier capability |
| Reject easy version | honest mechanism | more moving parts |

These are the reasoning notes the README and paper compress. Open any of them
by pointing at the file above.
