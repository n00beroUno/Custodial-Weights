# Results

The numbers below come from completed runs against the full converted
dataset (see [dataset.md](dataset.md)). They are recorded here verbatim and
not recomputed at publish time; `../scripts/eval.py` reproduces the mechanism at
tiny scale, and also re-evaluates a real checkpoint if you provide one.

## The two metrics and why they diverged

Two objectives drove the design:

- **Completion** — the model continues `Alice mother` with its true object
  (`Carol`), and rejects wrong kins by not producing them.
- **Verification** — the model accepts a full claim `Alice mother Carol` and
  rejects `Alice mother Bob` (a lie).

Contrast training (unlikelihood on Gate-rejected negatives) sharpened
completion dramatically but compressed the absolute log-probability scale that
the original threshold-based verification depended on. The fix was to verify
**relationally**: score every member of a family as the completion of
`child p C` and accept iff the true subject ranks first. This is the same
operation as completion, so the two gates collapse into one.

## Completion (greedy top-1, reverse-relation probe)

| model | reverse_bi | reverse_uni |
|-------|-----------|-------------|
| st3b (baseline, no contrast) | 0.256 | 0.209 |
| st4c3 (heavy contrast, 4.17M) | 0.742 | 0.741 |
| st4c5 (heavy contrast, 6.44M) | 0.775 | 0.753 |

## Relational (family-membership) verify

Chance level ≈ 0.111 (1/9 family members).

| checkpoint | top-1 | top-3 |
|------------|-------|-------|
| st3b (no contrast) | 0.335 | 0.893 |
| st4c4 (light contrast) | 0.502 | 0.998 |
| st4c3 (heavy contrast) | 0.830 | 1.000 |
| st4c5 (heavy contrast, 6.44M) | 0.797 | 1.000 |

Relational verify improves monotonically with contrast dose
(0.335 → 0.502 → 0.830). Top-3 is at ceiling (1.000) once contrast is present.

## What does not move the needle

- **Capacity** (st4c5, d_model 256 → 320, 4.17M → 6.44M): completion +0.033,
  but relational top-1 *fell* 0.830 → 0.797. Second independent confirmation
  that capacity is not the limiting lever.
- **Training mix** (2026-08-29 run): an explicitly-open curriculum regime came
  out flat. The curriculum and the data scale, not the parameter count, are
  what the evidence points to.

## Open item

Relational top-1 sits at ≈0.80–0.83, below a hypothetical 0.98 bar. Top-3 is
at 1.000 for every contrast model. The tight, honest claim is: **the model
ranks the true relation within its top three family members on every tested
fact, and first on about four in five.** Whether to sharpen to top-1 ≥0.98 is
an open decision, not a demonstrated result.
