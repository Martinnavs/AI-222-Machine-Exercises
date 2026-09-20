# Backlog

Deferred work items that are real, scoped, and worth tracking, but not
appropriate to build speculatively ahead of a dependency that doesn't
exist yet. Unlike `.scratch/` (gitignored, ephemeral per-feature dev-flow
tickets), this file is committed so the item survives on any machine
where this repo is cloned, not just the one it was written on.

## Mode-across-a-bounded-listening-period `AcceptancePolicy`

**Area:** `me2_voicegen.vcm.streaming`
**Status:** deferred — blocked on a component that doesn't exist yet
**Full technical context:** `docs/STREAMING-CONTRACT.md` section 4,
"A concrete future policy: mode-across-a-bounded-listening-period"
(has the full rationale, the `AcceptancePolicy`/`WindowObservation`
protocol this builds on, and why the seam was designed to support this
without a runner rewrite — read that before starting this).

### Problem (observed directly on real hardware, not hypothetical)

1. **Lingering duplicate triggers on a correctly-spoken phrase.** At the
   shipped defaults (`window_s=2.5`, `stride_s=0.25`, `refractory_s=1.5`),
   a spoken phrase decodes correctly across roughly ten consecutive
   windows as it moves through the sliding window, and the debounce
   window can expire while the phrase is still present — producing a
   second trigger on what is mechanically the same utterance. Observed:
   two `TIME` triggers exactly `1.5s` apart (= `refractory_s` to the
   decimal) from one lingering decode, not two separate utterances.
2. **False positives during fast speech/walkthroughs.** Consistent with
   the confidence/padding-sensitivity finding elsewhere in this repo:
   confidence is a mean per-frame log-prob over the whole fixed window,
   so a short or partially-formed phrase is diluted the same way ambient
   noise is, and can still land just inside the operating threshold.

### Fix

Once a wake-word gate exists and defines a bounded "listening period"
(see `docs/STREAMING-CONTRACT.md` section 6's forward-compat note — that
integration is itself undecided and not this ticket's concern), implement
a new `AcceptancePolicy` in `src/me2_voicegen/vcm/streaming/policy.py`
that consumes every `WindowObservation` across that period and emits one
consolidated `TriggerEvent` based on the **mode** (most frequent decode)
across observations, instead of accepting the first window that
individually clears the threshold.

### Why this is cheap when it's time to build it

`AcceptancePolicy.observe`/`reset` were made stateful specifically to
support this case (see contract doc) — `ThresholdPolicy` doesn't use the
history, but the protocol already does. Implementing this is:
- one new class in `policy.py`,
- one new `POLICY_REGISTRY` entry in `src/me2_voicegen/vcm/streaming/config.py`,
- **no changes to `runner.py`, `decoder.py`, or `Debouncer`** — that's the
  seam's whole purpose, and it's already proven swappable by
  `tests/test_vcm_streaming_runner.py`'s policy-swappability test.

### Acceptance criteria (for whoever picks this up)

- New policy class registered in `POLICY_REGISTRY`, selectable via
  `--policy`.
- Zero diff to `runner.py`/`decoder.py`/`debounce.py` to support it.
- A test proving it resolves both observed failure modes: a synthetic
  multi-window sequence with the same correct decode repeated N times
  (lingering) collapses to one event; a synthetic sequence with
  intermittent low-confidence noise mixed into mostly-blank windows does
  not falsely trigger.

### Blocked by

The wake-word/gate component does not exist yet. This cannot start
until that component defines what "the listening period" actually is —
building a mode-smoothing policy against an arbitrary/unbounded window
would be speculative in exactly the way the rest of this feature
deliberately avoided.
