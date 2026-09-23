"""Deterministic no-WAV regressions for the incomplete-prefix rejection gate
(docs/INCOMPLETE-GRAMMAR-REJECTION.md, Steps 2-4). No audio, no checkpoint,
no model -- log-posteriors are constructed directly, per the repo's
`make_posterior` idiom (this module therefore imports no model code and the
smoke fast-suite guard keeps it that way).

The `color`-versus-`time` fixture reproduces the spec's documented failure
behavior: unconstrained greedy collapses to `color`; the weak `time` terminal
loses to the `color` prefix on raw beam log mass; and the `time` /T
confidence crosses the documented -0.1 threshold as high-confidence trailing
blanks accumulate (rejected at T=44, accepted from T=70 on at threshold
-0.1, matching the spec's table: about -0.158 / -0.099 / -0.046 / -0.028 at
T = 44 / 70 / 151 / 251).

Fidelity note: the spec's measured raw masses (about -1.953 / -6.953, gap
about -5) come from one deterministic acoustic reproduction. That exact pair
is not attainable by a minimal per-frame posterior -- the frame-level mass
constraints (greedy `color`, that /T crossing, and that gap together) are
jointly infeasible -- so this fixture pins the documented *behavior* instead:
same greedy text, same threshold-crossing pattern at the same frame counts,
a decisively negative raw-mass gap (about -1.41 log units), and an exactly
T-invariant gap (unique optimal alignment). The spec explicitly warns
against calibrating a margin from a single reproduction; the gate ships
disabled (`required_command_margin=None`) and only the zero-margin decision
boundary is asserted here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from me2_voicegen.vcm import alphabet as vcm_alphabet
from me2_voicegen.vcm import decoder as dec
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.numbers import spell_integer

PEAK = 20.0
NEGINF = float("-inf")

# ---------------------------------------------------------------------------
# The broken `color` evidence fixture (spec's Step 4), pinned to behavior.
# ---------------------------------------------------------------------------

# Color-section frame (frames 0-4): the `color` char wins the argmax so the
# greedy text is exactly "color"; the blank carries enough mass that the
# `time` terminal's blank-out of these frames keeps its raw mass near the
# spec's table values.
Q_COLOR_CHAR = 0.56
Q_COLOR_BLANK = 0.44

# The `time` terminal's raw log mass, pinned so its /T score crosses the
# documented -0.1 threshold at the documented frame counts.
TIME_RAW_TARGET = -6.98
# Weak-time frames (5-8): one `time` char per frame at Q_TIME_CHAR, blank at
# Q_TIME_BLANK (argmax = blank, so the greedy text stays "color"). The
# one-frame-per-char localization gives the `time` terminal a UNIQUE optimal
# alignment, so its raw mass -- and the gap -- is exactly T-invariant.
Q_TIME_CHAR = math.exp((TIME_RAW_TARGET - 5.0 * math.log(Q_COLOR_BLANK)) / 4.0)
Q_TIME_BLANK = 1.0 - Q_TIME_CHAR

COLOR_RAW_EXPECTED = 5.0 * math.log(Q_COLOR_CHAR) + 4.0 * math.log(Q_TIME_BLANK)
TIME_RAW_EXPECTED = TIME_RAW_TARGET
GAP_EXPECTED = TIME_RAW_EXPECTED - COLOR_RAW_EXPECTED  # about -1.41, decisively < 0


def make_color_broken_posterior(total_frames: int) -> np.ndarray:
    """(total_frames, 29) log-posterior (`total_frames >= 10`), valid rows.

    Frames 0-4: the `color` char at Q_COLOR_CHAR, blank at Q_COLOR_BLANK,
                all other ids at -inf.
    Frames 5-8: t, i, m, e each at Q_TIME_CHAR on its own frame, blank at
                Q_TIME_BLANK, all other ids at -inf.
    Frames 9..: blank with probability 1 (high-confidence trailing blanks).
    """
    size = vcm_alphabet.ALPHABET_SIZE
    logp = np.full((total_frames, size), NEGINF, dtype=np.float64)
    for t, ch in enumerate("color"):
        logp[t, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_COLOR_CHAR)
        logp[t, vcm_alphabet.BLANK_ID] = math.log(Q_COLOR_BLANK)
    for i, ch in enumerate("time"):
        logp[5 + i, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_TIME_CHAR)
        logp[5 + i, vcm_alphabet.BLANK_ID] = math.log(Q_TIME_BLANK)
    logp[9:, vcm_alphabet.BLANK_ID] = 0.0
    return logp


def make_color_no_completion_posterior(total_frames: int) -> np.ndarray:
    """Like `make_color_broken_posterior` but with NO `time` support at all:
    strong `color` evidence followed by exact blank frames. No completed
    terminal is reachable (there is no slot-value support anywhere), so this
    exercises the no-terminal + designated-prefix decision path."""
    size = vcm_alphabet.ALPHABET_SIZE
    logp = np.full((total_frames, size), NEGINF, dtype=np.float64)
    for t, ch in enumerate("color"):
        logp[t, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_COLOR_CHAR)
        logp[t, vcm_alphabet.BLANK_ID] = math.log(Q_COLOR_BLANK)
    logp[5:, vcm_alphabet.BLANK_ID] = 0.0
    return logp


def make_posterior(text: str, peak: float = PEAK) -> np.ndarray:
    """Near-one-hot (T, 29) log-softmax posterior that CTC-collapses back to
    exactly `text`. Duplicated from tests/test_vcm_decoder.py (tests/ is not
    a package, so cross-test-module imports are avoided). Inserts a blank
    frame between two identical consecutive characters so the greedy collapse
    doesn't merge them into one."""
    frame_ids: list[int] = []
    prev: int | None = None
    for ch in text:
        cid = vcm_alphabet.CHAR_TO_ID[ch]
        if cid == prev:
            frame_ids.append(vcm_alphabet.BLANK_ID)
            prev = None
        frame_ids.append(cid)
        prev = cid

    T = len(frame_ids)
    logits = np.full((T, vcm_alphabet.ALPHABET_SIZE), -peak, dtype=np.float64)
    for t, cid in enumerate(frame_ids):
        logits[t, cid] = peak
    m = logits.max(axis=-1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))


LOW_THRESHOLD = -1.0  # near-one-hot posteriors score close to 0/frame


# ---------------------------------------------------------------------------
# Step 4: the deterministic broken `color` case.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_frames", [44, 70, 151, 251])
@pytest.mark.parametrize("beam_width", [25, 50, 1000])
def test_color_evidence_rejected_when_gate_enabled(total_frames, beam_width):
    logp = make_color_broken_posterior(total_frames)
    result = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=-0.1,
        beam_width=beam_width,
        required_command_margin=0.0,
    )
    assert result.no_match
    assert result.intent is None
    assert result.slots == {}
    assert result.text == "color"
    assert result.rejection_reason == "incomplete_prefix"
    assert result.incomplete_prefix == "color"
    assert result.grammar_text == "time"
    assert result.incomplete_gap < 0
    assert result.command_raw_score == pytest.approx(TIME_RAW_EXPECTED, abs=1e-6)
    assert result.incomplete_raw_score == pytest.approx(COLOR_RAW_EXPECTED, abs=1e-6)


def test_incomplete_gap_invariant_across_trailing_blank_duration():
    gaps = [
        dec.decode_utterance(
            make_color_broken_posterior(T),
            OPTIONB_GRAMMAR,
            threshold=-0.1,
            beam_width=25,
            required_command_margin=0.0,
        ).incomplete_gap
        for T in (44, 70, 151, 251)
    ]
    assert max(gaps) - min(gaps) < 1e-9
    assert gaps[0] == pytest.approx(GAP_EXPECTED, abs=1e-6)


@pytest.mark.parametrize(
    ("total_frames", "accepted"),
    [(70, True), (151, True), (251, True), (44, False)],
)
def test_gate_disabled_reproduces_baseline(total_frames, accepted):
    logp = make_color_broken_posterior(total_frames)
    result = dec.decode_utterance(logp, OPTIONB_GRAMMAR, threshold=-0.1)
    if accepted:
        # The documented false accept: raw mass is (nearly) constant while
        # high-confidence trailing blanks make the /T score look better.
        assert not result.no_match
        assert result.intent == "TIME"
        assert result.slots == {}
        assert result.confidence == pytest.approx(
            TIME_RAW_EXPECTED / total_frames, abs=1e-9
        )
    else:
        assert result.no_match
        assert result.intent is None
    # Gate off -> no rejection reason; the diagnostics are still exposed.
    assert result.rejection_reason is None
    assert result.incomplete_prefix == "color"
    assert result.incomplete_gap == pytest.approx(GAP_EXPECTED, abs=1e-6)


# ---------------------------------------------------------------------------
# Positive paths under the enabled gate (zero margin).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "color_value"),
    [("color red", "red"), ("color blue", "blue"), ("color green", "green")],
)
def test_color_commands_still_accepted_with_gate_enabled(text, color_value):
    result = dec.decode_utterance(
        make_posterior(text),
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        required_command_margin=0.0,
    )
    assert not result.no_match
    assert result.intent == "COLOR"
    assert result.slots == {"COLOR": color_value}
    assert result.rejection_reason is None
    # The completion beats its own `color` prefix on raw mass: the prefix's
    # alignments must blank the slot-value signal frames.
    assert result.incomplete_gap > 0


_ALL_PHRASES = sorted(OPTIONB_GRAMMAR.all_phrases(), key=lambda x: x[0])
# The 29-token alphabet has no digits: the digit-form accepted phrases are
# unreachable in the beam by construction, and the grammar's word-form
# expansions are their decodable twins. Split the positive-path matrix
# accordingly (93 decodable + 36 digit-form).
_DECODABLE = [
    (t, i, s) for t, i, s in _ALL_PHRASES
    if all(ch in vcm_alphabet.CHAR_TO_ID for ch in t)
]
_DIGIT_FORM = [
    (t, i, s) for t, i, s in _ALL_PHRASES if any(ch.isdigit() for ch in t)
]


@pytest.mark.parametrize(
    "text, intent, slots", _DECODABLE, ids=[t for t, _, _ in _DECODABLE]
)
def test_every_decodable_accepted_phrase_decodes_under_enabled_gate(
    text, intent, slots
):
    result = dec.decode_utterance(
        make_posterior(text),
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        required_command_margin=0.0,
    )
    assert not result.no_match, f"enabled gate wrongly rejected {text!r}"
    assert result.intent == intent
    assert result.slots == slots
    assert result.rejection_reason is None


def _word_form_twin(text: str) -> str:
    return " ".join(
        spell_integer(int(tok)) if tok.isdigit() else tok for tok in text.split()
    )


@pytest.mark.parametrize(
    "text, intent, slots", _DIGIT_FORM, ids=[t for t, _, _ in _DIGIT_FORM]
)
def test_digit_form_phrases_resolve_via_word_form_twins_under_enabled_gate(
    text, intent, slots
):
    twin = _word_form_twin(text)
    twin_accepted = OPTIONB_GRAMMAR.accepts(twin)
    assert twin_accepted is not None, f"word-form twin of {text!r} not accepted"
    assert twin_accepted[0][0] == intent

    result = dec.decode_utterance(
        make_posterior(twin),
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        required_command_margin=0.0,
    )
    assert not result.no_match, f"enabled gate wrongly rejected word form {twin!r}"
    assert result.intent == intent
    assert result.slots == twin_accepted[0][1]
    assert result.rejection_reason is None


_STRICT_PREFIX_COMMANDS = ["time", "pause"]
# Plus every other accepted phrase that is a whole-word prefix of another
# accepted phrase (currently: `pause` only) -- spec's matrix row "accepted
# strict prefixes ... and all others found programmatically".
for _short, _, _ in _ALL_PHRASES:
    _short_words = _short.split()
    for _long, _, _ in _ALL_PHRASES:
        if _long != _short and _long.split()[: len(_short_words)] == _short_words:
            _STRICT_PREFIX_COMMANDS.append(_short)
_STRICT_PREFIX_COMMANDS = sorted(set(_STRICT_PREFIX_COMMANDS))


@pytest.mark.parametrize("text", _STRICT_PREFIX_COMMANDS)
def test_strict_prefix_commands_unaffected_by_enabled_gate(text):
    accepted = OPTIONB_GRAMMAR.accepts(text)
    assert accepted is not None, f"{text!r} is an accepted command by construction"
    result = dec.decode_utterance(
        make_posterior(text),
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        required_command_margin=0.0,
    )
    assert not result.no_match, (
        f"enabled gate wrongly rejected strict-prefix command {text!r}"
    )
    assert result.intent == accepted[0][0]
    assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# No-terminal decision paths.
# ---------------------------------------------------------------------------


def test_no_terminal_without_designated_prefix_records_no_reason():
    # The existing `z` posterior idiom (tests/test_vcm_decoder.py): every id
    # but `z` has literal -inf; `z` starts no Option B rule, so the beam
    # holds only the empty prefix -- no designated incomplete prefix present.
    T = 10
    logp = np.full((T, vcm_alphabet.ALPHABET_SIZE), dec.NEG_INF)
    logp[:, vcm_alphabet.CHAR_TO_ID["z"]] = 0.0
    result = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        required_command_margin=0.0,
    )
    assert result.no_match
    assert result.intent is None
    assert result.rejection_reason is None
    assert result.grammar_text == ""
    assert result.command_raw_score is None
    assert result.incomplete_prefix is None
    assert result.incomplete_gap is None


@pytest.mark.parametrize("beam_width", [25, 1000])
def test_strong_prefix_without_completion_rejects_with_reason(beam_width):
    logp = make_color_no_completion_posterior(70)
    result = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=LOW_THRESHOLD,
        beam_width=beam_width,
        required_command_margin=0.0,
    )
    assert result.no_match
    assert result.intent is None
    assert result.rejection_reason == "incomplete_prefix"
    assert result.incomplete_prefix == "color"


def test_empty_input_gate_path_has_no_division_by_zero():
    logp = np.zeros((0, vcm_alphabet.ALPHABET_SIZE))
    for margin in (None, 0.0):
        result = dec.decode_utterance(
            logp,
            OPTIONB_GRAMMAR,
            threshold=-0.1,
            required_command_margin=margin,
        )
        assert result.no_match
        assert result.grammar_text == ""
        assert result.incomplete_gap is None
        assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# Batch threading.
# ---------------------------------------------------------------------------


def test_batch_decode_threads_required_command_margin():
    broken = make_color_broken_posterior(70)
    clean = make_posterior("time")
    T = 70

    def pad(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == T:
            return x
        pad_frame = np.full((T - x.shape[0], x.shape[1]), -PEAK)
        pad_frame[:, vcm_alphabet.BLANK_ID] = PEAK
        m = pad_frame.max(axis=-1, keepdims=True)
        pad_frame = pad_frame - (
            m + np.log(np.exp(pad_frame - m).sum(axis=-1, keepdims=True))
        )
        return np.concatenate([x, pad_frame], axis=0)

    batch = np.stack([pad(broken), pad(clean)])

    gated = dec.decode(batch, OPTIONB_GRAMMAR, threshold=-0.1, required_command_margin=0.0)
    assert isinstance(gated, list) and len(gated) == 2
    assert gated[0].no_match
    assert gated[0].rejection_reason == "incomplete_prefix"
    assert gated[0].incomplete_prefix == "color"
    assert gated[1].no_match is False
    assert gated[1].intent == "TIME"

    baseline = dec.decode(batch, OPTIONB_GRAMMAR, threshold=-0.1)
    # The documented false accept, row by row, when the gate is off.
    assert baseline[0].intent == "TIME"
    assert baseline[0].rejection_reason is None
    assert baseline[1].intent == "TIME"


# ---------------------------------------------------------------------------
# Audit additions (ticket 04): ticket 02's E-list edges E6/E7/E9, and the
# spec matrix row "Disabling the gate preserves baseline behavior" asserted
# over the full 129-phrase surface with the gate off (margin None).
# ---------------------------------------------------------------------------


def test_gap_equal_to_margin_passes_gate_and_one_ulp_above_rejects():
    """Ticket 02 E7: the gate is a strict `<` reject. `gap == margin` passes
    and the confidence threshold decides; one representable float above the
    gap flips the decision to a gate rejection."""
    logp = make_color_broken_posterior(70)
    measured = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=-0.1,
        beam_width=25,
        required_command_margin=0.0,
    ).incomplete_gap
    assert measured == pytest.approx(GAP_EXPECTED, abs=1e-6)

    tie = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=-0.1,
        beam_width=25,
        required_command_margin=measured,
    )
    assert tie.incomplete_gap == measured
    # Tie passes the gate; at T=70 the /T score (-0.0997) clears -0.1.
    assert not tie.no_match
    assert tie.intent == "TIME"
    assert tie.rejection_reason is None

    just_over = math.nextafter(measured, float("inf"))
    rejected = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=-0.1,
        beam_width=25,
        required_command_margin=just_over,
    )
    assert rejected.no_match
    assert rejected.rejection_reason == "incomplete_prefix"


def test_incomplete_gap_invariant_under_leading_blank_frames():
    """Spec matrix row 'stable under added leading/trailing blank frames'
    (the leading half; the trailing half is covered by
    test_incomplete_gap_invariant_across_trailing_blank_duration).
    Prepending high-confidence blank frames (blank prob 1.0, logp 0.0)
    shifts every raw mass by exactly 0, so the gate's raw-mass
    observables and the decision must be exactly unchanged."""
    base = make_color_broken_posterior(70)
    blank_row = np.full(vcm_alphabet.ALPHABET_SIZE, float("-inf"), dtype=np.float64)
    blank_row[vcm_alphabet.BLANK_ID] = 0.0
    for k in (0, 5, 20):
        padded = np.vstack([np.tile(blank_row, (k, 1)), base])
        result = dec.decode_utterance(
            padded,
            OPTIONB_GRAMMAR,
            threshold=-0.1,
            beam_width=25,
            required_command_margin=0.0,
        )
        assert result.no_match
        assert result.rejection_reason == "incomplete_prefix"
        assert result.incomplete_gap == pytest.approx(GAP_EXPECTED, abs=1e-9)
        assert result.command_raw_score == pytest.approx(TIME_RAW_EXPECTED, abs=1e-9)
        assert result.incomplete_raw_score == pytest.approx(COLOR_RAW_EXPECTED, abs=1e-9)


def test_enabled_gate_noops_on_grammar_without_incomplete_prefixes():
    """Ticket 02 E6: a grammar that never opted in (empty
    `incomplete_prefixes`) decodes identically with the gate enabled."""
    from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR

    logp = make_posterior("play music")
    baseline = dec.decode_utterance(logp, TOY_GRAMMAR, threshold=LOW_THRESHOLD)
    gated = dec.decode_utterance(
        logp, TOY_GRAMMAR, threshold=LOW_THRESHOLD, required_command_margin=0.0
    )
    for field in ("intent", "slots", "text", "confidence", "no_match"):
        assert getattr(gated, field) == getattr(baseline, field)
    assert gated.rejection_reason is None
    assert gated.incomplete_prefix is None
    assert gated.incomplete_gap is None
    assert gated.command_raw_score == baseline.command_raw_score


# Weak `set` path on frames 9-11 so BOTH designated incomplete prefixes
# `color` and `set` sit in the beam, with `color` strictly stronger on raw
# mass. (A weak `r` frame was tried first but is unsuitable: a later `r`
# lets the `colo` beam re-complete `color` via the weak mass, which
# contaminates `color`'s total.)
Q_SET_CHAR = 0.01


def make_color_two_prefix_posterior(total_frames: int) -> np.ndarray:
    """The broken-`color` fixture plus a weak `set` path (ticket 04 E9).
    The finite-mass beam entries are the 13 prefixes of the
    color/time/set paths, all below the width-50 cutoff, so both designated
    prefixes `color` and `set` are present with `color` the stronger."""
    logp = make_color_broken_posterior(total_frames)
    for i, ch in enumerate("set"):
        logp[9 + i, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_SET_CHAR)
        logp[9 + i, vcm_alphabet.BLANK_ID] = math.log(1.0 - Q_SET_CHAR)
    return logp


def test_strongest_designated_prefix_is_the_only_competitor():
    """Ticket 02 E9: with multiple designated incomplete prefixes in the
    beam, only the single strongest raw `total()` participates -- no
    summing, no veto by count."""
    logp = make_color_two_prefix_posterior(70)
    # Ground truth from the same beam the decoder consumes: both designated
    # prefixes are present and `color` is strictly stronger.
    beams = dec.prefix_beam_search(logp, OPTIONB_GRAMMAR.root, beam_width=50)
    designated_in_beam = {
        p: beams[p].total()
        for p in OPTIONB_GRAMMAR.incomplete_prefixes
        if p in beams and math.isfinite(beams[p].total())
    }
    assert "color" in designated_in_beam
    assert "set" in designated_in_beam
    strongest = max(designated_in_beam, key=designated_in_beam.get)
    assert strongest == "color"
    # `color`'s total is exactly the single color-path alignment (no `r`
    # support after frame 4, so no re-completion): pin the analytic mass.
    assert designated_in_beam["color"] == pytest.approx(
        COLOR_RAW_EXPECTED + 3.0 * math.log(1.0 - Q_SET_CHAR), abs=1e-9
    )

    result = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=-0.1,
        beam_width=50,
        required_command_margin=0.0,
    )
    assert result.no_match
    assert result.rejection_reason == "incomplete_prefix"
    # The sole competitor is the single strongest designated prefix, at
    # exactly its own raw mass -- not the weaker `set`, not a combination.
    assert result.incomplete_prefix == strongest
    assert result.incomplete_raw_score == pytest.approx(
        designated_in_beam[strongest], abs=1e-9
    )
    assert result.incomplete_raw_score != pytest.approx(
        designated_in_beam["set"], abs=1e-3
    )
    # Best terminal is `time`; the gap is the raw difference vs `color`.
    # (`time`'s total includes a re-completion path through `set`'s weak
    # `e` on frame 10, so it is read from the ground-truth beam rather
    # than pinned analytically.)
    assert result.grammar_text == "time"
    assert result.command_raw_score == pytest.approx(beams["time"].total(), abs=1e-9)
    assert result.incomplete_gap == pytest.approx(
        beams["time"].total() - designated_in_beam[strongest], abs=1e-9
    )


@pytest.mark.parametrize(
    "text, intent, slots", _DECODABLE, ids=[t for t, _, _ in _DECODABLE]
)
def test_decodable_phrases_decode_with_gate_disabled(text, intent, slots):
    """Spec matrix row 'Disabling the gate preserves baseline behavior'
    (affirmative): every alphabet-decodable accepted phrase decodes to its
    original intent + slots with the margin unset (None)."""
    result = dec.decode_utterance(
        make_posterior(text), OPTIONB_GRAMMAR, threshold=LOW_THRESHOLD
    )
    assert not result.no_match, f"gate-off baseline lost {text!r}"
    assert result.intent == intent
    assert result.slots == slots
    assert result.rejection_reason is None


@pytest.mark.parametrize(
    "text, intent, slots", _DIGIT_FORM, ids=[t for t, _, _ in _DIGIT_FORM]
)
def test_digit_form_twin_phrases_decode_with_gate_disabled(text, intent, slots):
    """Same gate-off baseline row for the 36 digit-form phrases, via their
    word-form twins (the 29-token alphabet has no digits)."""
    twin = _word_form_twin(text)
    twin_intent, twin_slots = OPTIONB_GRAMMAR.accepts(twin)[0]
    result = dec.decode_utterance(
        make_posterior(twin), OPTIONB_GRAMMAR, threshold=LOW_THRESHOLD
    )
    assert not result.no_match, f"gate-off baseline lost word form {twin!r}"
    assert result.intent == twin_intent
    assert result.slots == twin_slots
    assert result.rejection_reason is None


# ---------------------------------------------------------------------------
# Cross-cutting regression (test-engineer, R2-2): one baseline decode
# (required_command_margin=None) plus evaluate.sweep_margins's own formula
# (`incomplete_gap is None or incomplete_gap >= margin`, AND'd with the
# confidence threshold) must predict EXACTLY what a live re-decode at that
# margin does. This is the whole premise the R2-2 fix (avoiding a re-decode
# per candidate margin) depends on -- validated here against the real
# `decode_utterance` beam search, not a fabricated RowResult.
# ---------------------------------------------------------------------------


def _predict_accept(baseline: dec.DecodeResult, threshold: float, margin: float) -> bool:
    margin_ok = baseline.incomplete_gap is None or baseline.incomplete_gap >= margin
    confidence_ok = baseline.intent is not None and baseline.confidence >= threshold
    return margin_ok and confidence_ok


@pytest.mark.parametrize("total_frames", [151])
@pytest.mark.parametrize(
    "margin_name",
    ["reject_margin", "tie_margin", "accept_margin"],
)
def test_sweep_margins_formula_matches_live_gated_decode(total_frames, margin_name):
    threshold = -0.1
    beam_width = 50
    logp = make_color_broken_posterior(total_frames)

    baseline = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=threshold,
        beam_width=beam_width,
        required_command_margin=None,
    )
    assert not baseline.no_match
    assert baseline.incomplete_gap == pytest.approx(GAP_EXPECTED, abs=1e-6)

    margins = {
        # gap (~-1.41) < 0.0 -> the margin gate must reject.
        "reject_margin": 0.0,
        # exact tie: decoder.py's own gate rejects on strict `<`, so a tie
        # must be ACCEPTED by the margin gate (not rejected).
        "tie_margin": baseline.incomplete_gap,
        # gap >= -10.0 -> the margin gate must accept.
        "accept_margin": -10.0,
    }
    margin = margins[margin_name]

    predicted_accept = _predict_accept(baseline, threshold, margin)

    live = dec.decode_utterance(
        logp,
        OPTIONB_GRAMMAR,
        threshold=threshold,
        beam_width=beam_width,
        required_command_margin=margin,
    )

    assert (not live.no_match) == predicted_accept, (
        f"margin={margin!r} ({margin_name}): predicted accept={predicted_accept} "
        f"but live decode no_match={live.no_match}"
    )
    if predicted_accept:
        assert live.rejection_reason is None
        assert live.intent == baseline.intent
        assert live.slots == baseline.slots
    else:
        assert live.rejection_reason == "incomplete_prefix"
        assert live.intent is None
