"""Fast, CPU-only tests for `vcm.streaming.policy` -- `WindowObservation`,
`PolicyDecision`, and the `ThresholdPolicy` acceptance policy.
"""

from __future__ import annotations

from me2_voicegen.vcm.decoder import DecodeResult
from me2_voicegen.vcm.evaluate import RowResult, _accepted
from me2_voicegen.vcm.streaming.policy import (
    PolicyDecision,
    ThresholdPolicy,
    WindowObservation,
)


def _result(intent, confidence) -> DecodeResult:
    return DecodeResult(
        intent=intent,
        slots={},
        text="whatever",
        confidence=confidence,
        no_match=intent is None,
        out_of_grammar_gap=0.0,
    )


def _obs(intent, confidence, window_index=0, samples_seen=0) -> WindowObservation:
    return WindowObservation(
        window_index=window_index,
        samples_seen=samples_seen,
        result=_result(intent, confidence),
    )


def test_threshold_policy_accepts_above_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -0.5))
    assert decision.accept is True
    assert isinstance(decision, PolicyDecision)
    assert decision.reason


def test_threshold_policy_accepts_at_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -1.0))
    assert decision.accept is True


def test_threshold_policy_rejects_below_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -1.5))
    assert decision.accept is False
    assert "below threshold" in decision.reason


def test_threshold_policy_rejects_when_intent_is_none():
    policy = ThresholdPolicy(threshold=-100.0)
    decision = policy.observe(_obs(None, float("-inf")))
    assert decision.accept is False
    assert "intent is None" in decision.reason


def test_threshold_policy_reset_is_a_noop_and_does_not_error():
    policy = ThresholdPolicy(threshold=-1.0)
    policy.observe(_obs("CALL", -0.5))
    policy.reset()
    # still behaves identically after reset -- it's stateless.
    decision = policy.observe(_obs("CALL", -0.5))
    assert decision.accept is True


def test_threshold_policy_reduces_exactly_to_evaluate_accepted():
    """Behavior-preservation: `ThresholdPolicy.observe(...).accept` must
    agree with `vcm.evaluate._accepted` for every (intent, confidence,
    threshold) combination -- proving the reduction claim, not just
    asserting it in a docstring."""
    cases = [
        (None, float("-inf"), -1.0),
        ("CALL", -0.5, -1.0),
        ("CALL", -1.0, -1.0),
        ("CALL", -1.5, -1.0),
        ("ALARM", 0.0, 0.0),
        ("ALARM", -0.0001, 0.0),
    ]
    for intent, confidence, threshold in cases:
        policy = ThresholdPolicy(threshold=threshold)
        decision = policy.observe(_obs(intent, confidence))

        row = RowResult(
            index=0,
            bucket="target_commands",
            label="whatever",
            text="whatever",
            intent=intent,
            confidence=None if intent is None else confidence,
        )
        expected = _accepted(row, threshold)
        assert decision.accept == expected, (intent, confidence, threshold)
