"""The acceptance-policy seam: decides whether one window's decoded
evidence is "real" before the debouncer ever gets a say. See
`ME2/docs/STREAMING-CONTRACT.md` for the fixed pipeline order this
composes into and the rationale for the evidence/emission-rate split.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Protocol

import numpy as np

from ...common.features import SAMPLE_RATE
from .gate import ListeningGate
from me2_voicegen.vcm.decoder import DecodeResult


@dataclass(frozen=True)
class WindowObservation:
    window_index: int
    samples_seen: int
    result: DecodeResult  # decoded at threshold=-inf
    waveform: Optional[np.ndarray] = None  # window audio (ring-buffer snapshot) for the gate


@dataclass(frozen=True)
class PolicyDecision:
    accept: bool
    reason: str  # surfaced by --log-all-windows
    result: Optional[DecodeResult] = None  # authoritative override; None = window's own result


# Optional period-lifecycle sink (the CLI's `--log-periods` digest wires a
# printer here): called with (event, samples_seen, decision), where event is
# "open" (a press), "reopened" (a mid-period re-press), or "closed" (the
# flush -- the decision is the period's consolidated result).
PeriodEventCallback = Callable[[str, int, Optional[PolicyDecision]], None]


class AcceptancePolicy(Protocol):
    def observe(self, obs: WindowObservation) -> PolicyDecision: ...
    def reset(self) -> None: ...


class ThresholdPolicy:
    """The ONE shipped `AcceptancePolicy` implementation: accept iff
    `result.intent is not None and result.confidence >= threshold` --
    exactly `vcm.evaluate._accepted`'s body. Stateless (ignores window
    history); `observe`/`reset` are still stateful-shaped per the
    `AcceptancePolicy` protocol so a future history-based policy (e.g.
    multi-window smoothing, N-consecutive voting, active-region scoring)
    can be swapped in without changing the protocol.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def observe(self, obs: WindowObservation) -> PolicyDecision:
        result = obs.result
        if result.intent is None:
            return PolicyDecision(accept=False, reason="no_match: intent is None")
        if result.confidence < self.threshold:
            return PolicyDecision(
                accept=False,
                reason=(
                    f"confidence {result.confidence!r} below threshold "
                    f"{self.threshold!r}"
                ),
            )
        return PolicyDecision(
            accept=True,
            reason=(
                f"intent={result.intent!r} confidence={result.confidence!r} "
                f">= threshold {self.threshold!r}"
            ),
        )

    def reset(self) -> None:
        return None


class ModePeriodPolicy:
    """`AcceptancePolicy` that accepts at most once per listening period:
    collects every observation the gate's period covers, then decides at
    the period's end from the mode (most frequent `(intent, slots)`
    decode) across it, comparing the mode's mean confidence to the
    operating threshold. Winner ties break by count, then confidence
    sum, then first-seen order -- deterministic, no randomness. A press
    landing on the exact stride a period ends returns the flush decision
    for that observation AND seeds the new period with the same
    observation (SPEC observe algorithm: step 2 and step 3 both apply to
    that observation). An optional `on_period_event` sink is called with
    the period-lifecycle events ("open" at a press, "reopened" at a
    mid-period re-press, "closed" at the flush with the decision) -- the
    CLI's `--log-periods` digest wires a printer here; default `None`
    keeps behavior identical."""

    def __init__(
        self,
        threshold: float,
        *,
        gate: ListeningGate,
        period_s: float,
        on_period_event: Optional[PeriodEventCallback] = None,
    ) -> None:
        self.threshold = threshold
        self._gate = gate
        self._period_samples = int(period_s * SAMPLE_RATE)
        self._collected: list[WindowObservation] = []
        self._open_at: Optional[int] = None
        self._on_period_event = on_period_event

    def observe(self, obs: WindowObservation) -> PolicyDecision:
        state = self._gate.poll(obs.samples_seen, obs.waveform)

        if self._collected and obs.samples_seen >= self._open_at + self._period_samples:
            decision = self._flush()
            self._emit_period_event("closed", obs.samples_seen, decision)
            if state.is_open:
                # A press that landed on this exact flush stride: the same
                # observation also seeds the new period's collection.
                self._open_at = state.open_at_samples
                self._collected = [obs]
                self._emit_period_event("open", obs.samples_seen, None)
            return decision

        if state.is_open:
            if self._collected and state.open_at_samples != self._open_at:
                self._collected = []  # re-press: discard in-flight, restart
                self._emit_period_event("reopened", obs.samples_seen, None)
            elif not self._collected:
                self._emit_period_event("open", obs.samples_seen, None)
            self._open_at = state.open_at_samples
            self._collected.append(obs)
            return self._collecting_decision()

        return PolicyDecision(accept=False, reason="gate closed (not in a listening period)")

    def reset(self) -> None:
        self._collected = []
        self._open_at = None

    def _emit_period_event(
        self, event: str, samples_seen: int, decision: Optional[PolicyDecision]
    ) -> None:
        if self._on_period_event is not None:
            self._on_period_event(event, samples_seen, decision)

    def _flush(self) -> PolicyDecision:
        collected = self._collected
        grouped = self._group_by_decode_class(collected)
        winner_key = self._winner_key(grouped)
        _first, count, total_conf, latest = grouped[winner_key]
        self._collected = []

        if winner_key[0] is None:
            return PolicyDecision(
                accept=False,
                reason=f"mode_period: silence dominated the period (None {count}/{len(collected)})",
            )

        mean = total_conf / count
        result = replace(latest.result, confidence=mean)
        if mean >= self.threshold:
            return PolicyDecision(
                accept=True,
                reason=(
                    f"mode_period: intent={winner_key[0]!r} mean confidence {mean!r} "
                    f">= threshold {self.threshold!r} over {count} obs"
                ),
                result=result,
            )
        return PolicyDecision(
            accept=False,
            reason=(
                f"mode_period: intent={winner_key[0]!r} mean confidence {mean!r} "
                f"below threshold {self.threshold!r} over {count} obs"
            ),
            result=result,
        )

    def _collecting_decision(self) -> PolicyDecision:
        grouped = self._group_by_decode_class(self._collected)
        winner_key = self._winner_key(grouped)
        _first, count, _total, _latest = grouped[winner_key]
        n = len(self._collected)
        return PolicyDecision(
            accept=False,
            reason=f"collecting ({n} obs, running mode {winner_key[0]!r} {count}/{n})",
        )

    @staticmethod
    def _group_by_decode_class(collected):
        """Group observations by their decode class `(intent, sorted-slots)`;
        returns key -> (first_seen_index, count, confidence_sum, latest_obs)."""
        grouped = {}
        for index, obs in enumerate(collected):
            key = (obs.result.intent, tuple(sorted(obs.result.slots.items())))
            first, count, total_conf, latest = grouped.get(key, (index, 0, 0.0, None))
            grouped[key] = (first, count + 1, total_conf + obs.result.confidence, obs)
        return grouped

    @staticmethod
    def _winner_key(grouped):
        return min(grouped.items(), key=lambda item: (-item[1][1], -item[1][2], item[1][0]))[0]
