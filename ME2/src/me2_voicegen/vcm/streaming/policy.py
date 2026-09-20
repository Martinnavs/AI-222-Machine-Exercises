"""The acceptance-policy seam: decides whether one window's decoded
evidence is "real" before the debouncer ever gets a say. See
`ME2/docs/STREAMING-CONTRACT.md` for the fixed pipeline order this
composes into and the rationale for the evidence/emission-rate split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from me2_voicegen.vcm.decoder import DecodeResult


@dataclass(frozen=True)
class WindowObservation:
    window_index: int
    samples_seen: int
    result: DecodeResult  # decoded at threshold=-inf


@dataclass(frozen=True)
class PolicyDecision:
    accept: bool
    reason: str  # surfaced by --log-all-windows


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
