"""Wav/waveform -> intent pipeline: log-mel features (`common.features`) ->
trained CTC model (`vcm.model`) -> grammar-constrained decoder
(`vcm.decoder`) -> a final `{intent, slots, confidence}` result or an
explicit rejection.

Two modes:
    - whole-clip (`infer_waveform`): one full waveform in, one
      `DecodeResult` out. This is what `vcm.evaluate` uses over the
      manifest's `test` split.
    - sliding-window (`SlidingWindowPipeline`): a minimal streaming mode --
      1.5s window, 100ms stride, simple ring-buffer semantics -- with a
      confidence threshold plus a refractory/debounce cooldown period so a
      single spoken command held across several overlapping windows isn't
      reported as several separate triggers. Sample-counted (not
      wall-clock-timed) so its debounce behavior is deterministic and
      testable without real time passing; a caller streaming real
      microphone audio at `common.features.SAMPLE_RATE` gets real-time
      behavior for free since sample count IS elapsed time at a fixed
      sample rate.

Checkpoint loading here must match the `{model_state_dict, preset, config,
alphabet_size, license, ...}` shape `vcm.train.main` writes to
`checkpoint.pt` -- see that module's `torch.save` call.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.decoder import DecodeResult, decode
from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC

WINDOW_S = 1.5
STRIDE_S = 0.1
DEFAULT_REFRACTORY_S = 1.0

WINDOW_SAMPLES = int(round(WINDOW_S * SAMPLE_RATE))
STRIDE_SAMPLES = int(round(STRIDE_S * SAMPLE_RATE))


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MatchboxNetCTC, dict]:
    """Load a `vcm.train`-written checkpoint. Returns `(model, checkpoint)`
    -- `model` is in `eval()` mode on `device`; `checkpoint` is the raw
    dict (carries `license`, `preset`, `val_loss`, `epoch`, etc. for report
    headers)."""
    checkpoint = torch.load(path, map_location=device)
    config = MatchboxNetConfig(**checkpoint["config"])
    model = MatchboxNetCTC(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def logp_for_waveform(
    model: MatchboxNetCTC,
    feature_extractor: LogMelFeatureExtractor,
    waveform: torch.Tensor,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """waveform (samples,) -> (T, alphabet_size) log-posterior numpy array,
    the decoder's input contract (docs/VCM-CONTRACT.md section 7)."""
    features = feature_extractor(waveform).unsqueeze(0).to(device)
    logits = model(features)
    log_probs = logits.log_softmax(dim=-1)[0]
    return log_probs.cpu().numpy()


@torch.no_grad()
def infer_waveform(
    model: MatchboxNetCTC,
    feature_extractor: LogMelFeatureExtractor,
    waveform: torch.Tensor,
    grammar: Grammar,
    threshold: float,
    beam_width: int = 50,
    device: str | torch.device = "cpu",
) -> DecodeResult:
    """Whole-clip mode: one waveform -> one `DecodeResult`. Used by
    `vcm.evaluate` over the manifest's `test`/`val` splits."""
    logp = logp_for_waveform(model, feature_extractor, waveform, device=device)
    return decode(logp, grammar, threshold, beam_width=beam_width)


@dataclasses.dataclass
class TriggerEvent:
    """One non-suppressed accepted trigger from `SlidingWindowPipeline.feed`."""

    window_index: int
    samples_seen: int
    result: DecodeResult


class SlidingWindowPipeline:
    """Minimal streaming wrapper: feed it waveform chunks of any length,
    get back any newly-triggered (non-suppressed) `TriggerEvent`s.

    Ring-buffer semantics: an internal buffer holds at most the last
    `window_samples` samples fed so far (shorter than that only at
    startup, before the buffer has filled once). Every `stride_samples`
    of newly-fed audio, the pipeline runs one whole-clip-style inference
    over the current buffer contents. An accepted (non-`no_match`) result
    only becomes a `TriggerEvent` if the refractory/debounce cooldown has
    fully elapsed since the last trigger; otherwise it is suppressed
    (counted, not emitted) -- this is what stops one held command spanning
    several overlapping windows from firing repeatedly.
    """

    def __init__(
        self,
        model: MatchboxNetCTC,
        feature_extractor: LogMelFeatureExtractor,
        grammar: Grammar,
        threshold: float,
        beam_width: int = 50,
        window_s: float = WINDOW_S,
        stride_s: float = STRIDE_S,
        refractory_s: float = DEFAULT_REFRACTORY_S,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.grammar = grammar
        self.threshold = threshold
        self.beam_width = beam_width
        self.window_samples = int(round(window_s * SAMPLE_RATE))
        self.stride_samples = int(round(stride_s * SAMPLE_RATE))
        self.refractory_samples = int(round(refractory_s * SAMPLE_RATE))
        self.device = device

        self._buffer = torch.zeros(0)
        self._samples_since_last_window = 0
        self._cooldown_samples_remaining = 0
        self._window_index = 0
        self._samples_seen = 0
        self.suppressed_count = 0

    def reset(self) -> None:
        self._buffer = torch.zeros(0)
        self._samples_since_last_window = 0
        self._cooldown_samples_remaining = 0
        self._window_index = 0
        self._samples_seen = 0
        self.suppressed_count = 0

    def feed(self, chunk: torch.Tensor) -> list[TriggerEvent]:
        """Append `chunk` (1-D waveform tensor, any length > 0) to the ring
        buffer and evaluate once per `stride_samples` of newly-arrived
        audio. Returns the list of newly-emitted (non-suppressed)
        `TriggerEvent`s from this call (usually 0 or 1, but a large chunk
        spanning several strides can yield more than one)."""
        if chunk.numel() == 0:
            return []
        chunk = chunk.reshape(-1)

        events: list[TriggerEvent] = []
        self._buffer = torch.cat([self._buffer, chunk])
        if self._buffer.numel() > self.window_samples:
            self._buffer = self._buffer[-self.window_samples :]
        self._samples_since_last_window += chunk.numel()
        self._samples_seen += chunk.numel()

        while self._samples_since_last_window >= self.stride_samples:
            self._samples_since_last_window -= self.stride_samples
            self._window_index += 1

            if self._cooldown_samples_remaining > 0:
                self._cooldown_samples_remaining = max(
                    0, self._cooldown_samples_remaining - self.stride_samples
                )

            result = infer_waveform(
                self.model,
                self.feature_extractor,
                self._buffer,
                self.grammar,
                self.threshold,
                beam_width=self.beam_width,
                device=self.device,
            )

            if not result.no_match:
                if self._cooldown_samples_remaining <= 0:
                    events.append(
                        TriggerEvent(
                            window_index=self._window_index,
                            samples_seen=self._samples_seen,
                            result=result,
                        )
                    )
                    self._cooldown_samples_remaining = self.refractory_samples
                else:
                    self.suppressed_count += 1

        return events
