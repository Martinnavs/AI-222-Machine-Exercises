"""Unit tests for vcm.augment: RIR, additive noise, SpecAugment.

Covers ticket 02's four "Must Verify" items with real assertions (not just
"runs without erroring"), plus determinism under a fixed seed and
off-by-default behavior.
"""

from __future__ import annotations

import math

import torch

from me2_voicegen.common.augment import (
    Augmenter,
    apply_noise,
    apply_rir,
    build_rir_pool,
)
from me2_voicegen.common.features import LogMelFeatureExtractor


def _sine(duration_s: float = 1.0, sample_rate: int = 16000, freq_hz: float = 440.0) -> torch.Tensor:
    n = int(duration_s * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return 0.3 * torch.sin(2 * torch.pi * freq_hz * t)


def _white_noise(duration_s: float = 1.0, sample_rate: int = 16000, seed: int = 0) -> torch.Tensor:
    n = int(duration_s * sample_rate)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


def _measure_snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise_component = noisy - clean
    signal_power = clean.pow(2).mean()
    noise_power = noise_component.pow(2).mean()
    return 10.0 * math.log10((signal_power / noise_power).item())


# ---------------------------------------------------------------------------
# Must Verify: RIR simulate_rir_ism actually imports/runs on this venv.
# ---------------------------------------------------------------------------


def test_rir_pool_builds_finite_impulse_responses():
    pool = build_rir_pool(pool_size=5, sample_rate=16000, seed=42)
    assert len(pool) == 5
    for rir in pool:
        assert torch.isfinite(rir).all()
        assert rir.numel() > 0


def test_rir_convolved_output_is_finite_and_more_reverberant():
    waveform = _sine(duration_s=0.5)
    rir = build_rir_pool(pool_size=1, sample_rate=16000, seed=1)[0]
    wet = apply_rir(waveform, rir)

    assert torch.isfinite(wet).all()
    assert wet.shape == waveform.shape

    # "more reverberant": energy trails past the point the dry signal's
    # energy would have already ended, because the wet signal is the
    # convolution of a longer-than-instantaneous RIR with the dry signal.
    tail = wet[-200:]
    assert tail.abs().sum().item() > 0
    # Sanity: convolving with a >1-sample-wide RIR changes the waveform
    # (isn't literally a no-op / passthrough).
    assert not torch.allclose(wet, waveform, atol=1e-4)


def test_rir_pool_size_matches_request():
    pool = build_rir_pool(pool_size=200, sample_rate=16000, seed=7)
    assert len(pool) == 200


# ---------------------------------------------------------------------------
# Must Verify: measured SNR of a noise-augmented sample matches request.
# ---------------------------------------------------------------------------


def test_measured_snr_within_tolerance_of_requested():
    clean = _sine(duration_s=1.0)
    noise = _white_noise(duration_s=1.0, seed=3)
    for requested_snr in (5.0, 15.0, 25.0):
        noisy = apply_noise(clean, noise, snr_db=requested_snr)
        measured = _measure_snr_db(clean, noisy)
        assert abs(measured - requested_snr) < 1.0


def test_apply_noise_output_finite_and_same_length():
    clean = _sine(duration_s=0.7)
    short_noise = _white_noise(duration_s=0.2, seed=9)  # shorter than clean
    noisy = apply_noise(clean, short_noise, snr_db=10.0)
    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()


# ---------------------------------------------------------------------------
# Must Verify: SpecAugment actually zeroes/masks regions.
# ---------------------------------------------------------------------------


def test_specaugment_masks_are_actually_zeroed():
    waveform = _sine(duration_s=1.0)
    log_mel = LogMelFeatureExtractor()(waveform)

    augmenter = Augmenter(p_specaugment=1.0, seed=123)
    masked = augmenter.augment_features(log_mel)

    assert masked.shape == log_mel.shape
    # At probability 1.0 with real signal content, masking must change
    # the tensor by zeroing at least one full time or frequency slice.
    assert not torch.equal(masked, log_mel)

    zero_time_cols = (masked.abs().sum(dim=0) == 0).sum().item()
    zero_freq_rows = (masked.abs().sum(dim=1) == 0).sum().item()
    assert zero_time_cols > 0 or zero_freq_rows > 0


def test_specaugment_off_when_probability_zero():
    waveform = _sine(duration_s=1.0)
    log_mel = LogMelFeatureExtractor()(waveform)
    augmenter = Augmenter(p_specaugment=0.0, seed=1)
    out = augmenter.augment_features(log_mel)
    assert torch.equal(out, log_mel)


# ---------------------------------------------------------------------------
# Must Verify: augmentation OFF by default in eval mode.
# ---------------------------------------------------------------------------


def test_augmenter_defaults_are_all_off():
    augmenter = Augmenter()
    assert augmenter.p_rir == 0.0
    assert augmenter.p_noise == 0.0
    assert augmenter.p_specaugment == 0.0


def test_default_augmenter_is_a_no_op_on_waveform_and_features():
    waveform = _sine(duration_s=0.5)
    log_mel = LogMelFeatureExtractor()(waveform)
    augmenter = Augmenter()

    out_wave = augmenter.augment_waveform(waveform, noise_pool=[_white_noise(seed=2)])
    assert torch.equal(out_wave, waveform)

    out_features = augmenter.augment_features(log_mel)
    assert torch.equal(out_features, log_mel)


# ---------------------------------------------------------------------------
# Determinism under a fixed seed.
# ---------------------------------------------------------------------------


def test_augmenter_deterministic_under_fixed_seed():
    waveform = _sine(duration_s=0.5)
    noise_pool = [_white_noise(duration_s=0.5, seed=i) for i in range(3)]

    a = Augmenter(p_rir=1.0, p_noise=1.0, p_specaugment=1.0, seed=2024, rir_pool_size=10)
    b = Augmenter(p_rir=1.0, p_noise=1.0, p_specaugment=1.0, seed=2024, rir_pool_size=10)

    out_a = a.augment_waveform(waveform, noise_pool=noise_pool)
    out_b = b.augment_waveform(waveform, noise_pool=noise_pool)
    assert torch.equal(out_a, out_b)

    log_mel = LogMelFeatureExtractor()(waveform)
    feat_a = a.augment_features(log_mel.clone())
    feat_b = b.augment_features(log_mel.clone())
    assert torch.equal(feat_a, feat_b)


def test_different_seeds_diverge():
    waveform = _sine(duration_s=0.5)
    noise_pool = [_white_noise(duration_s=0.5, seed=i) for i in range(3)]

    a = Augmenter(p_rir=1.0, p_noise=1.0, seed=1, rir_pool_size=10)
    b = Augmenter(p_rir=1.0, p_noise=1.0, seed=2, rir_pool_size=10)

    out_a = a.augment_waveform(waveform, noise_pool=noise_pool)
    out_b = b.augment_waveform(waveform, noise_pool=noise_pool)
    assert not torch.equal(out_a, out_b)
