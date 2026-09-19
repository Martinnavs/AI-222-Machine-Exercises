"""Fast, CPU-only, checkpoint-free tests for `vcm.evaluate`.

Ticket 05's own Execution Log flagged this module as untested by any
dedicated unit test file (its correctness was only checked via the one
real evaluation run against the real checkpoint) -- this file is
ticket 08's independent gap-audit closing that gap. Covers the pure
threshold-sweep/confusion/false-accept arithmetic, `render_markdown`'s
grammar-coverage framing, and -- the two paths ticket 05 explicitly left
unexercised -- `main()`'s graceful skip when Task 07's slot-eval-set
manifest is absent, and its non-fatal catch-and-continue when that
manifest is present but evaluation against it fails.

Uses Task 02's shared `vcm_stub_model_factory`/`vcm_fake_manifest_factory`
fixtures throughout; no real checkpoint or GPU is ever touched here.
"""

from __future__ import annotations

import csv
import json

import pytest

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.evaluate import (
    RowResult,
    choose_operating_threshold,
    confusion_counts,
    false_accept_stats,
    main,
    render_markdown,
    sweep_thresholds,
)
from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR

CALL_IDS = alphabet.encode("call")


# ---------------------------------------------------------------------------
# --grammar selection (ticket 04, D11): GRAMMAR_REGISTRY / _intent_labels_for
# and main()'s CLI wiring for spec/toy/optionb.
# ---------------------------------------------------------------------------


def test_intent_labels_for_spec_and_toy_use_intent_phrases():
    from me2_voicegen.vcm.evaluate import _intent_labels_for
    from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

    assert _intent_labels_for("spec") == sorted(INTENT_PHRASES)
    assert _intent_labels_for("toy") == sorted(INTENT_PHRASES)


def test_intent_labels_for_optionb_uses_optionb_grammars_own_intents():
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
    from me2_voicegen.vcm.evaluate import _intent_labels_for
    from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

    labels = _intent_labels_for("optionb")
    assert labels == sorted({intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()})
    # Option B's label vocabulary is disjoint from vcm's -- this must not
    # silently fall back to INTENT_PHRASES.
    assert labels != sorted(INTENT_PHRASES)
    assert "BRIGHTNESS" in labels
    assert "DIM_UP" not in labels


# ---------------------------------------------------------------------------
# sweep_thresholds / choose_operating_threshold
# ---------------------------------------------------------------------------


def _row(bucket, label, intent, confidence):
    return RowResult(index=0, bucket=bucket, label=label, text="", intent=intent, confidence=confidence)


def test_sweep_thresholds_computes_target_and_false_accept_rates():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1),
        _row("target_commands", "CALL", "CALL", -0.4),
        _row("target_commands", "STOP", None, None),  # never reached any grammar terminal
        _row("babble", "n/a", "CALL", -0.2),
        _row("silence", "n/a", None, None),
    ]
    sweep = sweep_thresholds(results, thresholds=(0.0, -0.3, -1.0))

    by_threshold = {row["threshold"]: row for row in sweep}
    assert by_threshold[0.0]["target_accept_rate"] == pytest.approx(0.0)
    assert by_threshold[0.0]["false_accept_rate"] == pytest.approx(0.0)

    assert by_threshold[-0.3]["target_accept_rate"] == pytest.approx(1 / 3)
    assert by_threshold[-0.3]["false_accept_rate"] == pytest.approx(0.5)
    assert by_threshold[-0.3]["youden_j"] == pytest.approx(1 / 3 - 0.5)

    assert by_threshold[-1.0]["target_accept_rate"] == pytest.approx(2 / 3)
    assert by_threshold[-1.0]["false_accept_rate"] == pytest.approx(0.5)


def test_sweep_thresholds_empty_bucket_yields_zero_rate_not_zerodiv():
    results = [_row("target_commands", "CALL", "CALL", -0.1)]
    sweep = sweep_thresholds(results, thresholds=(0.0,))
    assert sweep[0]["false_accept_rate"] == 0.0
    assert sweep[0]["n_reject_probes"] == 0


def test_choose_operating_threshold_picks_max_youden_j():
    sweep = [
        {"threshold": 0.0, "youden_j": 0.2},
        {"threshold": -0.1, "youden_j": 0.9},
        {"threshold": -0.2, "youden_j": 0.5},
    ]
    chosen = choose_operating_threshold(sweep)
    assert chosen["threshold"] == -0.1


def test_choose_operating_threshold_ties_break_toward_least_permissive():
    # least-permissive == highest / least-negative threshold among ties.
    sweep = [
        {"threshold": -0.5, "youden_j": 0.7},
        {"threshold": -0.1, "youden_j": 0.7},
        {"threshold": -0.9, "youden_j": 0.7},
    ]
    chosen = choose_operating_threshold(sweep)
    assert chosen["threshold"] == -0.1


# ---------------------------------------------------------------------------
# confusion_counts / false_accept_stats
# ---------------------------------------------------------------------------


def test_confusion_counts_only_covers_target_commands_bucket():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1),
        _row("target_commands", "CALL", "STOP", -0.1),  # confusion
        _row("target_commands", "STOP", None, None),  # rejected
        _row("babble", "n/a", "CALL", -0.1),  # excluded from confusion table
    ]
    counts = confusion_counts(results, threshold=-0.5)
    assert counts["CALL"] == {"CALL": 1, "STOP": 1}
    assert counts["STOP"] == {"REJECTED": 1}
    assert "n/a" not in counts


def test_confusion_counts_below_threshold_counts_as_rejected():
    results = [_row("target_commands", "CALL", "CALL", -5.0)]
    counts = confusion_counts(results, threshold=-0.1)
    assert counts["CALL"] == {"REJECTED": 1}


def test_false_accept_stats_empty_bucket_reports_rate_none_not_zero():
    stats = false_accept_stats([], threshold=0.0, bucket="babble")
    assert stats == {"n": 0, "false_accepts": 0, "rate": None}


def test_false_accept_stats_counts_accepts_at_threshold():
    results = [
        _row("babble", "n/a", "CALL", -0.1),
        _row("babble", "n/a", "CALL", -5.0),
        _row("babble", "n/a", None, None),
    ]
    stats = false_accept_stats(results, threshold=-1.0, bucket="babble")
    assert stats == {"n": 3, "false_accepts": 1, "rate": pytest.approx(1 / 3)}


# ---------------------------------------------------------------------------
# render_markdown: grammar-coverage framing must actually appear, not just
# render without crashing.
# ---------------------------------------------------------------------------


def _minimal_grammar_section(label: str) -> dict:
    sweep = sweep_thresholds(
        [_row("target_commands", "CALL", "CALL", -0.1), _row("babble", "n/a", None, None)],
        thresholds=(0.0, -0.1),
    )
    chosen = choose_operating_threshold(sweep)
    return {
        "grammar": label,
        "threshold_sweep_on_val": sweep,
        "chosen_operating_threshold": chosen["threshold"],
        "chosen_operating_point_val_stats": chosen,
        "test_split": {
            "n_target_commands": 1,
            "n_accepted": 1,
            "n_exact_correct": 1,
            "accept_rate": 1.0,
            "exact_accuracy": 1.0,
            "per_intent_confusion": {"CALL": {"CALL": 1}},
            "false_accept_rate_babble": {"n": 0, "false_accepts": 0, "rate": None},
            "false_accept_rate_silence": {"n": 0, "false_accepts": 0, "rate": None},
        },
    }


def test_render_markdown_includes_spec_grammar_coverage_note():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("SPEC_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "Grammar-coverage note" in md
    assert "12 of the 20" in md
    assert "not because the model failed" in md


def test_render_markdown_toy_grammar_section_has_no_coverage_note():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "Grammar-coverage note" not in md


def test_render_markdown_reports_slot_eval_skip_reason():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": "out/vcm/slot_eval/manifest.csv not found",
    }
    md = render_markdown(report)
    assert "Skipped: out/vcm/slot_eval/manifest.csv not found" in md


# ---------------------------------------------------------------------------
# evaluate_slot_eval_set: the graceful-skip contract from Task 05's own
# Execution Log ("present but malformed" path was unexercised by any
# automated test) -- exercised for real here, CPU-only, stub model.
# ---------------------------------------------------------------------------


def _write_slot_eval_manifest(tmp_path, wav_factory, *, malformed=False):
    slot_dir = tmp_path / "slot_eval"
    slot_dir.mkdir()
    manifest_path = slot_dir / "manifest.csv"
    if malformed:
        # References a wav file that was never written -- forces the real
        # torchaudio.load() call inside evaluate_slot_eval_set to raise,
        # which is exactly the "present but evaluation failed" path
        # main()'s broad except is there to catch.
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "path", "label", "sample_rate"])
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "missing.wav",
                    "path": "audio/missing.wav",
                    "label": "CALL",
                    "sample_rate": "16000",
                }
            )
        return manifest_path

    wav_factory(slot_dir / "audio" / "call_mom.wav", duration_s=0.2)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "path", "label", "sample_rate"])
        writer.writeheader()
        writer.writerow(
            {
                "filename": "call_mom.wav",
                "path": "audio/call_mom.wav",
                "label": "CALL",
                "sample_rate": "16000",
            }
        )
    return manifest_path


def test_evaluate_slot_eval_set_scores_accept_and_intent_correct(
    tmp_path, vcm_wav_factory, vcm_stub_model_factory
):
    from me2_voicegen.vcm.evaluate import evaluate_slot_eval_set
    from me2_voicegen.common.features import LogMelFeatureExtractor

    manifest_path = _write_slot_eval_manifest(tmp_path, vcm_wav_factory)
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()

    section = evaluate_slot_eval_set(
        model, fe, manifest_path, TOY_GRAMMAR, "TOY_GRAMMAR", threshold=-50.0, beam_width=50, device="cpu"
    )

    assert section["n_clips"] == 1
    assert section["n_accepted"] == 1
    assert section["n_intent_correct"] == 1
    assert "KNOWN slot-vocabulary training gap" in section["framing"]
    assert section["rows"][0]["predicted_intent"] == "CALL"


# ---------------------------------------------------------------------------
# main(): the two slot-eval-set control-flow paths ticket 05 flagged as
# untested -- graceful skip when absent, non-fatal catch when malformed.
# Monkeypatches load_checkpoint so no real checkpoint.pt is ever touched.
# ---------------------------------------------------------------------------


def _build_main_manifest(vcm_fake_manifest_factory):
    specs = []
    for split in ("val", "test"):
        specs.append(
            {
                "bucket": "target_commands",
                "source_dataset": "sanitized_clean",
                "label": "CALL",
                "split": split,
            }
        )
        specs.append(
            {
                "bucket": "babble",
                "source_dataset": "common_voice_negative",
                "label": "unknown",
                "split": split,
                "transcript": "some other speech",
            }
        )
        specs.append(
            {
                "bucket": "silence",
                "source_dataset": "background_noise",
                "label": "unknown",
                "split": split,
            }
        )
    return vcm_fake_manifest_factory(specs)


def _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory):
    import me2_voicegen.vcm.evaluate as evaluate_mod

    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    checkpoint_meta = {"preset": "default", "epoch": 1, "val_loss": 1.0, "license": "CC-BY-NC-SA-4.0"}
    monkeypatch.setattr(evaluate_mod, "load_checkpoint", lambda path, device: (model, checkpoint_meta))


def test_main_gracefully_skips_absent_slot_eval_manifest(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    missing_slot_manifest = tmp_path / "does_not_exist" / "manifest.csv"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(missing_slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_sections"] is None
    assert "not found" in report["slot_eval_skipped_reason"]
    assert (out_dir / "metadata" / "eval_report.md").exists()
    assert "Skipped:" in (out_dir / "metadata" / "eval_report.md").read_text()


def test_main_catches_and_continues_on_malformed_slot_eval_manifest(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_wav_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)
    slot_manifest = _write_slot_eval_manifest(tmp_path, vcm_wav_factory, malformed=True)

    out_dir = tmp_path / "eval_out"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_sections"] is None
    assert "present but evaluation failed" in report["slot_eval_skipped_reason"]


def test_main_runs_slot_eval_section_when_manifest_present_and_valid(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_wav_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)
    slot_manifest = _write_slot_eval_manifest(tmp_path, vcm_wav_factory, malformed=False)

    out_dir = tmp_path / "eval_out"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_skipped_reason"] is None
    assert report["slot_eval_sections"] is not None
    assert len(report["slot_eval_sections"]) == 2
    for section in report["slot_eval_sections"]:
        assert section["n_clips"] == 1


def test_main_default_grammar_is_spec_toy(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(tmp_path / "does_not_exist" / "manifest.csv"),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    labels = [s["grammar"] for s in report["grammar_sections"]]
    assert labels == ["SPEC_GRAMMAR", "TOY_GRAMMAR"]


def _build_optionb_main_manifest(vcm_fake_manifest_factory):
    specs = []
    for split in ("val", "test"):
        specs.append(
            {
                "bucket": "target_commands",
                "source_dataset": "optionb",
                "label": "CALL",
                "split": split,
                "transcript": "Call",
            }
        )
        specs.append(
            {
                "bucket": "babble",
                "source_dataset": "common_voice_negative",
                "label": "unknown",
                "split": split,
                "transcript": "some other speech",
            }
        )
        specs.append(
            {
                "bucket": "silence",
                "source_dataset": "background_noise",
                "label": "unknown",
                "split": split,
            }
        )
    return vcm_fake_manifest_factory(specs)


def test_main_grammar_optionb_selects_optionb_grammar_and_its_own_intent_labels(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_optionb_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(tmp_path / "does_not_exist" / "manifest.csv"),
            "--grammar",
            "optionb",
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert len(report["grammar_sections"]) == 1
    section = report["grammar_sections"][0]
    assert section["grammar"] == "OPTIONB_GRAMMAR"
    assert "DIM_UP" not in section["intent_labels"]
    assert "BRIGHTNESS" in section["intent_labels"]
    # Non-degenerate sweep (D5/D11): babble+silence reject probes present.
    ts = section["test_split"]
    assert ts["n_target_commands"] == 1
    assert ts["false_accept_rate_babble"]["n"] == 1
    assert ts["false_accept_rate_silence"]["n"] == 1


def test_main_rejects_unknown_grammar_key(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    with pytest.raises(SystemExit):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--checkpoint",
                "unused.pt",
                "--out-dir",
                str(tmp_path / "eval_out"),
                "--grammar",
                "bogus",
            ]
        )
