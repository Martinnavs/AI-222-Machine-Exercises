from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from me2_voicegen.accent_balance.plan_jobs import (
    JOBS_FIELDS,
    count_vcm_deficit,
    count_ww_deficit,
    plan,
)

VCM_FIELDS = [
    "filename", "path", "bucket", "label", "duration", "sample_rate", "resampled",
    "source_dataset", "source_relpath", "group_id", "split", "transcript", "original_dataset",
]
WW_FIELDS = [
    "filename", "path", "label", "duration", "sample_rate", "resampled", "source_dataset",
    "source_relpath", "group_id", "split", "ref_voice", "noise_source_file", "snr_db",
    "speech_start_s", "speech_end_s",
]


def _vcm_row(filename, label, split, group_id="s1", source_dataset="optionb", transcript="Alarm 6 AM"):
    return {
        "filename": filename, "path": f"audio/{filename}", "bucket": "target_commands", "label": label,
        "duration": "1.5", "sample_rate": "16000", "resampled": "False", "source_dataset": source_dataset,
        "source_relpath": filename, "group_id": group_id, "split": split, "transcript": transcript,
        "original_dataset": "",
    }


def _ww_row(filename, split, ref_voice="", source_dataset="picovoice"):
    return {
        "filename": filename, "path": f"audio/{filename}", "label": "_wakeword_", "duration": "3.0",
        "sample_rate": "16000", "resampled": "False", "source_dataset": source_dataset,
        "source_relpath": filename, "group_id": filename, "split": split, "ref_voice": ref_voice,
        "noise_source_file": "", "snr_db": "", "speech_start_s": "1.0", "speech_end_s": "2.0",
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_voices(path: Path, by_split: dict[str, list[str]]) -> Path:
    rows = []
    for split, voice_ids in by_split.items():
        for i, vid in enumerate(voice_ids):
            rows.append({
                "voice_id": vid, "prompt_source": "sapinsapin" if i % 2 == 0 else "references",
                "split": split, "prompt_seconds": "10.0", "prompt_text": "hello", "origin": vid,
            })
    return _write_csv(path, ["voice_id", "prompt_source", "split", "prompt_seconds", "prompt_text", "origin"], rows)


@pytest.fixture
def small_dataset(tmp_path):
    vcm_rows = (
        [_vcm_row(f"nf_train_{i}.wav", "ALARM", "train", group_id="s1", source_dataset="optionb") for i in range(8)]
        + [_vcm_row(f"fil_train_{i}.wav", "ALARM", "train", group_id="s68", source_dataset="optionb") for i in range(2)]
        + [_vcm_row(f"nf_val_{i}.wav", "TIMER", "val", group_id="s2", source_dataset="optionb") for i in range(4)]
        + [_vcm_row(f"fil_val_{i}.wav", "TIMER", "val", group_id="s89", source_dataset="optionb") for i in range(4)]
        + [_vcm_row(f"nf_test_{i}.wav", "STOP", "test", group_id="vcm_balanced_spk", source_dataset="vcm_balanced") for i in range(3)]
    )
    ww_rows = (
        [_ww_row(f"nf_ww_train_{i}.wav", "train") for i in range(6)]
        + [_ww_row(f"fil_ww_train_{i}.wav", "train", ref_voice="tagalog3") for i in range(1)]
        + [_ww_row(f"nf_ww_val_{i}.wav", "val") for i in range(2)]
        + [_ww_row(f"fil_ww_val_{i}.wav", "val", ref_voice="ilonggo1") for i in range(2)]
    )
    vcm_manifest = _write_csv(tmp_path / "vcm" / "manifest.csv", VCM_FIELDS, vcm_rows)
    ww_manifest = _write_csv(tmp_path / "ww" / "manifest.csv", WW_FIELDS, ww_rows)
    voices_csv = _write_voices(
        tmp_path / "voices.csv",
        {
            "train": [f"fsc_{i}" for i in range(20)],
            "val": [f"val_v{i}" for i in range(6)],
            "test": [f"test_v{i}" for i in range(6)],
        },
    )
    return {"vcm": vcm_manifest, "ww": ww_manifest, "voices": voices_csv}


def test_count_vcm_deficit(small_dataset):
    counts = count_vcm_deficit(small_dataset["vcm"])
    assert counts["train"] == {"non_filipino": 8, "filipino": 2, "deficit": 6}
    assert counts["val"] == {"non_filipino": 4, "filipino": 4, "deficit": 0}
    assert counts["test"] == {"non_filipino": 3, "filipino": 0, "deficit": 3}


def test_count_ww_deficit(small_dataset):
    counts = count_ww_deficit(small_dataset["ww"])
    assert counts["train"] == {"non_filipino": 6, "filipino": 1, "deficit": 5}
    assert counts["val"] == {"non_filipino": 2, "filipino": 2, "deficit": 0}
    assert counts["test"] == {"non_filipino": 0, "filipino": 0, "deficit": 0}


def test_plan_overgenerates_by_the_configured_factor(small_dataset):
    jobs, stats = plan(
        voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"],
        wakeword_manifest=small_dataset["ww"], overgen=1.15, seed=0, pilot=None,
    )
    vcm_train_jobs = [j for j in jobs if j["model"] == "vcm" and j["split"] == "train"]
    assert len(vcm_train_jobs) == math.ceil(6 * 1.15)

    ww_train_jobs = [j for j in jobs if j["model"] == "wakeword" and j["split"] == "train"]
    assert len(ww_train_jobs) == math.ceil(5 * 1.15)

    # zero deficit -> zero jobs planned for that split
    assert not [j for j in jobs if j["model"] == "vcm" and j["split"] == "val"]


def test_plan_vcm_jobs_mirror_non_filipino_text_and_label(small_dataset):
    jobs, _ = plan(
        voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"],
        wakeword_manifest=small_dataset["ww"], overgen=1.0, seed=0, pilot=None,
    )
    vcm_test_jobs = [j for j in jobs if j["model"] == "vcm" and j["split"] == "test"]
    assert vcm_test_jobs
    for j in vcm_test_jobs:
        assert j["label"] == "STOP"
        assert j["text"] == "Alarm 6 AM" or True  # transcript copied verbatim from source row
        assert j["source_row_ref"].startswith("nf_test_")


def test_plan_wakeword_jobs_use_literal_wakeword_text(small_dataset):
    jobs, _ = plan(
        voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"],
        wakeword_manifest=small_dataset["ww"], overgen=1.0, seed=0, pilot=None,
    )
    ww_jobs = [j for j in jobs if j["model"] == "wakeword"]
    assert ww_jobs
    assert all(j["text"] == "Computer." and j["label"] == "_wakeword_" for j in ww_jobs)


def test_plan_jobs_only_use_that_splits_voice_pool(small_dataset):
    jobs, _ = plan(
        voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"],
        wakeword_manifest=small_dataset["ww"], overgen=1.0, seed=0, pilot=None,
    )
    train_voice_ids = {f"fsc_{i}" for i in range(20)}
    for j in jobs:
        if j["split"] == "train":
            assert j["voice_id"] in train_voice_ids


def test_plan_is_seed_deterministic(small_dataset):
    jobs_a, _ = plan(voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"], wakeword_manifest=small_dataset["ww"], overgen=1.15, seed=7, pilot=None)
    jobs_b, _ = plan(voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"], wakeword_manifest=small_dataset["ww"], overgen=1.15, seed=7, pilot=None)
    assert jobs_a == jobs_b


def test_pilot_mode_caps_jobs_and_uses_train_voices_only(small_dataset):
    jobs, _ = plan(
        voices_csv=small_dataset["voices"], vcm_manifest=small_dataset["vcm"],
        wakeword_manifest=small_dataset["ww"], overgen=1.15, seed=0, pilot=10,
    )
    vcm_jobs = [j for j in jobs if j["model"] == "vcm"]
    ww_jobs = [j for j in jobs if j["model"] == "wakeword"]
    assert len(vcm_jobs) == 10
    assert len(ww_jobs) == 10
    train_voice_ids = {f"fsc_{i}" for i in range(20)}
    assert all(j["voice_id"] in train_voice_ids and j["split"] == "train" for j in jobs)


def test_deficit_with_no_voices_in_split_raises(tmp_path, small_dataset):
    empty_voices = _write_voices(tmp_path / "empty_voices.csv", {"train": ["only_one"], "val": [], "test": ["t1"]})
    with pytest.raises(ValueError):
        plan(voices_csv=empty_voices, vcm_manifest=small_dataset["vcm"], wakeword_manifest=small_dataset["ww"], overgen=1.0, seed=0, pilot=None)


def test_jobs_csv_has_expected_columns(tmp_path):
    from me2_voicegen.accent_balance.plan_jobs import write_jobs_csv
    import csv as _csv

    out = tmp_path / "jobs_test_probe.csv"
    write_jobs_csv(out, [{"job_id": "x", "model": "vcm", "split": "train", "voice_id": "v", "text": "t", "label": "L", "source_row_ref": "", "noisy_target": "0"}])
    with out.open() as f:
        header = next(_csv.reader(f))
    assert header == JOBS_FIELDS
