"""Fast, CPU-only tests for `vcm.streaming.config`. Config resolution logic
uses fixture JSON/dirs under `tmp_path`; the one real-artifact check (ONNX
vs. torch parity) lives in `test_vcm_streaming_backends.py` and is
`@pytest.mark.slow`.
"""

from __future__ import annotations

import json

import pytest

from me2_voicegen.vcm.streaming.config import (
    FALLBACK_THRESHOLD,
    MODEL_REGISTRY,
    POLICY_REGISTRY,
    StreamingConfig,
    resolve_model,
    resolve_policy,
    resolve_threshold,
)
from me2_voicegen.vcm.streaming.policy import ThresholdPolicy


def _make_run_dir(tmp_path, name="run", with_export=True, with_checkpoint=True, variants=("fp32",)):
    run_dir = tmp_path / name
    if with_export:
        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True)
        for variant in variants:
            (export_dir / f"vcm_model.{variant}.onnx").write_bytes(b"fake-onnx")
    if with_checkpoint:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    return run_dir


def _write_eval_report(run_dir, threshold=-0.1, grammar_label="OPTIONB_GRAMMAR"):
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "grammar_sections": [
            {"grammar": grammar_label, "chosen_operating_threshold": threshold}
        ]
    }
    (metadata_dir / "eval_report.json").write_text(json.dumps(report))
    return metadata_dir / "eval_report.json"


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_registry_name_onnx(monkeypatch, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "fake-registry-entry", run_dir)
    path = resolve_model("fake-registry-entry", backend="onnx", variant="fp32")
    assert path == run_dir / "export" / "vcm_model.fp32.onnx"


def test_resolve_model_registry_name_torch(monkeypatch, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "fake-registry-entry", run_dir)
    path = resolve_model("fake-registry-entry", backend="torch")
    assert path == run_dir / "checkpoints" / "checkpoint.pt"


def test_resolve_model_run_dir_form(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    assert resolve_model(str(run_dir), backend="onnx") == run_dir / "export" / "vcm_model.fp32.onnx"
    assert resolve_model(str(run_dir), backend="torch") == run_dir / "checkpoints" / "checkpoint.pt"


def test_resolve_model_direct_file_form(tmp_path):
    onnx_file = tmp_path / "somewhere" / "my_model.onnx"
    onnx_file.parent.mkdir(parents=True)
    onnx_file.write_bytes(b"fake")
    assert resolve_model(str(onnx_file), backend="onnx") == onnx_file


def test_resolve_model_onnx_variant_int8_selects_int8_artifact(tmp_path):
    run_dir = _make_run_dir(tmp_path, variants=("fp32", "int8"))
    path = resolve_model(str(run_dir), backend="onnx", variant="int8")
    assert path.name == "vcm_model.int8.onnx"


def test_resolve_model_nonexistent_is_actionable_system_exit(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(tmp_path / "does-not-exist"), backend="onnx")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "registry" in message.lower()


def test_resolve_model_missing_export_names_torch_alternative(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_export=False, with_checkpoint=True)
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(run_dir), backend="onnx")
    message = str(excinfo.value)
    assert "export" in message
    assert "--backend torch" in message


def test_resolve_model_missing_checkpoint_names_export_onnx_alternative(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_export=True, with_checkpoint=False)
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(run_dir), backend="torch")
    message = str(excinfo.value)
    assert "checkpoint.pt" in message
    assert "export_onnx" in message or "--backend onnx" in message


def test_resolve_model_unknown_backend_is_system_exit(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(SystemExit):
        resolve_model(str(run_dir), backend="tensorflow")


# ---------------------------------------------------------------------------
# resolve_threshold
# ---------------------------------------------------------------------------


def test_resolve_threshold_reads_real_report_value(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1)
    assert resolve_threshold(run_dir, override=None) == -0.1


def test_resolve_threshold_override_wins(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1)
    assert resolve_threshold(run_dir, override=-2.5) == -2.5


def test_resolve_threshold_warns_and_falls_back_when_report_missing(tmp_path, capsys):
    run_dir = tmp_path / "run-with-no-report"
    run_dir.mkdir()
    value = resolve_threshold(run_dir, override=None)
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_resolve_threshold_warns_and_falls_back_when_section_absent(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1, grammar_label="SOME_OTHER_GRAMMAR")
    value = resolve_threshold(run_dir, override=None, grammar_label="OPTIONB_GRAMMAR")
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_resolve_threshold_none_run_dir_warns_and_falls_back(capsys):
    value = resolve_threshold(None, override=None)
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_fallback_threshold_is_not_the_real_checked_in_operating_point():
    assert FALLBACK_THRESHOLD != -0.1


# ---------------------------------------------------------------------------
# resolve_policy
# ---------------------------------------------------------------------------


def test_resolve_policy_known_name_returns_configured_instance():
    policy = resolve_policy("threshold", threshold=-0.3)
    assert isinstance(policy, ThresholdPolicy)
    assert policy.threshold == -0.3


def test_resolve_policy_unknown_name_lists_valid_choices():
    with pytest.raises(SystemExit) as excinfo:
        resolve_policy("nonexistent-policy", threshold=0.0)
    message = str(excinfo.value)
    assert "nonexistent-policy" in message
    for choice in POLICY_REGISTRY:
        assert choice in message


# ---------------------------------------------------------------------------
# StreamingConfig.from_json / precedence merge
# ---------------------------------------------------------------------------


def test_from_json_overrides_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3.0, "backend": "torch"}))
    config = StreamingConfig.from_json(config_path)
    assert config.window_s == 3.0
    assert config.backend == "torch"
    # Untouched fields keep dataclass defaults.
    assert config.stride_s == StreamingConfig().stride_s


def test_from_json_unknown_field_is_actionable_system_exit(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"not_a_real_field": 1}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "not_a_real_field" in str(excinfo.value)


def test_from_json_missing_file_is_system_exit(tmp_path):
    with pytest.raises(SystemExit):
        StreamingConfig.from_json(tmp_path / "missing.json")


def test_from_json_invalid_json_is_system_exit(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json")
    with pytest.raises(SystemExit):
        StreamingConfig.from_json(config_path)


def test_merge_precedence_defaults_lt_json_lt_cli(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3.0, "stride_s": 0.5, "backend": "torch"}))

    # No JSON, no CLI -> pure defaults.
    defaults_only = StreamingConfig.merge()
    assert defaults_only.window_s == StreamingConfig().window_s

    # JSON overrides defaults.
    json_only = StreamingConfig.merge(json_path=config_path)
    assert json_only.window_s == 3.0
    assert json_only.stride_s == 0.5
    assert json_only.backend == "torch"
    # Field JSON didn't touch keeps the dataclass default.
    assert json_only.refractory_s == StreamingConfig().refractory_s

    # CLI overrides JSON (only for keys explicitly set, i.e. not None).
    merged = StreamingConfig.merge(
        json_path=config_path,
        cli_overrides={"window_s": 4.0, "backend": None, "device": "cuda"},
    )
    assert merged.window_s == 4.0  # CLI wins over JSON
    assert merged.stride_s == 0.5  # JSON wins over default (CLI didn't set it)
    assert merged.backend == "torch"  # CLI value was None -> not explicitly set, JSON wins
    assert merged.device == "cuda"  # CLI wins over default (JSON didn't set it)


def test_merge_unknown_cli_override_is_system_exit():
    with pytest.raises(SystemExit):
        StreamingConfig.merge(cli_overrides={"not_a_real_field": 1})


def test_from_json_rejects_onnx_variant_outside_cli_choices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"onnx_variant": "not-a-real-variant"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "onnx_variant" in str(excinfo.value)


def test_from_json_rejects_backend_outside_cli_choices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"backend": "tensorflow"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "backend" in str(excinfo.value)


def test_from_json_rejects_non_int_ort_threads_that_does_not_coerce(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": "not-a-number"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "ort_threads" in str(excinfo.value)


def test_from_json_coerces_numeric_string_ort_threads_to_int(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": "4"}))
    config = StreamingConfig.from_json(config_path)
    assert config.ort_threads == 4
    assert isinstance(config.ort_threads, int)


def test_from_json_rejects_bool_for_int_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": True}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "ort_threads" in str(excinfo.value)


def test_from_json_rejects_wrong_type_for_string_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"device": 123}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "device" in str(excinfo.value)


def test_from_json_rejects_wrong_type_for_bool_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"realtime": "true"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "realtime" in str(excinfo.value)


def test_from_json_accepts_int_for_float_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3}))
    config = StreamingConfig.from_json(config_path)
    assert config.window_s == 3.0
    assert isinstance(config.window_s, float)


def test_from_json_allows_null_for_nullable_fields(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"threshold": None, "mic_command": None, "listen_for": None}))
    config = StreamingConfig.from_json(config_path)
    assert config.threshold is None
    assert config.mic_command is None
    assert config.listen_for is None


def test_from_json_rejects_null_for_non_nullable_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": None}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "model" in str(excinfo.value)


def test_streaming_config_is_frozen():
    config = StreamingConfig()
    with pytest.raises(dataclasses_frozen_error_types()):
        config.window_s = 99.0


def dataclasses_frozen_error_types():
    import dataclasses

    return (dataclasses.FrozenInstanceError,)
