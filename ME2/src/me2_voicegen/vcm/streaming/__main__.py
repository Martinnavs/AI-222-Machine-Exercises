"""`python -m me2_voicegen.vcm.streaming [flags]`: the spoken-command
streaming CLI. Wires `StreamingConfig` (registry/threshold resolution) and
`StreamingRunner` (loop modes) together; see
`ME2/docs/STREAMING-CONTRACT.md` for the shapes this consumes.

No argument opens the live microphone by default. The microphone is only
ever opened inside `main()`, immediately before the runner loop starts --
importing this module, building a parser, or resolving a `StreamingConfig`
never touches a device.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.streaming.backends import InferenceBackend, OnnxBackend, TorchBackend
from me2_voicegen.vcm.streaming.config import (
    StreamingConfig,
    _candidate_run_dir,
    resolve_grammar,
    resolve_model,
    resolve_policy,
    resolve_threshold,
)
from me2_voicegen.vcm.streaming.runner import StreamingRunner
from me2_voicegen.vcm.streaming.sources import (
    DEFAULT_MIC_COMMAND,
    MicrophoneUnavailableError,
    WavFileSource,
    open_microphone_source,
)

LICENSE_NOTE = (
    "Checkpoint trained on out/conversions/v2/test_set/, which includes "
    "background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per "
    "docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data "
    "inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on "
    "redistribution."
)

_CONFIG_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(StreamingConfig))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="StreamingConfig JSON file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["onnx", "torch"])
    parser.add_argument(
        "--onnx-variant", dest="onnx_variant", type=str, default=None, choices=["fp32", "int8"]
    )
    parser.add_argument("--ort-threads", dest="ort_threads", type=int, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--grammar", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window-s", dest="window_s", type=float, default=None)
    parser.add_argument("--stride-s", dest="stride_s", type=float, default=None)
    parser.add_argument("--refractory-s", dest="refractory_s", type=float, default=None)
    parser.add_argument("--beam-width", dest="beam_width", type=int, default=None)
    parser.add_argument("--block-s", dest="block_s", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="'mic' (default) for the live microphone, or a path to a wav file to replay",
    )
    parser.add_argument("--mic-command", dest="mic_command", type=str, default=None)
    parser.add_argument(
        "--listen-for",
        dest="listen_for",
        type=float,
        default=None,
        help="stop after this many seconds of captured audio (realtime mode only)",
    )
    parser.add_argument(
        "--log-all-windows",
        dest="log_all_windows",
        action="store_true",
        default=None,
        help="emit a JSONL 'window' record for every evaluated window, not just triggers",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    return {name: getattr(args, name) for name in _CONFIG_FIELD_NAMES if hasattr(args, name)}


def _load_checkpoint_meta(run_dir: Optional[Path]) -> dict:
    if run_dir is None:
        return {}
    report_path = Path(run_dir) / "metadata" / "eval_report.json"
    try:
        report = json.loads(report_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return report.get("checkpoint_meta") or {}


def _print_banner(
    cfg: StreamingConfig,
    model_path: Path,
    run_dir: Optional[Path],
    threshold: float,
    grammar_label: str,
    out: object = sys.stderr,
) -> None:
    meta = _load_checkpoint_meta(run_dir)
    preset = meta.get("preset", cfg.model)
    license_note = meta.get("license", LICENSE_NOTE)
    lines = [
        "me2_voicegen streaming spoken-command runner",
        f"  checkpoint/run dir: {run_dir if run_dir is not None else model_path}",
        f"  preset: {preset}",
        f"  backend: {cfg.backend} (onnx variant: {cfg.onnx_variant})",
        f"  grammar: {cfg.grammar} ({grammar_label})",
        f"  window_s={cfg.window_s} stride_s={cfg.stride_s} "
        f"refractory_s={cfg.refractory_s} beam_width={cfg.beam_width}",
        f"  resolved threshold: {threshold}",
        f"  license: {license_note}",
    ]
    for line in lines:
        print(line, file=out)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = StreamingConfig.merge(json_path=args.config, cli_overrides=_cli_overrides(args))

    model_path = resolve_model(cfg.model, cfg.backend, cfg.onnx_variant)
    run_dir = _candidate_run_dir(cfg.model)
    grammar, grammar_label = resolve_grammar(cfg.grammar)
    threshold = resolve_threshold(run_dir, override=cfg.threshold, grammar_label=grammar_label)
    policy = resolve_policy(cfg.policy, threshold)

    backend: InferenceBackend
    if cfg.backend == "onnx":
        backend = OnnxBackend(model_path, ort_threads=cfg.ort_threads)
    else:
        backend = TorchBackend(model_path, device=cfg.device)

    _print_banner(cfg, model_path, run_dir, threshold, grammar_label)

    block_samples = max(1, int(round(cfg.block_s * SAMPLE_RATE)))

    if cfg.source == "mic":
        try:
            source = open_microphone_source(cfg.mic_command or DEFAULT_MIC_COMMAND, block_samples)
        except MicrophoneUnavailableError as exc:
            raise SystemExit(str(exc)) from None
    else:
        wav_path = Path(cfg.source)
        if not wav_path.exists():
            raise SystemExit(
                f"--source {cfg.source!r} is not 'mic' and does not exist as a wav "
                f"file. Pass --source <wav-path> for file replay, or 'mic' (with an "
                f"optional --mic-command) for the live microphone."
            )
        source = WavFileSource(wav_path, block_samples, realtime=False)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=grammar,
        policy=policy,
        window_s=cfg.window_s,
        stride_s=cfg.stride_s,
        refractory_s=cfg.refractory_s,
        beam_width=cfg.beam_width,
        listen_for_s=cfg.listen_for,
        log_all_windows=cfg.log_all_windows,
    )
    runner.run()


if __name__ == "__main__":
    main()
