"""Export a trained `vcm.model` checkpoint (Task 04) to ONNX, with a
dynamic time axis, and quantize it to static INT8.

Per docs/VCM-CONTRACT.md section 5, this module's dummy/calibration inputs
are always `(B, 40, T)` log-mel features built from `vcm.features` -- the
same front-end training/eval use -- so no train/infer skew is introduced
here.

Static (not dynamic) quantization is used deliberately: `onnxruntime`'s
`quantize_dynamic` is known to often leave `Conv` layers in fp32, which
would defeat the purpose for this 1D-CNN model (see ticket
.scratch/vcm-toy/tickets/06-onnx-export-benchmark.md's "Unknowns"). Static
quantization needs a calibration data reader, built here from a small
sample of the real `val` split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC

WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
)


def load_checkpoint(checkpoint_path: str | Path) -> tuple[MatchboxNetCTC, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = MatchboxNetConfig(**ckpt["config"])
    model = MatchboxNetCTC(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def window_n_frames(window_samples: int = WINDOW_SAMPLES) -> int:
    """Number of log-mel frames `vcm.features` produces for a
    `window_samples`-sample waveform (151 for the default 1.5s/24000-sample
    window at the real 16kHz/480/160 front-end parameters)."""
    extractor = LogMelFeatureExtractor()
    return int(extractor(torch.zeros(window_samples)).shape[-1])


def dummy_features(n_frames: int | None = None, batch: int = 1) -> torch.Tensor:
    """`(batch, 40, n_frames)` log-mel-shaped tensor for tracing/benchmarking.

    Content is arbitrary (fixed seed for reproducibility) -- only the shape
    and dtype need to match a real `(B, 40, T)` log-mel batch (docs/
    VCM-CONTRACT.md section 5) for export/benchmark purposes.
    """
    if n_frames is None:
        n_frames = window_n_frames()
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, 40, n_frames, generator=generator)


def export_fp32(model: MatchboxNetCTC, out_path: str | Path, n_frames: int | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = dummy_features(n_frames)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch", 2: "time"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return out_path


class ValSplitCalibrationReader:
    """`onnxruntime.quantization.CalibrationDataReader`-compatible reader
    sourced from the real `val` split (per this ticket's Unknowns section:
    static quantization needs real calibration data, not synthetic noise).

    Duck-typed rather than subclassing `CalibrationDataReader` directly --
    it only needs `get_next()`/`rewind()`, and duck-typing keeps this
    importable/testable without onnxruntime.quantization on the fast-test
    path.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        audio_root: str | Path | None = None,
        n_samples: int = 32,
        window_samples: int = WINDOW_SAMPLES,
        seed: int = 0,
    ) -> None:
        dataset = VCMDataset(manifest_path, audio_root=audio_root, split="val", augmenter=None)
        extractor = LogMelFeatureExtractor()

        rng = np.random.default_rng(seed)
        n = len(dataset)
        indices = rng.permutation(n)[: min(n_samples, n)].tolist()

        self._batches: list[dict[str, np.ndarray]] = []
        for idx in indices:
            waveform = dataset[idx].waveform
            if waveform.shape[-1] < window_samples:
                waveform = torch.nn.functional.pad(waveform, (0, window_samples - waveform.shape[-1]))
            else:
                waveform = waveform[:window_samples]
            feats = extractor(waveform).unsqueeze(0).numpy().astype(np.float32)
            self._batches.append({"features": feats})
        self._iter = iter(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._batches)


def quantize_int8_static(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_reader: ValSplitCalibrationReader,
) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    preprocessed_path = fp32_path.with_suffix(".preprocessed.onnx")
    quant_pre_process(str(fp32_path), str(preprocessed_path))

    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,
    )
    return int8_path


def onnx_vs_pytorch_logits(model: MatchboxNetCTC, onnx_path: str | Path, n_frames: int | None = None):
    """Run the same fixed input through the PyTorch model and an ONNX
    Runtime session over `onnx_path`, return `(torch_logits, onnx_logits)`
    as numpy arrays -- for export-correctness comparison."""
    import onnxruntime as ort

    model.eval()
    feats = dummy_features(n_frames)
    with torch.no_grad():
        torch_logits = model(feats).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(None, {"features": feats.numpy()})
    return torch_logits, onnx_logits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("out/vcm/checkpoint.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/vcm"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="export fp32 only; skip static INT8 quantization (fallback path)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.checkpoint)
    n_frames = window_n_frames()

    fp32_path = args.out_dir / "vcm_model.fp32.onnx"
    export_fp32(model, fp32_path, n_frames=n_frames)
    print(f"exported fp32 ONNX -> {fp32_path} ({fp32_path.stat().st_size:,} bytes)")

    if args.skip_quantization:
        return

    int8_path = args.out_dir / "vcm_model.int8.onnx"
    reader = ValSplitCalibrationReader(
        manifest_path=args.manifest, n_samples=args.calibration_samples
    )
    quantize_int8_static(fp32_path, int8_path, reader)
    print(f"exported static INT8 ONNX -> {int8_path} ({int8_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
