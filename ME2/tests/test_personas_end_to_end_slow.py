"""Real, end-to-end slow test for the persona-batch CLI path (Ticket 02 of the
persona-batch-generation plan).

Runs the REAL synthesis path through the factory against the real downloaded
CosyVoice2-0.5B weights and the real GPU, using a 2-entry persona manifest
built in tmp_path from the vendored zero_shot_prompt.wav asset - no mocking of
CosyVoice internals, AutoModel, or ONNX Runtime. Skipped (not failed) when the
weights or the vendored asset aren't present on this machine, mirroring
tests/test_end_to_end_slow.py's skip pattern, since that's an environment
precondition, not a code defect.

Both manifest entries point at the same vendored source clip
(zero_shot_prompt.wav) under two different persona names, because no
distinct-persona reference clips exist in this repo (see Ticket 01/02's
"Unknowns"). Identical voice timbre across the two output wavs is therefore
EXPECTED in this test and is not a bug - it only proves the mechanical path
(one model load, N distinctly-named non-silent outputs). Verifying that two
genuinely different reference clips actually produce two distinct timbres is
a manual, out-of-scope step the user performs separately with real persona
clips (see plan Non-Goals).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from me2_voicegen.download_model import missing_manifest_entries, target_dir
from me2_voicegen.personas import load_personas
from me2_voicegen.synthesis.base import save_wav
from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer
from me2_voicegen.synthesis.factory import create_synthesizer

pytestmark = pytest.mark.slow


def _skip_reasons() -> list[str]:
    reasons = []
    model_dir = target_dir()
    if not model_dir.is_dir():
        reasons.append("<entire model_dir missing>")
    else:
        reasons.extend(missing_manifest_entries(model_dir))
    if not CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV.is_file():
        reasons.append(
            f"vendored asset missing: {CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV}"
        )
    return reasons


@pytest.mark.skipif(
    bool(_skip_reasons()),
    reason=(
        "real CosyVoice2-0.5B weights and/or the vendored CosyVoice asset are "
        f"not present/complete (missing: {_skip_reasons()}); run "
        "`make download-model` and ensure vendor/CosyVoice is cloned first. "
        "Not mocked per this test's explicit scope."
    ),
)
def test_persona_batch_produces_distinct_wavs_from_single_model_load(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "personas.json"
    manifest_path.write_text(
        json.dumps(
            {
                "personas": [
                    {
                        "name": "english_woman",
                        "wav_path": str(CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV),
                        "text": CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT,
                    },
                    {
                        "name": "indian_man",
                        "wav_path": str(CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV),
                        "text": CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT,
                    },
                ]
            }
        )
    )

    personas = load_personas(manifest_path)
    assert len(personas) == 2

    synthesizer = create_synthesizer(
        "cosyvoice2", model_dir=str(target_dir()), fp16=False
    )
    assert isinstance(synthesizer, CosyVoice2Synthesizer)

    text = "Hey computer, could you please turn on the lights in the living room?"
    out_dir = tmp_path / "out"
    out_paths: list[Path] = []

    for persona in personas:
        result = synthesizer.synthesize(text, prompt=persona.prompt)
        assert result.sample_rate == 24000
        assert result.audio.dtype == np.float32
        assert result.audio.shape[-1] > 0

        out_path = out_dir / f"sample_{persona.name}.wav"
        save_wav(result, out_path)
        out_paths.append(out_path)

    assert len(out_paths) == 2
    assert out_paths[0] != out_paths[1]
    assert out_paths[0].name != out_paths[1].name

    for out_path in out_paths:
        assert out_path.exists()
        data, sr = sf.read(str(out_path), dtype="float32")
        assert sr == 24000
        assert len(data) > 0
        rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
        assert rms > 0.0, f"{out_path} is silent (RMS == 0)"
