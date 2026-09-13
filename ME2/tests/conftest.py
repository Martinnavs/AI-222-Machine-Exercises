import sys

import pytest

import me2_voicegen.cosyvoice_env as cosyvoice_env


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Patch cosyvoice_env's module-level paths onto a throwaway tmp_path tree.

    Callers create whichever of vendor/CosyVoice, vendor/CosyVoice/third_party/
    Matcha-TTS they need for a given test; nothing is created by default.
    """
    project_root = tmp_path
    vendor_root = project_root / "vendor"
    cosyvoice_dir = vendor_root / "CosyVoice"
    matcha_dir = cosyvoice_dir / "third_party" / "Matcha-TTS"
    models_dir = project_root / "models"
    out_dir = project_root / "out"

    monkeypatch.setattr(cosyvoice_env, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(cosyvoice_env, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(cosyvoice_env, "COSYVOICE_DIR", cosyvoice_dir)
    monkeypatch.setattr(cosyvoice_env, "MATCHA_TTS_DIR", matcha_dir)
    monkeypatch.setattr(cosyvoice_env, "MODELS_DIR", models_dir)
    monkeypatch.setattr(cosyvoice_env, "OUT_DIR", out_dir)

    before = list(sys.path)
    yield cosyvoice_env, project_root, cosyvoice_dir, matcha_dir
    sys.path[:] = before
