# ME2 — CosyVoice 2 zero-shot TTS spike

Phase-1 spike proving that CosyVoice2 zero-shot voice-cloning synthesis runs end-to-end on this
HPC node (3x idle A100-SXM4-40GB, no root/sudo). It is **not** the synthetic-data-generation
pipeline itself.

## Scope

**In scope (this phase):**
- Vendoring upstream CosyVoice at a pinned commit and getting it importable under `uv`.
- Downloading the CosyVoice2-0.5B checkpoint.
- A swappable synthesis interface (`Synthesizer` ABC + factory) with one real backend
  (`cosyvoice2`) wired through a thin CLI.
- Proving a single zero-shot synthesis call produces a real, non-silent, intelligible wav on
  the A100, with a fast unit-test suite plus one real end-to-end GPU test.

**Explicitly NOT in scope (later work):**
- The full synthetic-data-generation pipeline for the wake-word/KWS corpus.
- Batch/looped generation over a phrase list.
- Phonetic adversaries (near-miss wake-word variants).
- Noise/RIR augmentation of generated clips.
- A second backend implementation (the interface is designed to make one easy to add later —
  see `docs/adding-a-tts-backend.md` — but none is implemented here).

See `sources.md`, `potential_model_approach.md`, `voice_generation_approach.md`, and
`streaming_approach.md` in this directory for the original planning context (read-only —
not updated by this spike).

## Prerequisites

`uv` is **not** on `$PATH` by default on this node — you must load it first:

```bash
module load uv
```

(The project's `Makefile` itself doesn't depend on this — it resolves `/opt/uv/uv` directly if
`uv` isn't found on `$PATH` — but you need `uv` on your own shell's `$PATH` for anything you run
by hand outside `make`, e.g. `uv run pytest -m slow` below.)

No Docker, no root/sudo required or used anywhere in this setup.

## Setup: the four `make` steps, in order

Run from `ME2/`:

```bash
cd ME2
make sync            # 1
make vendor           # 2
make download-model   # 3
make generate         # 4
```

1. **`make sync`** — `uv sync`. Creates `.venv` on Python 3.10 (pinned via `.python-version`
   and `UV_PYTHON`), installs the trimmed dependency set from `pyproject.toml` (see below),
   including `torch==2.3.1+cu121`/`torchaudio==2.3.1+cu121` from the explicit
   `pytorch-cu121` index. No manual `pip install` steps. Uses `/tmp/uv-cache-$USER` for uv's
   cache (not `$HOME`, which is on JuiceFS and slow for many-small-files operations like a
   package cache).

2. **`make vendor`** — clones upstream
   [`FunAudioLLM/CosyVoice`](https://github.com/FunAudioLLM/CosyVoice) (Apache-2.0) into
   `ME2/vendor/CosyVoice` at pinned SHA `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`, **recursively**
   (the `third_party/Matcha-TTS` submodule, MIT-licensed, pinned at
   `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`, is mandatory — a non-recursive clone silently
   omits it and the failure only surfaces later, as `ModuleNotFoundError: matcha` at model
   construction time, not at clone time). Idempotent: re-running when already at the pinned SHA
   is a no-op ("vendor/CosyVoice already at pinned SHA ..., skipping clone"). Upstream ships no
   `setup.py`/`pyproject.toml` — it's not pip-installable, which is why this is a `git clone`
   into a `sys.path`-shimmed vendor directory rather than a dependency.

3. **`make download-model`** — `uv run python -m me2_voicegen.download_model`. Downloads the
   CosyVoice2-0.5B checkpoint (Apache-2.0) into `ME2/models/CosyVoice2-0.5B/`: **19 files,
   ~4.9GB total** (`llm.pt` 2.02GB, `CosyVoice-BlankEN/model.safetensors` 988MB,
   `speech_tokenizer_v2.onnx` 496MB, `flow.pt` 451MB, plus `cosyvoice2.yaml`, `hift.pt`,
   `campplus.onnx`, and the rest of the `CosyVoice-BlankEN/` subdirectory). After downloading it
   verifies the file manifest and fails loudly, naming exactly which entries are missing, if
   incomplete. Idempotent — a second run detects a complete manifest and skips the download
   entirely (no network call; confirmed at ~0.4s on a warm check).

   **Intended path:** default source is `modelscope.snapshot_download('iic/CosyVoice2-0.5B', ...)`;
   pass `--source huggingface` to use `huggingface_hub.snapshot_download('FunAudioLLM/CosyVoice2-0.5B', ...)`
   instead:

   ```bash
   uv run python -m me2_voicegen.download_model --source huggingface
   ```

   **What actually happened on this node:** modelscope.cn was unreliable from here — a real
   attempt via the default source timed out after 7.5 minutes (`ReadTimeoutError`) and a retry
   was still crawling at 28MB transferred when abandoned. A `--source huggingface` attempt did
   complete (fast, ~12s, thanks to `huggingface_hub`'s content-addressed dedup) but the model's
   verified state on disk was ultimately established via a **manual out-of-band transfer**: the
   user downloaded the same checkpoint independently and uploaded it via FileZilla/SFTP directly
   into `ME2/models/CosyVoice2-0.5B/`. `make download-model`/`download_model.py`'s manifest check
   and idempotency logic both work correctly regardless of how the files got there — it only
   inspects what's on disk. If you're re-running this from scratch on a similarly flaky network
   path, don't be surprised if you need to fall back to a manual transfer too; the automated path
   is the intended default, not a guaranteed one from this node.

   Note: the manifest does **not** include `spk2info.pt` despite it appearing as a
   `CosyVoiceFrontEnd.__init__` parameter — it isn't shipped by either upstream registry and is
   an optional runtime artifact (created only when a speaker embedding is cached via
   `add_zero_shot_spk`/`save_spkinfo`); requiring it would make the manifest permanently
   unsatisfiable.

4. **`make generate`** — `uv run python -m me2_voicegen.generate_sample --backend $(BACKEND) --text "$(TEXT)"`
   (`BACKEND ?= cosyvoice2`, `TEXT ?=` — an empty `TEXT` falls through to `generate_sample.py`'s
   own natural-sentence default). Runs one real zero-shot synthesis call against the vendored
   CosyVoice2 code and the downloaded weights, writes a `.wav` to `ME2/out/`, and prints:

   ```
   output path: ME2/out/sample_<timestamp>.wav
   duration: 4.160s
   sample rate: 24000
   wall-clock synthesis time: 6.745s
   RTF: 1.621
   peak GPU memory: 2779971584 bytes
   ```

   Verified on the real A100: the synthesized English text ("Hey computer, could you please turn
   on the lights in the living room?"), cloned from the vendored Chinese zero-shot prompt asset
   (`asset/zero_shot_prompt.wav` + its transcript), was independently checked non-silent
   (RMS ~0.058, non-zero frame count) and re-transcribed with Whisper to confirm intelligibility
   — near-exact match to the input text. `inference_zero_shot` (not the `inference_cross_lingual`
   fallback) handles the Chinese-prompt/English-text cross-lingual case correctly on its own.

   You'll see a harmless warning during this step:
   ```
   UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
   ```
   This is expected — CosyVoice's own ONNX-based components (speech tokenizer, campplus) fall
   back to CPU because this project deliberately installs CPU-only `onnxruntime` rather than
   `onnxruntime-gpu` (see "Trimmed dependencies" below). It's a warning, not a failure; the
   overall pipeline still runs on the GPU (torch models load onto CUDA normally).

Override the backend or text at the command line, e.g.:

```bash
make generate TEXT="turn off the kitchen lights"
```

## The `--backend`/`--opt` CLI shape

`generate_sample.py` is a thin, backend-agnostic CLI — no backend-specific identifier (e.g.
`cosyvoice2`, `AutoModel`, `inference_zero_shot`, `cosyvoice`) appears anywhere in that file
except the `DEFAULT_BACKEND` constant's value. Dispatch goes entirely through
`synthesis.factory`. This is proven, not just asserted: `tests/test_generate_sample_cli.py` drives
a `FakeSynthesizer` with a deliberately unrelated constructor signature end-to-end through the
real CLI, and `test_smoke_fast_suite.py` asserts the real `cosyvoice` package never lands in
`sys.modules` from `--help` or `list_backends()` alone.

```bash
uv run python -m me2_voicegen.generate_sample \
  --backend cosyvoice2 \
  --text "some sentence" \
  --prompt-wav /path/to/reference.wav \
  --prompt-text "transcript of the reference clip" \
  --device auto \
  --opt fp16=true \
  --opt model_dir=/custom/model/dir
```

- `--backend` choices are built live from the factory's registry (`list_backends()`) — `--help`
  auto-updates as backends are added, no hardcoded choice list to keep in sync.
- Common flags (`--device` today) map to conventional constructor kwarg names and are only
  forwarded if the selected backend's `__init__` actually declares that parameter (checked via
  `inspect.signature`) — a backend that doesn't accept `device` simply doesn't receive it, no
  error.
- `--opt KEY=VALUE` (repeatable) is the escape hatch for anything backend-specific not covered by
  a common flag — passed through **unfiltered**, so an unknown key is a loud, immediate error
  rather than silently ignored. Values are coerced: `true`/`false` → `bool`, `none` → `None`,
  else `int`/`float`/`str`.
- Because the interface is genuinely swappable this way, a future backend (e.g. Piper, XTTS-v2)
  plugs in without touching this CLI file at all — see `docs/adding-a-tts-backend.md`.

## Trimmed dependencies

`pyproject.toml`'s dependency list is **not** a copy of upstream CosyVoice's
`requirements.txt`. The following are deliberately excluded, because this spike only needs
`AutoModel(...)` + `inference_zero_shot(...)` on a single node with no training, no
TensorRT/DeepSpeed acceleration, and no web UI:

| Excluded | Why |
|---|---|
| `deepspeed==0.15.1` | training-only (`bin/train.py`, `utils/train_utils.py`) |
| `tensorrt-cu12`, `tensorrt-cu12-bindings`, `tensorrt-cu12-libs` | needs `load_trt=True`, which defaults to `False` and is never set here |
| `onnxruntime-gpu==1.18.0` | replaced with CPU `onnxruntime==1.18.0` — CosyVoice's ONNX components (speech tokenizer, campplus) fall back to CPU with a warning, not a failure, when handed `CUDAExecutionProvider` and it isn't available |
| `gradio`, `fastapi`, `fastapi-cli`, `uvicorn`, `grpcio`, `grpcio-tools` | webui/runtime-server only, unused by direct `AutoModel` inference |
| `tensorboard`, `onnx` | training helpers |
| `gdown`, `wget` | **re-added** during this spike — see below |

**`gdown` and `wget` were re-added.** They looked like training-only helpers from an import grep
of `cosyvoice/` alone, but the real import chain (`cosyvoice.flow.flow_matching` →
`matcha.models.components.flow_matching` → `matcha.utils` → `matcha.utils.utils`) reaches them
at *module import time* via Matcha-TTS, not CosyVoice's own code. Re-added at their exact
upstream-pinned versions (`gdown==5.1.0`, `wget==3.2`) per the escape-hatch rule below —
this is the escape hatch working as designed, not a violation of the trimming decision. A plain
`setuptools<81` runtime dependency was added for the same reason (`lightning.fabric` does
`pkg_resources.declare_namespace(...)` at import time, and `setuptools>=81` removed
`pkg_resources`) — distinct from the `[tool.uv.extra-build-dependencies]` entry that only
affects `openai-whisper`'s isolated build environment.

**Escape-hatch rule for any future `ModuleNotFoundError`:** add back that *one* missing module,
at its exact `vendor/CosyVoice/requirements.txt`-pinned version, and log why in the relevant
ticket/PR. Do not upgrade, un-pin, or bulk-restore the whole upstream requirements file. If
`deepspeed` or `tensorrt` ever turn out to be genuinely required for real inference (as opposed to
training or accelerated-but-optional paths), stop and escalate to a human rather than silently
restoring them — those are the two packages the trimming decision was actually built around.

## Running the tests

```bash
make test                    # fast suite: no GPU, no vendor clone, no weights required
uv run pytest -m slow         # the one real end-to-end test: real GPU, real weights
```

`make test` runs `uv run pytest` with `addopts = "-m 'not slow'"`, so the default run excludes
the slow test automatically. As of the last full run: 57 fast tests pass (1 deselected). The
fast suite is guaranteed not to touch the vendor clone, model weights, or a GPU —
`test_smoke_fast_suite.py` asserts the real `cosyvoice` package never lands in `sys.modules` as a
side effect of importing the CLI or calling `list_backends()`/`get_backend_class()`, and that
`--help` exits 0 without it either (this was originally verified manually by renaming `vendor/`
away; that test makes the same guarantee an automatic regression check).

`uv run pytest -m slow` runs the one `@pytest.mark.slow` test
(`tests/test_end_to_end_slow.py`), which drives the real factory → real `AutoModel` →
`inference_zero_shot` → `save_wav` against the real downloaded weights and the real GPU, and
asserts the output is a valid, non-silent 24kHz wav. No mocks of CosyVoice internals, `AutoModel`,
or ONNX Runtime anywhere in this test — if the real thing can't run, that's meant to surface as a
real failure, not be papered over. It skips (doesn't fail) with a clear message if
`ME2/models/CosyVoice2-0.5B/` isn't present/complete. Last real run: 1 passed in ~55s on the A100.

## Cleaning up

`make clean` removes `.venv vendor models out .pytest_cache` — note this deletes the ~4.9GB
model download and the vendor clone along with cheap-to-regenerate build artifacts, so
re-running setup after `make clean` means repeating steps 2-3 above (`make vendor`,
`make download-model`) too, not just `make sync`.
