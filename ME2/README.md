# ME2 — Voice Cloning TTS (CosyVoice2)

**BLUF:** We can clone a voice from a few seconds of reference audio and generate new speech in
that voice, on our own GPUs. Working end-to-end, proven on real hardware. Potential use: generating
synthetic voice data (e.g. wake-word/command phrases across many voices) for training data.

## TL;DR

- **What:** [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) (open-source, Apache-2.0),
  zero-shot voice cloning — feed it a reference clip + its transcript, get new text spoken in
  that voice. No per-speaker training or fine-tuning needed.
- **Status:** Phase-1 spike. Proven working on our A100 GPUs, not yet the full data-generation
  pipeline.
- **Can do today:** synthesize one sentence in one voice; batch the same sentence across many
  reference voices ("personas") in one run.
- **Can't do yet:** batch multiple different sentences in one run, background noise/room
  augmentation, or swap in a second TTS engine (the code is built to make that easy later, just
  not done).
- **Heads up:** cloning someone's voice needs their consent — don't point this at a clip you don't
  have rights to use. No audio is committed to this repo.

## Try it

```bash
cd ME2
module load uv
make sync && make vendor && make download-model   # one-time setup (~5GB model download)
make generate TEXT="turn off the kitchen lights"
```

That's it — a `.wav` lands in `ME2/out/`, spoken in the sample voice, playable locally.

To clone your own voice: point `--prompt-wav`/`--prompt-text` at a reference clip + its exact
transcript (or use a personas manifest to batch several voices at once — see the implementation
doc).

## Why this could matter

If we need training data in many different voices (e.g. wake-word variants), this lets us
generate it from a handful of reference recordings instead of collecting real speakers for every
variant. Quality and speed are already validated on our hardware (~1.6x real-time per clip on an
A100); what's not built yet is the pipeline that turns this into a full labeled dataset.

## Everything else

Setup rationale, dependency decisions, the full CLI reference, batch/persona manifest format,
test suite, and troubleshooting notes: **[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)**.
