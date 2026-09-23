UV ?= $(shell command -v uv 2>/dev/null || echo /opt/uv/uv)
export UV_PYTHON := /usr/bin/python3.10
UV_CACHE_DIR ?= /tmp/uv-cache-$(USER)
export UV_CACHE_DIR

VENDOR_DIR := vendor/CosyVoice
COSYVOICE_SHA := 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

BACKEND ?= cosyvoice2
TEXT ?=
MANIFEST ?= personas.json
RAW_DIR ?= $(HOME)/cosy-voice-data/Raw_Audio
REFS_DIR ?= $(HOME)/cosy-voice-data/References
RESYNTH_PROB ?= 0.3
SEED ?=

VCM_MANIFEST ?= out/conversions/v2/test_set/manifest.csv
VCM_OUT_DIR ?= out/vcm
VCM_PRESET ?= default
VCM_DEVICE ?= cpu
VCM_MAX_MINUTES ?= 30
VCM_SEED ?= 0

VCM_CHECKPOINT ?= $(VCM_OUT_DIR)/checkpoints/checkpoint.pt
VCM_BENCH_CALIBRATION_SAMPLES ?= 32
VCM_BENCH_N_ITERS ?= 100

VCM_EVAL_BEAM_WIDTH ?= 50
VCM_SLOT_EVAL_MANIFEST ?= $(VCM_OUT_DIR)/metadata/slot_eval/manifest.csv
VCM_GRAMMAR ?= spec,toy

OPTIONB_MANIFEST ?= out/conversions/v2/optionb/manifest.csv
OPTIONB_OUT_DIR ?= out/vcm/optionb

STREAM_MODEL ?= optionc
STREAM_BACKEND ?= onnx
STREAM_SOURCE ?= mic
STREAM_ARGS ?=

.PHONY: sync vendor download-model generate generate-personas generate-conversions test clean vcm-train vcm-bench vcm-eval optionb-train optionb-eval stream wakeword-fetch-positives wakeword-fetch-positives-dry-run wakeword-convert-positives wakeword-convert-positives-dry-run wakeword-generate-silence wakeword-generate-silence-dry-run wakeword-generate-adversaries wakeword-generate-adversaries-dry-run wakeword-mix-noise wakeword-mix-noise-dry-run wakeword-consolidate-manifest wakeword-build-unknown-external wakeword-build-unknown-external-dry-run wakeword-build-dataset

WAKEWORD_CLONE_DIR ?= $(if $(TMPDIR),$(TMPDIR),/tmp)/me2_wakeword_clone
WAKEWORD_POSITIVES_OUT_DIR ?= out/conversions/v2/wakeword/positives_real
WAKEWORD_CONVERTED_OUT_DIR ?= out/conversions/v2/wakeword/positives_converted
WAKEWORD_REFS_DIR ?= $(REFS_DIR)
WAKEWORD_K ?= 8
WAKEWORD_SEED ?= 0
WAKEWORD_MAX_CONVERSIONS ?=
WAKEWORD_SILENCE_OUT_DIR ?= out/conversions/v2/wakeword/silence_synthetic
WAKEWORD_SILENCE_SEED ?= 0
WAKEWORD_SILENCE_TOTAL_COUNT ?= 1806
WAKEWORD_ADVERSARIES_REFS_DIR ?= $(REFS_DIR)
WAKEWORD_ADVERSARIES_OUT_DIR ?= out/conversions/v2/wakeword/adversaries
WAKEWORD_ROOT ?= out/conversions/v2/wakeword
WAKEWORD_NOISE_ROOT ?= out/conversions/v2/background_noise
WAKEWORD_NOISE_PROB ?= 0.35
WAKEWORD_NOISE_SEED ?= 0
WAKEWORD_COMMON_VOICE_ROOT ?= out/conversions/v2/common_voice_negative
WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR ?= out/conversions/v2/wakeword/common_voice_negative_sample
WAKEWORD_UNKNOWN_TARGET_COUNT ?= 5172
WAKEWORD_UNKNOWN_SEED ?= 0
WAKEWORD_BUILD_SEED ?= 0

sync:
	$(UV) sync

vendor:
	@if [ -d "$(VENDOR_DIR)/.git" ] && [ "$$(cd $(VENDOR_DIR) && git rev-parse HEAD)" = "$(COSYVOICE_SHA)" ]; then \
		echo "vendor/CosyVoice already at pinned SHA $(COSYVOICE_SHA), skipping clone"; \
	else \
		rm -rf $(VENDOR_DIR); \
		mkdir -p vendor; \
		git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $(VENDOR_DIR); \
		cd $(VENDOR_DIR) && git checkout $(COSYVOICE_SHA) && git submodule update --init --recursive; \
	fi

download-model:
	$(UV) run python -m me2_voicegen.generation.download_model

generate:
	$(UV) run python -m me2_voicegen.generation.generate_sample --backend $(BACKEND) --text "$(TEXT)"

generate-personas:
	$(UV) run python -m me2_voicegen.generation.generate_personas --manifest "$(MANIFEST)" --backend $(BACKEND) --text "$(TEXT)"

generate-conversions:
	$(UV) run python -m me2_voicegen.generation.generate_conversions --raw-dir "$(RAW_DIR)" --refs-dir "$(REFS_DIR)" --backend $(BACKEND) --resynth-prob $(RESYNTH_PROB) $(if $(SEED),--seed $(SEED))

vcm-train:
	$(UV) run python -m me2_voicegen.vcm.train --manifest "$(VCM_MANIFEST)" --out-dir "$(VCM_OUT_DIR)" --preset $(VCM_PRESET) --device $(VCM_DEVICE) --max-minutes $(VCM_MAX_MINUTES) --seed $(VCM_SEED)

vcm-bench:
	$(UV) run python -m me2_voicegen.vcm.benchmark --checkpoint "$(VCM_CHECKPOINT)" --out-dir "$(VCM_OUT_DIR)" --manifest "$(VCM_MANIFEST)" --calibration-samples $(VCM_BENCH_CALIBRATION_SAMPLES) --n-iters $(VCM_BENCH_N_ITERS)

vcm-eval:
	$(UV) run python -m me2_voicegen.vcm.evaluate --manifest "$(VCM_MANIFEST)" --checkpoint "$(VCM_CHECKPOINT)" --out-dir "$(VCM_OUT_DIR)" --device $(VCM_DEVICE) --beam-width $(VCM_EVAL_BEAM_WIDTH) --slot-eval-manifest "$(VCM_SLOT_EVAL_MANIFEST)" --grammar $(VCM_GRAMMAR)

# Option B convenience targets: same vcm-train/vcm-eval machinery, pointed
# at out/conversions/v2/optionb/manifest.csv (D6c) and OPTIONB_GRAMMAR
# (D11, .scratch/optionb-dataset/tickets/04-pipeline-wiring.md), so the
# documented invocation is one command instead of a long VCM_MANIFEST=...
# override. `optionb-eval` intentionally has no --slot-eval-manifest
# default override -- Task 07's slot-eval-set is VCM-specific and does not
# exist for Option B; evaluate.py skips it gracefully when absent.
optionb-train:
	$(UV) run python -m me2_voicegen.vcm.train --manifest "$(OPTIONB_MANIFEST)" --out-dir "$(OPTIONB_OUT_DIR)" --preset $(VCM_PRESET) --device $(VCM_DEVICE) --max-minutes $(VCM_MAX_MINUTES) --seed $(VCM_SEED)

optionb-eval:
	$(UV) run python -m me2_voicegen.vcm.evaluate --manifest "$(OPTIONB_MANIFEST)" --checkpoint "$(OPTIONB_OUT_DIR)/checkpoints/checkpoint.pt" --out-dir "$(OPTIONB_OUT_DIR)" --device $(VCM_DEVICE) --beam-width $(VCM_EVAL_BEAM_WIDTH) --slot-eval-manifest "$(OPTIONB_OUT_DIR)/metadata/slot_eval/manifest.csv" --grammar optionb

# Streaming spoken-command runner (ticket 04 of feature optionb-streaming).
# STREAM_SOURCE defaults to the live microphone; override with a wav path
# (STREAM_SOURCE=path/to/clip.wav) for deterministic file-replay runs.
# STREAM_ARGS carries any extra flags (e.g. STREAM_ARGS="--listen-for 10
# --log-all-windows") straight through to the CLI.
stream:
	$(UV) run python -m me2_voicegen.vcm.streaming --model $(STREAM_MODEL) --backend $(STREAM_BACKEND) --source "$(STREAM_SOURCE)" $(STREAM_ARGS)

stream-single-period:
	$(MAKE) stream STREAM_ARGS="--policy single_period --gate spacebar --gate-period 3 --threshold -0.132222 --log-periods"

# Wakeword ("computer"-only) dataset-build targets. See
# docs/WAKEWORD-DATASET-CONTRACT.md for the manifest schema/output layout
# `fetch_positives.py` (ticket 01) produces; later tickets (02/03/04/05)
# add their own subset-fetch/convert/assemble targets alongside these.
#
# `?=` above does not repopulate a variable that is defined-but-empty (e.g.
# `make wakeword-fetch-positives WAKEWORD_POSITIVES_OUT_DIR=`), which would
# otherwise silently pass `--out-root ""` through to a tool that can rmtree
# that path (security-auditor finding R2-1) -- `fetch_positives.py` itself
# now refuses an empty/`.`-resolving --out-root/--clone-dir, but these two
# checks below fail the `make` invocation itself, before even launching
# python, as a second independent guard on the exact variables this
# Makefile exposes as overridable.
wakeword-fetch-positives:
	@test -n "$(strip $(WAKEWORD_CLONE_DIR))" || { echo "WAKEWORD_CLONE_DIR must not be empty" >&2; exit 1; }
	@test -n "$(strip $(WAKEWORD_POSITIVES_OUT_DIR))" || { echo "WAKEWORD_POSITIVES_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.fetch_positives --clone-dir "$(WAKEWORD_CLONE_DIR)" --out-root "$(WAKEWORD_POSITIVES_OUT_DIR)"

wakeword-fetch-positives-dry-run:
	@test -n "$(strip $(WAKEWORD_CLONE_DIR))" || { echo "WAKEWORD_CLONE_DIR must not be empty" >&2; exit 1; }
	@test -n "$(strip $(WAKEWORD_POSITIVES_OUT_DIR))" || { echo "WAKEWORD_POSITIVES_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.fetch_positives --clone-dir "$(WAKEWORD_CLONE_DIR)" --out-root "$(WAKEWORD_POSITIVES_OUT_DIR)" --dry-run

# Ticket 03: K=8 stratified voice-conversion of ticket 01's real positives.
# --seed is required by convert_positives.py itself (fails loudly if
# unset); WAKEWORD_SEED's own default here just keeps `make` runnable
# out of the box, same posture as VCM_SEED above. WAKEWORD_MAX_CONVERSIONS
# is empty by default (no cap, full run); set it to pilot/chunk real work,
# e.g. `make wakeword-convert-positives WAKEWORD_MAX_CONVERSIONS=20`.
wakeword-convert-positives:
	@test -n "$(strip $(WAKEWORD_CONVERTED_OUT_DIR))" || { echo "WAKEWORD_CONVERTED_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.convert_positives --positives-manifest "$(WAKEWORD_POSITIVES_OUT_DIR)/manifest.csv" --refs-dir "$(WAKEWORD_REFS_DIR)" --out-root "$(WAKEWORD_CONVERTED_OUT_DIR)" --k $(WAKEWORD_K) --seed $(WAKEWORD_SEED) $(if $(WAKEWORD_MAX_CONVERSIONS),--max-conversions $(WAKEWORD_MAX_CONVERSIONS))

wakeword-convert-positives-dry-run:
	@test -n "$(strip $(WAKEWORD_CONVERTED_OUT_DIR))" || { echo "WAKEWORD_CONVERTED_OUT_DIR must not be empty" >&2; exit 1; }
	$(UV) run python -m me2_voicegen.wakeword.convert_positives --positives-manifest "$(WAKEWORD_POSITIVES_OUT_DIR)/manifest.csv" --refs-dir "$(WAKEWORD_REFS_DIR)" --out-root "$(WAKEWORD_CONVERTED_OUT_DIR)" --k $(WAKEWORD_K) --seed $(WAKEWORD_SEED) --dry-run

# `generate_silence.py` (ticket 05) -- synthetic Gaussian/pink/brown
# colored-noise `_silence_` chunks. Fully independent of the
# wakeword-fetch-positives targets above (no shared clone dir, no network).
wakeword-generate-silence:
	$(UV) run python -m me2_voicegen.wakeword.generate_silence --out-root "$(WAKEWORD_SILENCE_OUT_DIR)" --seed $(WAKEWORD_SILENCE_SEED) --total-count $(WAKEWORD_SILENCE_TOTAL_COUNT)

wakeword-generate-silence-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.generate_silence --out-root "$(WAKEWORD_SILENCE_OUT_DIR)" --seed $(WAKEWORD_SILENCE_SEED) --total-count $(WAKEWORD_SILENCE_TOTAL_COUNT) --dry-run

# `generate_adversaries.py` (ticket 02) -- phonetic near-miss TTS renders of
# "computer" for the `_unknown_` class, across WAKEWORD_ADVERSARIES_REFS_DIR
# (defaults to the same reference-voice pool generate-conversions already
# uses, REFS_DIR). Independent of the fetch-positives/generate-silence
# targets above (no shared clone dir, no network -- local TTS only).
wakeword-generate-adversaries:
	$(UV) run python -m me2_voicegen.wakeword.generate_adversaries --refs-dir "$(WAKEWORD_ADVERSARIES_REFS_DIR)" --out-root "$(WAKEWORD_ADVERSARIES_OUT_DIR)" --backend $(BACKEND)

wakeword-generate-adversaries-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.generate_adversaries --refs-dir "$(WAKEWORD_ADVERSARIES_REFS_DIR)" --out-root "$(WAKEWORD_ADVERSARIES_OUT_DIR)" --backend $(BACKEND) --dry-run

# `mix_background_noise.py` -- probabilistically mixes background_noise
# (ESC-50, CC-BY-NC-SA-4.0) into adversaries/positives_converted, additively,
# writing new `*_noisy` sibling subsets (originals untouched). This is a
# deliberate, documented licensing-encumbrance decision -- see
# docs/WAKEWORD-DATASET-CONTRACT.md section 7 before running it.
wakeword-mix-noise:
	$(UV) run python -m me2_voicegen.wakeword.mix_background_noise --wakeword-root "$(WAKEWORD_ROOT)" --noise-root "$(WAKEWORD_NOISE_ROOT)" --noise-prob $(WAKEWORD_NOISE_PROB) --seed $(WAKEWORD_NOISE_SEED)

wakeword-mix-noise-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.mix_background_noise --wakeword-root "$(WAKEWORD_ROOT)" --noise-root "$(WAKEWORD_NOISE_ROOT)" --noise-prob $(WAKEWORD_NOISE_PROB) --seed $(WAKEWORD_NOISE_SEED) --dry-run

# `consolidate_manifest.py` -- flat merge of every subset generated so far
# into out/conversions/v2/wakeword/generated_manifest.csv. NOT ticket 04's
# final assembled dataset (no external corpora, no split assignment).
wakeword-consolidate-manifest:
	$(UV) run python -m me2_voicegen.wakeword.consolidate_manifest --wakeword-root "$(WAKEWORD_ROOT)"

# `build_unknown_external.py` -- scrubs common_voice_negative (CC0) for a
# "computer" token, seeded-samples the remainder to WAKEWORD_UNKNOWN_TARGET_COUNT,
# and writes common_voice_negative_sample/ (broadens `_unknown_` beyond
# adversaries' phonetic near-misses; no license encumbrance).
wakeword-build-unknown-external:
	$(UV) run python -m me2_voicegen.wakeword.build_unknown_external --common-voice-root "$(WAKEWORD_COMMON_VOICE_ROOT)" --out-root "$(WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR)" --target-count $(WAKEWORD_UNKNOWN_TARGET_COUNT) --seed $(WAKEWORD_UNKNOWN_SEED)

wakeword-build-unknown-external-dry-run:
	$(UV) run python -m me2_voicegen.wakeword.build_unknown_external --common-voice-root "$(WAKEWORD_COMMON_VOICE_ROOT)" --out-root "$(WAKEWORD_UNKNOWN_EXTERNAL_OUT_DIR)" --target-count $(WAKEWORD_UNKNOWN_TARGET_COUNT) --seed $(WAKEWORD_UNKNOWN_SEED) --dry-run

# `build_dataset.py` -- ticket 04's real final assembly: merges
# positives_real/positives_converted/positives_converted_noisy (_wakeword_),
# adversaries/adversaries_noisy/common_voice_negative_sample (_unknown_),
# silence_synthetic (_silence_), then a group-disjoint 70/20/10 split. Run
# wakeword-build-unknown-external and wakeword-generate-silence (sized to
# the intended class composition) before this.
wakeword-build-dataset:
	$(UV) run python -m me2_voicegen.wakeword.build_dataset --wakeword-root "$(WAKEWORD_ROOT)" --seed $(WAKEWORD_BUILD_SEED)

test:
	$(UV) run pytest

clean:
	rm -rf .venv vendor models out .pytest_cache
