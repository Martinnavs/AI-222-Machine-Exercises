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

.PHONY: sync vendor download-model generate generate-personas generate-conversions test clean vcm-train vcm-bench vcm-eval optionb-train optionb-eval stream

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

test:
	$(UV) run pytest

clean:
	rm -rf .venv vendor models out .pytest_cache
