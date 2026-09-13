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

.PHONY: sync vendor download-model generate generate-personas generate-conversions test clean

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
	$(UV) run python -m me2_voicegen.download_model

generate:
	$(UV) run python -m me2_voicegen.generate_sample --backend $(BACKEND) --text "$(TEXT)"

generate-personas:
	$(UV) run python -m me2_voicegen.generate_personas --manifest "$(MANIFEST)" --backend $(BACKEND) --text "$(TEXT)"

generate-conversions:
	$(UV) run python -m me2_voicegen.generate_conversions --raw-dir "$(RAW_DIR)" --refs-dir "$(REFS_DIR)" --backend $(BACKEND) --resynth-prob $(RESYNTH_PROB) $(if $(SEED),--seed $(SEED))

test:
	$(UV) run pytest

clean:
	rm -rf .venv vendor models out .pytest_cache
