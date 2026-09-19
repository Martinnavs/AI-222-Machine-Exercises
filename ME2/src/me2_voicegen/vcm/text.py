"""Text normalization, the canonical intent-phrase table, and transcript
resolution across the five out/conversions/v2 test_set source datasets.

See docs/VCM-CONTRACT.md for the full transcript-resolution table this
module implements; this file is its source of truth, not the other way
around.
"""

from __future__ import annotations

import csv
import string
from functools import lru_cache
from pathlib import Path
from typing import Mapping

# src/me2_voicegen/vcm/text.py -> parents[3] is the ME2 project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONVERSIONS_V2_DIR = PROJECT_ROOT / "out" / "conversions" / "v2"

_ALLOWED_CHARS = set(string.ascii_lowercase) | {" ", "'"}
_CURLY_APOSTROPHES = {"‘", "’", "ʼ"}


def normalize_text(text: str) -> str:
    """Lowercase; fold curly apostrophes to `'`; drop/space out any other
    punctuation or symbol not in the 29-token alphabet; collapse runs of
    whitespace to a single space; strip leading/trailing whitespace.

    Guarantees the result is encodable by `vcm.alphabet.encode` unchanged.
    """
    text = text.lower()
    chars = []
    for ch in text:
        if ch in _CURLY_APOSTROPHES:
            chars.append("'")
        elif ch in _ALLOWED_CHARS:
            chars.append(ch)
        else:
            # Punctuation/digits/symbols/unicode letters outside a-z are
            # dropped, replaced by a word boundary so adjacent words don't
            # get glued together (e.g. "lights,on" -> "lights on").
            chars.append(" ")
    return " ".join("".join(chars).split())


# INTENT_PHRASES: <INTENT label> -> canonical normalized phrase.
# Verified against out/conversions/v2/reports/tmp-qa-<INTENT>-*.md "expected"
# column this session (see also the slow drift-guard test in
# tests/test_vcm_text.py that re-derives this table from those reports).
INTENT_PHRASES: dict[str, str] = {
    "ALARM": "set alarm",  # tmp-qa-ALARM-*.md
    "CALL": "call",  # tmp-qa-CALL-*.md
    "DIM_DOWN": "dimmer",  # tmp-qa-DIM-DOWN-*.md
    "DIM_UP": "brighter",  # tmp-qa-DIM-UP-*.md
    "LIGHT_OFF": "lights off",  # tmp-qa-LIGHT-OFF-*.md
    "LIGHT_ON": "lights on",  # tmp-qa-LIGHT-ON-*.md
    "LIST_REMINDERS": "list reminders",  # tmp-qa-LIST-REMINDERS-*.md
    "MESSAGE": "message",  # tmp-qa-MESSAGE-*.md
    "NEXT": "next",  # tmp-qa-NEXT-*.md
    "PAUSE": "pause",  # tmp-qa-PAUSE-*.md
    "PLAY_MUSIC": "play music",  # tmp-qa-PLAY-MUSIC-*.md
    "SET_REMINDER": "set reminder",  # tmp-qa-SET-REMINDER-*.md
    "STOP": "stop",  # tmp-qa-STOP-*.md
    "TEMP_DOWN": "cooler",  # tmp-qa-TEMP-DOWN-*.md
    "TEMP_UP": "warmer",  # tmp-qa-TEMP-UP-*.md
    "TIME": "time",  # tmp-qa-TIME-*.md
    "TIMER": "set timer",  # tmp-qa-TIMER-*.md
    "VOLUME_DOWN": "volume down",  # tmp-qa-VOLUME-DOWN-*.md
    "VOLUME_UP": "volume up",  # tmp-qa-VOLUME-UP-*.md
    "WEATHER": "weather",  # tmp-qa-WEATHER-*.md
}


@lru_cache(maxsize=None)
def _load_source_manifest(source_dataset: str) -> Mapping[str, dict]:
    """Load out/conversions/v2/<source_dataset>/manifest.csv keyed by its
    own `filename` column. Cached per source_dataset for the process
    lifetime (these files are read-only inputs, not mutated at runtime).
    """
    path = CONVERSIONS_V2_DIR / source_dataset / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def resolve_transcript(manifest_row: Mapping[str, str]) -> str | None:
    """Resolve the ground-truth transcript for one out/conversions/v2/
    test_set/manifest.csv row, joining back to that row's own source
    dataset's manifest. See docs/VCM-CONTRACT.md for the full rules table;
    summary:

    - sanitized_clean: no source manifest exists -> INTENT_PHRASES[label].
    - common_voice_negative: source manifest `transcript` column (per-chunk
      Whisper transcript; may be "").
    - youtube_institutional: source manifest `transcript` column (blank for
      non-speech `ambient`-bucket rows -- that's correct, not missing data).
    - background_noise: no transcript column, no speech present -> "".
    - filipino_speech_corpus: per decision (B), always None, regardless of
      whether the row is a whole-clip or `_cNN.wav` chunked row -- excluded
      from CTC loss uniformly, kept only as an eval rejection probe.
    """
    source_dataset = manifest_row["source_dataset"]

    if source_dataset == "sanitized_clean":
        return INTENT_PHRASES[manifest_row["label"]]

    if source_dataset == "background_noise":
        return ""

    if source_dataset == "filipino_speech_corpus":
        return None

    if source_dataset in ("common_voice_negative", "youtube_institutional"):
        source_manifest = _load_source_manifest(source_dataset)
        basename = Path(manifest_row["source_relpath"]).name
        source_row = source_manifest[basename]
        return source_row["transcript"]

    raise ValueError(f"unknown source_dataset: {source_dataset!r}")
