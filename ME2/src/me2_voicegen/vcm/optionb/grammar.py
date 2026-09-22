"""OPTIONB_GRAMMAR: the closed-world Option B spoken-command grammar.

See docs/OPTIONB-GRAMMAR-CONTRACT.md for the full contract (rules, slot
vocabularies, normalization, canonical-93-vs-accepted-129 arithmetic). This
module is that contract's source of truth for the rules themselves.

Unlike `vcm.optiona.grammar`'s SPEC_GRAMMAR/TOY_GRAMMAR (which use the optional-
component combinator to productively generate optional-word variants),
Option B's dataset defines exactly 3 fixed phrasings/templates per intent --
there is no optional-word structure to encode. Every rule below is a flat
`alt()` of exactly its 3 literal phrasings/templates. The optional-component
combinator is intentionally never used here: using it would over-generate
phrases the dataset was never recorded with.
"""

from __future__ import annotations

import dataclasses

from me2_voicegen.common.grammar_core import (
    Grammar,
    alt,
    compile_grammar,
    derive_incomplete_prefixes,
    literal,
    seq,
    slot,
    with_intent,
    _vocab,
)
from .numbers import spell_integer


def _numeric_vocab(
    entries: list[tuple[int, tuple[str, ...], str]],
) -> list[tuple[tuple[str, ...], str]]:
    """Two (words, value) entries per (number, unit_words, canonical_value):
    the digit form and the spelled-out word form, both carrying the same
    canonical value."""
    out: list[tuple[tuple[str, ...], str]] = []
    for number, unit_words, canonical_value in entries:
        out.append(((str(number), *unit_words), canonical_value))
        out.append(((*spell_integer(number).split(), *unit_words), canonical_value))
    return out


# ---------------------------------------------------------------------------
# Slot vocabularies (README-verbatim canonical values).
# ---------------------------------------------------------------------------

DURATION: list[tuple[tuple[str, ...], str]] = _numeric_vocab([
    (10, ("seconds",), "10 seconds"),
    (30, ("seconds",), "30 seconds"),
    (1, ("minute",), "1 minute"),
])

ALARM_TIME: list[tuple[tuple[str, ...], str]] = _numeric_vocab([
    (6, ("am",), "6 AM"),
    (8, ("am",), "8 AM"),
    (9, ("pm",), "9 PM"),
])

DEGREES: list[tuple[tuple[str, ...], str]] = _numeric_vocab([
    (18, ("degrees",), "18 degrees"),
    (22, ("degrees",), "22 degrees"),
    (26, ("degrees",), "26 degrees"),
])

PERCENT: list[tuple[tuple[str, ...], str]] = _numeric_vocab([
    (20, ("percent",), "20 percent"),
    (60, ("percent",), "60 percent"),
    (100, ("percent",), "100 percent"),
])

COLOR: list[tuple[tuple[str, ...], str]] = _vocab("red", "blue", "green")
TASK: list[tuple[tuple[str, ...], str]] = _vocab("drink water", "study", "exercise")


# ---------------------------------------------------------------------------
# $CMD_* rules -- one alt() of exactly 3 literal phrasings/templates each,
# verbatim from docs/raw_requirements/optionb-dataset-readme.md's "Phrase
# variations" table, with exactly one documented exception
# (KNOWN_README_DIVERGENCES below): CMD_VOLUME_DOWN's v2 follows the vendored
# manifest-transcript table (optionb-dataset-manifest-summary.md), not the
# README, because the two disagree and the manifest is ground truth.
# ---------------------------------------------------------------------------

CMD_PLAY_MUSIC = with_intent(
    "PLAY_MUSIC",
    alt(literal("play", "music"), literal("play", "a", "song"), literal("start", "the", "music")),
)

CMD_VOLUME_UP = with_intent(
    "VOLUME_UP",
    alt(literal("volume", "up"), literal("increase", "the", "volume"), literal("turn", "the", "volume", "up")),
)

# v2 is "lower the volume", not the README's "decrease the volume" -- see
# KNOWN_README_DIVERGENCES below; the real dataset's 196 recorded rows of
# this phrase all say "lower", and "decrease" appears in zero rows.
CMD_VOLUME_DOWN = with_intent(
    "VOLUME_DOWN",
    alt(literal("volume", "down"), literal("lower", "the", "volume"), literal("turn", "the", "volume", "down")),
)

CMD_NEXT = with_intent(
    "NEXT",
    alt(literal("skip", "song"), literal("next", "song"), literal("play", "next", "song")),
)

CMD_PAUSE = with_intent(
    "PAUSE",
    alt(literal("pause"), literal("pause", "the", "music"), literal("pause", "this", "song")),
)

CMD_STOP = with_intent(
    "STOP",
    alt(literal("stop", "song"), literal("stop", "music"), literal("stop", "playing", "music")),
)

CMD_LIGHT_ON = with_intent(
    "LIGHT_ON",
    alt(literal("lights", "on"), literal("power", "on", "the", "lights"), literal("turn", "on", "the", "lights")),
)

CMD_LIGHT_OFF = with_intent(
    "LIGHT_OFF",
    alt(literal("lights", "off"), literal("kill", "the", "lights"), literal("turn", "off", "the", "lights")),
)

CMD_BRIGHTNESS = with_intent(
    "BRIGHTNESS",
    alt(
        seq(literal("brightness"), slot("PERCENT", PERCENT)),
        seq(literal("set", "the", "brightness", "to"), slot("PERCENT", PERCENT)),
        seq(literal("change", "the", "brightness", "to"), slot("PERCENT", PERCENT)),
    ),
)

CMD_COLOR = with_intent(
    "COLOR",
    alt(
        seq(literal("color"), slot("COLOR", COLOR)),
        seq(literal("change", "the", "lights", "to"), slot("COLOR", COLOR)),
        seq(literal("set", "the", "lights", "to"), slot("COLOR", COLOR)),
    ),
)

CMD_TEMPERATURE = with_intent(
    "TEMPERATURE",
    alt(
        seq(literal("temperature"), slot("DEGREES", DEGREES)),
        seq(literal("change", "the", "temperature", "to"), slot("DEGREES", DEGREES)),
        seq(literal("set", "the", "temperature", "to"), slot("DEGREES", DEGREES)),
    ),
)

CMD_WEATHER = with_intent(
    "WEATHER",
    alt(literal("weather"), literal("what's", "the", "weather"), literal("tell", "me", "the", "weather")),
)

CMD_TIME = with_intent(
    "TIME",
    alt(literal("time"), literal("what", "time", "is", "it"), literal("tell", "me", "the", "time")),
)

CMD_TIMER = with_intent(
    "TIMER",
    alt(
        seq(literal("timer"), slot("DURATION", DURATION)),
        seq(literal("countdown", "for"), slot("DURATION", DURATION)),
        seq(literal("start", "a", "timer", "for"), slot("DURATION", DURATION)),
    ),
)

CMD_ALARM = with_intent(
    "ALARM",
    alt(
        seq(literal("alarm"), slot("ALARM_TIME", ALARM_TIME)),
        seq(literal("wake", "me", "up", "at"), slot("ALARM_TIME", ALARM_TIME)),
        seq(literal("set", "an", "alarm", "for"), slot("ALARM_TIME", ALARM_TIME)),
    ),
)

CMD_CALL = with_intent(
    "CALL",
    alt(literal("call"), literal("make", "a", "call"), literal("make", "a", "phone", "call")),
)

CMD_MESSAGE = with_intent(
    "MESSAGE",
    alt(literal("message"), literal("send", "a", "message"), literal("send", "my", "message")),
)

CMD_CREATE_REMINDER = with_intent(
    "CREATE_REMINDER",
    alt(
        seq(literal("reminder"), slot("TASK", TASK)),
        seq(literal("remind", "me", "to"), slot("TASK", TASK)),
        seq(literal("create", "a", "reminder", "to"), slot("TASK", TASK)),
    ),
)

CMD_LIST_REMINDERS = with_intent(
    "LIST_REMINDERS",
    alt(literal("reminders"), literal("show", "my", "reminders"), literal("list", "my", "reminders")),
)


# ---------------------------------------------------------------------------
# Known divergences between the vendored README anchor and the vendored
# manifest-transcript anchor (docs/raw_requirements/optionb-dataset-
# manifest-summary.md). Each entry is (intent, readme_phrase, actual_phrase)
# where actual_phrase is what OPTIONB_GRAMMAR implements -- the manifest's
# version, since the manifest is ground truth. The drift guard
# (tests/test_optionb_grammar.py) asserts against both anchors and uses this
# table to reconcile the one known disagreement, rather than silently
# overriding or hiding it.
# ---------------------------------------------------------------------------

KNOWN_README_DIVERGENCES: list[tuple[str, str, str]] = [
    ("VOLUME_DOWN", "decrease the volume", "lower the volume"),
]


OPTIONB_RULES: dict[str, list[tuple[tuple[str, ...], dict, str]]] = {
    "$CMD_PLAY_MUSIC": CMD_PLAY_MUSIC,
    "$CMD_VOLUME_UP": CMD_VOLUME_UP,
    "$CMD_VOLUME_DOWN": CMD_VOLUME_DOWN,
    "$CMD_NEXT": CMD_NEXT,
    "$CMD_PAUSE": CMD_PAUSE,
    "$CMD_STOP": CMD_STOP,
    "$CMD_LIGHT_ON": CMD_LIGHT_ON,
    "$CMD_LIGHT_OFF": CMD_LIGHT_OFF,
    "$CMD_BRIGHTNESS": CMD_BRIGHTNESS,
    "$CMD_COLOR": CMD_COLOR,
    "$CMD_TEMPERATURE": CMD_TEMPERATURE,
    "$CMD_WEATHER": CMD_WEATHER,
    "$CMD_TIME": CMD_TIME,
    "$CMD_TIMER": CMD_TIMER,
    "$CMD_ALARM": CMD_ALARM,
    "$CMD_CALL": CMD_CALL,
    "$CMD_MESSAGE": CMD_MESSAGE,
    "$CMD_CREATE_REMINDER": CMD_CREATE_REMINDER,
    "$CMD_LIST_REMINDERS": CMD_LIST_REMINDERS,
}

# Inference-only rejection metadata (docs/INCOMPLETE-GRAMMAR-REJECTION.md):
# the designated whole-word prefixes ("color", "set the lights to", ...)
# attached to the grammar as immutable metadata. The accepted phrase surface
# is exactly the freshly compiled one -- the `dataclasses.replace` changes
# nothing but the new defaulted field, so `accepts`/`all_phrases` semantics
# are unchanged and "color" stays absent from the accepted grammar.
_OPTIONB_BASE: Grammar = compile_grammar("OPTIONB_GRAMMAR", OPTIONB_RULES)
OPTIONB_GRAMMAR: Grammar = dataclasses.replace(
    _OPTIONB_BASE,
    incomplete_prefixes=derive_incomplete_prefixes(_OPTIONB_BASE),
)
