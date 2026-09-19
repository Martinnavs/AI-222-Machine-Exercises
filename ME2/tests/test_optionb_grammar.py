"""Tests for me2_voicegen.optionb (see docs/OPTIONB-GRAMMAR-CONTRACT.md).

CANONICAL_93 and WORD_NUMBERS are hardcoded independently of the production
code under test: CANONICAL_93 is transcribed from the vendored README's
"Phrase variations"/"Slot values" tables, and WORD_NUMBERS is a hand-written
digit->word oracle, not derived from `optionb.numbers.spell_integer` --
otherwise a wrong mapping in the converter would be invisible to its own
test.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from me2_voicegen.optionb import OPTIONB_GRAMMAR, normalize_text, spell_integer

README_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "raw_requirements" / "optionb-dataset-readme.md"
)

# ---------------------------------------------------------------------------
# Hardcoded oracles.
# ---------------------------------------------------------------------------

CANONICAL_93: list[tuple[str, str, dict]] = [
    ("play music", "PLAY_MUSIC", {}),
    ("play a song", "PLAY_MUSIC", {}),
    ("start the music", "PLAY_MUSIC", {}),
    ("volume up", "VOLUME_UP", {}),
    ("increase the volume", "VOLUME_UP", {}),
    ("turn the volume up", "VOLUME_UP", {}),
    ("volume down", "VOLUME_DOWN", {}),
    ("decrease the volume", "VOLUME_DOWN", {}),
    ("turn the volume down", "VOLUME_DOWN", {}),
    ("skip song", "NEXT", {}),
    ("next song", "NEXT", {}),
    ("play next song", "NEXT", {}),
    ("pause", "PAUSE", {}),
    ("pause the music", "PAUSE", {}),
    ("pause this song", "PAUSE", {}),
    ("stop song", "STOP", {}),
    ("stop music", "STOP", {}),
    ("stop playing music", "STOP", {}),
    ("lights on", "LIGHT_ON", {}),
    ("power on the lights", "LIGHT_ON", {}),
    ("turn on the lights", "LIGHT_ON", {}),
    ("lights off", "LIGHT_OFF", {}),
    ("kill the lights", "LIGHT_OFF", {}),
    ("turn off the lights", "LIGHT_OFF", {}),
    ("weather", "WEATHER", {}),
    ("what's the weather", "WEATHER", {}),
    ("tell me the weather", "WEATHER", {}),
    ("time", "TIME", {}),
    ("what time is it", "TIME", {}),
    ("tell me the time", "TIME", {}),
    ("call", "CALL", {}),
    ("make a call", "CALL", {}),
    ("make a phone call", "CALL", {}),
    ("message", "MESSAGE", {}),
    ("send a message", "MESSAGE", {}),
    ("send my message", "MESSAGE", {}),
    ("reminders", "LIST_REMINDERS", {}),
    ("show my reminders", "LIST_REMINDERS", {}),
    ("list my reminders", "LIST_REMINDERS", {}),
    ("brightness 20 percent", "BRIGHTNESS", {"PERCENT": "20 percent"}),
    ("brightness 60 percent", "BRIGHTNESS", {"PERCENT": "60 percent"}),
    ("brightness 100 percent", "BRIGHTNESS", {"PERCENT": "100 percent"}),
    ("set the brightness to 20 percent", "BRIGHTNESS", {"PERCENT": "20 percent"}),
    ("set the brightness to 60 percent", "BRIGHTNESS", {"PERCENT": "60 percent"}),
    ("set the brightness to 100 percent", "BRIGHTNESS", {"PERCENT": "100 percent"}),
    ("change the brightness to 20 percent", "BRIGHTNESS", {"PERCENT": "20 percent"}),
    ("change the brightness to 60 percent", "BRIGHTNESS", {"PERCENT": "60 percent"}),
    ("change the brightness to 100 percent", "BRIGHTNESS", {"PERCENT": "100 percent"}),
    ("color red", "COLOR", {"COLOR": "red"}),
    ("color blue", "COLOR", {"COLOR": "blue"}),
    ("color green", "COLOR", {"COLOR": "green"}),
    ("change the lights to red", "COLOR", {"COLOR": "red"}),
    ("change the lights to blue", "COLOR", {"COLOR": "blue"}),
    ("change the lights to green", "COLOR", {"COLOR": "green"}),
    ("set the lights to red", "COLOR", {"COLOR": "red"}),
    ("set the lights to blue", "COLOR", {"COLOR": "blue"}),
    ("set the lights to green", "COLOR", {"COLOR": "green"}),
    ("temperature 18 degrees", "TEMPERATURE", {"DEGREES": "18 degrees"}),
    ("temperature 22 degrees", "TEMPERATURE", {"DEGREES": "22 degrees"}),
    ("temperature 26 degrees", "TEMPERATURE", {"DEGREES": "26 degrees"}),
    ("change the temperature to 18 degrees", "TEMPERATURE", {"DEGREES": "18 degrees"}),
    ("change the temperature to 22 degrees", "TEMPERATURE", {"DEGREES": "22 degrees"}),
    ("change the temperature to 26 degrees", "TEMPERATURE", {"DEGREES": "26 degrees"}),
    ("set the temperature to 18 degrees", "TEMPERATURE", {"DEGREES": "18 degrees"}),
    ("set the temperature to 22 degrees", "TEMPERATURE", {"DEGREES": "22 degrees"}),
    ("set the temperature to 26 degrees", "TEMPERATURE", {"DEGREES": "26 degrees"}),
    ("timer 10 seconds", "TIMER", {"DURATION": "10 seconds"}),
    ("timer 30 seconds", "TIMER", {"DURATION": "30 seconds"}),
    ("timer 1 minute", "TIMER", {"DURATION": "1 minute"}),
    ("countdown for 10 seconds", "TIMER", {"DURATION": "10 seconds"}),
    ("countdown for 30 seconds", "TIMER", {"DURATION": "30 seconds"}),
    ("countdown for 1 minute", "TIMER", {"DURATION": "1 minute"}),
    ("start a timer for 10 seconds", "TIMER", {"DURATION": "10 seconds"}),
    ("start a timer for 30 seconds", "TIMER", {"DURATION": "30 seconds"}),
    ("start a timer for 1 minute", "TIMER", {"DURATION": "1 minute"}),
    ("alarm 6 am", "ALARM", {"ALARM_TIME": "6 AM"}),
    ("alarm 8 am", "ALARM", {"ALARM_TIME": "8 AM"}),
    ("alarm 9 pm", "ALARM", {"ALARM_TIME": "9 PM"}),
    ("wake me up at 6 am", "ALARM", {"ALARM_TIME": "6 AM"}),
    ("wake me up at 8 am", "ALARM", {"ALARM_TIME": "8 AM"}),
    ("wake me up at 9 pm", "ALARM", {"ALARM_TIME": "9 PM"}),
    ("set an alarm for 6 am", "ALARM", {"ALARM_TIME": "6 AM"}),
    ("set an alarm for 8 am", "ALARM", {"ALARM_TIME": "8 AM"}),
    ("set an alarm for 9 pm", "ALARM", {"ALARM_TIME": "9 PM"}),
    ("reminder drink water", "CREATE_REMINDER", {"TASK": "drink water"}),
    ("reminder study", "CREATE_REMINDER", {"TASK": "study"}),
    ("reminder exercise", "CREATE_REMINDER", {"TASK": "exercise"}),
    ("remind me to drink water", "CREATE_REMINDER", {"TASK": "drink water"}),
    ("remind me to study", "CREATE_REMINDER", {"TASK": "study"}),
    ("remind me to exercise", "CREATE_REMINDER", {"TASK": "exercise"}),
    ("create a reminder to drink water", "CREATE_REMINDER", {"TASK": "drink water"}),
    ("create a reminder to study", "CREATE_REMINDER", {"TASK": "study"}),
    ("create a reminder to exercise", "CREATE_REMINDER", {"TASK": "exercise"}),
]

assert len(CANONICAL_93) == 93

# (digit, word) -- independent hand-written oracle, per D11.
WORD_NUMBERS: list[tuple[int, str]] = [
    (1, "one"),
    (6, "six"),
    (8, "eight"),
    (9, "nine"),
    (10, "ten"),
    (18, "eighteen"),
    (20, "twenty"),
    (22, "twenty two"),
    (26, "twenty six"),
    (30, "thirty"),
    (60, "sixty"),
    (100, "one hundred"),
]

_WORD_OF = dict(WORD_NUMBERS)

# Templates that carry a numeric slot, keyed by the digit-form value token
# used in CANONICAL_93 -- used to build the 36 expected word-form expansions
# without touching production code.
_NUMERIC_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "BRIGHTNESS": [
        ("brightness {v}", "PERCENT"),
        ("set the brightness to {v}", "PERCENT"),
        ("change the brightness to {v}", "PERCENT"),
    ],
    "TEMPERATURE": [
        ("temperature {v}", "DEGREES"),
        ("change the temperature to {v}", "DEGREES"),
        ("set the temperature to {v}", "DEGREES"),
    ],
    "TIMER": [
        ("timer {v}", "DURATION"),
        ("countdown for {v}", "DURATION"),
        ("start a timer for {v}", "DURATION"),
    ],
    "ALARM": [
        ("alarm {v}", "ALARM_TIME"),
        ("wake me up at {v}", "ALARM_TIME"),
        ("set an alarm for {v}", "ALARM_TIME"),
    ],
}

# (number, unit_words, canonical_value) per numeric slot -- hardcoded from
# the README's slot-value table, independent of optionb/grammar.py.
_NUMERIC_VALUES: dict[str, list[tuple[int, str, str]]] = {
    "PERCENT": [(20, "percent", "20 percent"), (60, "percent", "60 percent"), (100, "percent", "100 percent")],
    "DEGREES": [(18, "degrees", "18 degrees"), (22, "degrees", "22 degrees"), (26, "degrees", "26 degrees")],
    "DURATION": [(10, "seconds", "10 seconds"), (30, "seconds", "30 seconds"), (1, "minute", "1 minute")],
    "ALARM_TIME": [(6, "am", "6 AM"), (8, "am", "8 AM"), (9, "pm", "9 PM")],
}


def _word_form_rows() -> list[tuple[str, str, dict]]:
    rows = []
    for intent, templates in _NUMERIC_TEMPLATES.items():
        for template, slot_name in templates:
            for number, unit_word, canonical_value in _NUMERIC_VALUES[slot_name]:
                word_value = f"{_WORD_OF[number]} {unit_word}"
                phrase = template.format(v=word_value)
                rows.append((phrase, intent, {slot_name: canonical_value}))
    return rows


WORD_FORM_36 = _word_form_rows()
assert len(WORD_FORM_36) == 36

ACCEPTED_129 = CANONICAL_93 + WORD_FORM_36
assert len(ACCEPTED_129) == 129


# ---------------------------------------------------------------------------
# Criterion 1: exactly 129 alternatives, all distinct, over exactly 19 rules.
# ---------------------------------------------------------------------------

def test_all_phrases_count_and_distinct():
    phrases = list(OPTIONB_GRAMMAR.all_phrases())
    texts = [p for p, _, _ in phrases]
    assert len(texts) == 129
    assert len(set(texts)) == 129
    assert len(OPTIONB_GRAMMAR.rules) == 19


def test_19_distinct_intents():
    intents = {intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()}
    assert len(intents) == 19
    assert intents == {
        "PLAY_MUSIC", "VOLUME_UP", "VOLUME_DOWN", "NEXT", "PAUSE", "STOP",
        "LIGHT_ON", "LIGHT_OFF", "WEATHER", "TIME", "CALL", "MESSAGE",
        "LIST_REMINDERS", "BRIGHTNESS", "COLOR", "TEMPERATURE", "TIMER",
        "ALARM", "CREATE_REMINDER",
    }


# ---------------------------------------------------------------------------
# Criterion 2: every canonical phrase accepted, correct (intent, slots).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,intent,slots", CANONICAL_93)
def test_canonical_phrase_accepted(phrase, intent, slots):
    result = OPTIONB_GRAMMAR.accepts(phrase)
    assert result == [(intent, slots)]


# ---------------------------------------------------------------------------
# D11: spell_integer oracle.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number,word", WORD_NUMBERS)
def test_spell_integer_matches_oracle(number, word):
    assert spell_integer(number) == word


@pytest.mark.parametrize("bad", [-1, 101])
def test_spell_integer_out_of_range(bad):
    with pytest.raises(ValueError):
        spell_integer(bad)


# ---------------------------------------------------------------------------
# Criterion 3: digit/word parity.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,intent,slots", WORD_FORM_36)
def test_word_form_parity_with_digit_form(phrase, intent, slots):
    word_result = OPTIONB_GRAMMAR.accepts(phrase)
    assert word_result == [(intent, slots)]
    # Canonical value never contains a spelled-out number.
    canonical_value = next(iter(slots.values()))
    assert canonical_value[0].isdigit()


def test_digit_word_pairs_identical_result():
    pairs = [
        ("timer 10 seconds", "timer ten seconds"),
        ("alarm 6 am", "alarm six am"),
        ("temperature 18 degrees", "temperature eighteen degrees"),
        ("brightness 20 percent", "brightness twenty percent"),
    ]
    for digit_phrase, word_phrase in pairs:
        digit_result = OPTIONB_GRAMMAR.accepts(digit_phrase)
        word_result = OPTIONB_GRAMMAR.accepts(word_phrase)
        assert digit_result is not None
        assert digit_result == word_result
        slots = digit_result[0][1]
        canonical_value = next(iter(slots.values()))
        assert canonical_value[0].isdigit()


# ---------------------------------------------------------------------------
# Criterion 4: no over-generation.
# ---------------------------------------------------------------------------

def test_all_phrases_equals_canonical_union_word_forms():
    actual_texts = {t for t, _, _ in OPTIONB_GRAMMAR.all_phrases()}
    expected_texts = {p for p, _, _ in ACCEPTED_129}
    assert actual_texts == expected_texts
    assert len(expected_texts) == 129


# ---------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------

def test_normalize_text_preserves_digits_and_apostrophe():
    assert normalize_text("What's the weather?") == "what's the weather"
    assert normalize_text("Timer 10 Seconds!") == "timer 10 seconds"


def test_all_129_distinct_and_normalization_stable():
    texts = [t for t, _, _ in OPTIONB_GRAMMAR.all_phrases()]
    assert len(texts) == len(set(texts))
    for t in texts:
        assert normalize_text(t) == t


# ---------------------------------------------------------------------------
# Criterion 7: strict-prefix pairs.
#
# Established fact (ticket 02): among the canonical 93, exactly 5 strict
# character-level prefix pairs exist ("pause"/"pause the music",
# "pause"/"pause this song", "time"/"timer {duration}" x3 digit forms).
# Verified finding of *this* task (deviation from that ticket assumption,
# flagged for tech-lead): the same "time" vs "timer ..." relationship is a
# necessary consequence of the README's own phrase text ("Timer" starts
# with "Time") and recurs for the 3 TIMER word-form durations too, so the
# accepted-129 set has 8 strict-prefix pairs, not 5. This hardcodes the
# actual verified set rather than forcing an assertion that does not match
# the grammar's real (README-derived) behavior.
# ---------------------------------------------------------------------------

EXPECTED_STRICT_PREFIX_PAIRS = {
    ("pause", "pause the music"),
    ("pause", "pause this song"),
    ("time", "timer 10 seconds"),
    ("time", "timer 30 seconds"),
    ("time", "timer 1 minute"),
    ("time", "timer ten seconds"),
    ("time", "timer thirty seconds"),
    ("time", "timer one minute"),
}


def test_strict_prefix_pairs_exact_set():
    texts = sorted({t for t, _, _ in OPTIONB_GRAMMAR.all_phrases()})
    pairs = set()
    for a, b in itertools.combinations(texts, 2):
        if b.startswith(a):
            pairs.add((a, b))
        elif a.startswith(b):
            pairs.add((b, a))
    assert pairs == EXPECTED_STRICT_PREFIX_PAIRS


# ---------------------------------------------------------------------------
# Reject cases.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase",
    [
        "set brightness to 20 percent",
        "brightness percent",
        "alarm am",
        "play the music",
        "color",
        "timer one minutes",
        "brightness one hundred",
        "temperature twenty degrees",
        "",
    ],
)
def test_reject_cases(phrase):
    assert OPTIONB_GRAMMAR.accepts(phrase) is None


# ---------------------------------------------------------------------------
# Word-form over-generation reject cases: a genuine English number word,
# correctly paired with the slot's own unit word, but naming a value that
# was never recorded for that slot (digit or word) -- one probe per numeric
# slot, not just DEGREES, so a closed-world regression in any single slot's
# `_numeric_vocab(...)` call is caught.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase",
    [
        "brightness forty percent",
        "timer five seconds",
        "alarm ten am",
        "temperature twenty degrees",
    ],
)
def test_reject_word_form_over_generation_per_numeric_slot(phrase):
    assert OPTIONB_GRAMMAR.accepts(phrase) is None


# ---------------------------------------------------------------------------
# Drift guard (D7/D8): re-derive the canonical 93 from the vendored README.
# ---------------------------------------------------------------------------

_PLACEHOLDER_TO_SLOT = {
    "{duration}": "DURATION",
    "{time}": "ALARM_TIME",
    "{degrees}": "DEGREES",
    "{percent}": "PERCENT",
    "{color}": "COLOR",
    "{task}": "TASK",
}

def _parse_readme_slot_values_table(readme_text: str) -> dict[str, list[str]]:
    """Returns {slot_name: [value, ...]} parsed from the "Slot values" table."""
    lines = readme_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### Slot values"))
    slot_to_values: dict[str, list[str]] = {}
    for line in lines[start + 1 :]:
        if line.startswith("## ") or line.startswith("### "):
            break
        if not line.startswith("|") or "---" in line or line.strip().startswith("| Intent"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        _, slot_cell, values_cell = cells
        placeholder = slot_cell.strip("`")
        slot_name = _PLACEHOLDER_TO_SLOT[placeholder]
        values = [v.strip() for v in values_cell.split(";")]
        slot_to_values[slot_name] = values
    return slot_to_values


def _parse_readme_phrase_table(readme_text: str) -> list[tuple[str, str, str, str]]:
    """Returns (intent, v1, v2, v3) rows from the "Phrase variations" table."""
    lines = readme_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Phrase variations"))
    rows = []
    for line in lines[start:]:
        if line.startswith("### "):
            break
        if not line.startswith("|") or "---" in line or line.strip().startswith("| Group"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        _, intent_cell, v1, v2, v3 = cells
        intent = intent_cell.strip("`")
        rows.append((intent, v1, v2, v3))
    return rows


def _readme_phrase_to_words(phrase: str) -> list[str]:
    phrase = phrase.rstrip("?")
    return phrase.split()


def test_drift_guard_readme_matches_grammar():
    readme_text = README_PATH.read_text(encoding="utf-8")
    rows = _parse_readme_phrase_table(readme_text)
    assert len(rows) == 19
    slot_to_values = _parse_readme_slot_values_table(readme_text)
    assert len(slot_to_values) == 6

    derived_canonical: list[tuple[str, str, dict]] = []
    for intent, v1, v2, v3 in rows:
        for template in (v1, v2, v3):
            placeholder_match = re.search(r"\{\w+\}", template)
            if placeholder_match is None:
                text = normalize_text(" ".join(_readme_phrase_to_words(template)))
                derived_canonical.append((text, intent, {}))
                continue
            placeholder = placeholder_match.group(0)
            slot_name = _PLACEHOLDER_TO_SLOT[placeholder]
            for value in slot_to_values[slot_name]:
                filled = template.replace(placeholder, value)
                text = normalize_text(" ".join(_readme_phrase_to_words(filled)))
                derived_canonical.append((text, intent, {slot_name: value}))

    assert len(derived_canonical) == 93
    assert {p for p, _, _ in derived_canonical} == {p for p, _, _ in CANONICAL_93}

    for text, intent, slots in derived_canonical:
        result = OPTIONB_GRAMMAR.accepts(text)
        assert result == [(intent, slots)]

    actual_texts = {t for t, _, _ in OPTIONB_GRAMMAR.all_phrases()}
    expected_texts = {p for p, _, _ in derived_canonical} | {p for p, _, _ in WORD_FORM_36}
    assert actual_texts == expected_texts
