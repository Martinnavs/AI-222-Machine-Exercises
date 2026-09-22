"""Invariants for the experimental Option B incomplete-prefix grammar."""

from __future__ import annotations

from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.incomplete_prefix_grammar import (
    OPTIONB_INCOMPLETE_PREFIX_GRAMMAR,
    derive_incomplete_prefixes,
)


def test_slot_bearing_prefixes_are_rejection_competitors():
    rejection_grammar = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR

    assert rejection_grammar.is_incomplete("color")
    assert rejection_grammar.is_incomplete("set the lights to")
    assert rejection_grammar.is_incomplete("brightness")
    assert rejection_grammar.is_incomplete("set an alarm for")


def test_only_complete_words_are_included():
    incomplete = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes

    assert "colo" not in incomplete
    assert "lights o" not in incomplete
    assert "set the lights to " not in incomplete


def test_accepted_commands_are_never_marked_incomplete():
    accepted = {text for text, _, _ in OPTIONB_GRAMMAR.all_phrases()}
    incomplete = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes

    assert accepted.isdisjoint(incomplete)
    assert "time" not in incomplete
    assert "pause" not in incomplete
    assert OPTIONB_GRAMMAR.accepts("time") is not None
    assert OPTIONB_GRAMMAR.accepts("pause") is not None


def test_rejection_grammar_does_not_accept_color():
    assert OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.command_grammar is OPTIONB_GRAMMAR
    assert OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.command_grammar.accepts("color") is None


def test_every_incomplete_prefix_has_a_longer_accepted_completion():
    rejection_grammar = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR

    for prefix in rejection_grammar.incomplete_prefixes:
        completions = rejection_grammar.completions_for(prefix)
        assert completions
        assert all(command.startswith(f"{prefix} ") for command in completions)


def test_derivation_is_deterministic_and_validated():
    assert derive_incomplete_prefixes(OPTIONB_GRAMMAR) == (
        OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes
    )
    OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.validate()
