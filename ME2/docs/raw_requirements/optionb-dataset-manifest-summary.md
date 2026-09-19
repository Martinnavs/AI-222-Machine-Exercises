<!--
Derived from the real upstream `manifest.csv` (17,656 data rows), source repo
`markandrian30/AI231`, path `MEX2/OptionB/manifest.csv`. Fetched/derived
2026-09-19, commit `74cdfee2ff5a5d015fd951dad7b0ff396df4d47f`.

This is NOT a copy of the vendored README
(`docs/raw_requirements/optionb-dataset-readme.md`) -- it is independently
derived by grouping the manifest's own `transcript` column by
`(intent, phrase_id)` and, for slotted intents, substituting each row's own
`slot_value` back out of its `transcript` to recover the template. It exists
because the README and the real dataset disagree on one phrase (see
"Divergence from the README" below), and the drift guard in
`tests/test_optionb_grammar.py` treats this table as the primary correctness
anchor for that one phrase while keeping the README as the secondary anchor
for everything else. Quoted reference material -- do not edit by hand;
regenerate from a fresh manifest fetch if the upstream dataset changes.
-->

# Option B dataset: manifest-derived transcript summary

Derived directly from the real `manifest.csv`'s `transcript`, `intent`,
`phrase_id`, `slot`, and `slot_value` columns (17,656 data rows, 93 distinct
transcripts -- matches `OPTIONB_GRAMMAR`'s canonical-93 count exactly, but
**not the same 93 strings**; see divergence below).

## Phrase variations

Same 5-column shape as the README's own "Phrase variations" table (`Group`
retained but not meaningful here -- kept only so both anchors parse with the
same table-extraction logic in `tests/test_optionb_grammar.py`).

| Group | Intent | v1 | v2 | v3 |
|---|---|---|---|---|
| - | `PLAY_MUSIC` | Play music | Play a song | Start the music |
| - | `VOLUME_UP` | Volume up | Increase the volume | Turn the volume up |
| - | `VOLUME_DOWN` | Volume down | Lower the volume | Turn the volume down |
| - | `NEXT` | Skip song | Next song | Play next song |
| - | `PAUSE` | Pause | Pause the music | Pause this song |
| - | `STOP` | Stop song | Stop music | Stop playing music |
| - | `LIGHT_ON` | Lights on | Power on the lights | Turn on the lights |
| - | `LIGHT_OFF` | Lights off | Kill the lights | Turn off the lights |
| - | `BRIGHTNESS` | Brightness {percent} | Set the brightness to {percent} | Change the brightness to {percent} |
| - | `COLOR` | Color {color} | Change the lights to {color} | Set the lights to {color} |
| - | `TEMPERATURE` | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} |
| - | `WEATHER` | Weather | What's the weather? | Tell me the weather |
| - | `TIME` | Time | What time is it? | Tell me the time |
| - | `TIMER` | Timer {duration} | Countdown for {duration} | Start a timer for {duration} |
| - | `ALARM` | Alarm {time} | Wake me up at {time} | Set an alarm for {time} |
| - | `CALL` | Call | Make a call | Make a phone call |
| - | `MESSAGE` | Message | Send a message | Send my message |
| - | `CREATE_REMINDER` | Reminder {task} | Remind me to {task} | Create a reminder to {task} |
| - | `LIST_REMINDERS` | Reminders | Show my reminders | List my reminders |

### Slot values

Same 3-column shape as the README's own "Slot values" table.

| Intent | Slot | Values |
|---|---|---|
| `TIMER` | `{duration}` | 10 seconds; 30 seconds; 1 minute |
| `ALARM` | `{time}` | 6 AM; 8 AM; 9 PM |
| `TEMPERATURE` | `{degrees}` | 18 degrees; 22 degrees; 26 degrees |
| `BRIGHTNESS` | `{percent}` | 20 percent; 60 percent; 100 percent |
| `COLOR` | `{color}` | red; blue; green |
| `CREATE_REMINDER` | `{task}` | drink water; study; exercise |

Every slot value above is identical to the vendored README's slot-value
table -- the divergence below is confined to phrasing, not slot values (no
slot-value changes are in scope for this amendment; see the ticket's
non-goals).

## Divergence from the README

**Exactly one phrase diverges.** `VOLUME_DOWN`'s v2 phrasing is
**"Lower the volume"** in the real manifest -- verified across all 196
recorded rows of that transcript, zero exceptions -- while the vendored
README (`optionb-dataset-readme.md`) says **"Decrease the volume"**, which
appears in **zero** rows of the real manifest. Set difference over the two
93-transcript tables:

```
in MANIFEST but not in README: {"lower the volume"}
in README but not in MANIFEST: {"decrease the volume"}
```

The audio is ground truth; the README is stale documentation. This is
recorded as `KNOWN_README_DIVERGENCES` in `optionb/grammar.py` and asserted
(not silently tolerated) by both drift-guard tests in
`tests/test_optionb_grammar.py`. `OPTIONB_GRAMMAR`'s `$CMD_VOLUME_DOWN` v2
literal was changed from `"decrease the volume"` to `"lower the volume"` to
match the real dataset; this is a one-phrase substitution, so the canonical
count stays 93 and the accepted count stays 129.

If a future upstream fetch corrects the README to say "Lower the volume",
this file's row and the `KNOWN_README_DIVERGENCES` entry become obsolete and
should be removed together -- compare the fetched commit SHA above against
a fresh `git ls-remote` of `markandrian30/AI231` to detect that.
