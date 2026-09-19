# VCM Contract

Shared contract for the toy CTC voice-command model (`vcm`) feature. Every
task in this feature (02 dataset/features, 03 grammar decoder, 04/05
training/eval, 07 integration) reads this doc instead of re-deriving any of
the facts below. Source of truth for each section is the module/file named;
if this doc and that module ever disagree, the module wins and this doc is
stale and needs fixing.

## 1. Alphabet (source of truth: `src/me2_voicegen/vcm/alphabet.py`)

29-token CTC alphabet:

| id | token |
|---|---|
| 0 | CTC blank |
| 1-26 | `a`-`z` |
| 27 | ` ` (space) |
| 28 | `'` (apostrophe) |

**Discrepancy, documented, not "fixed":** the original spec text this
project derives from says the alphabet has "32 tokens", but the spec's own
enumeration (blank + 26 letters + space + apostrophe) only lists 29 distinct
symbols. `vcm.alphabet` implements the 29 tokens the spec's enumeration
actually describes. It does **not** pad to 32 with unused/reserved ids —
any later task that needs a 32-wide output layer for some other reason must
say so explicitly and explain why, rather than assuming 32 is this
contract's alphabet size.

`vcm.alphabet` provides `encode(text) -> list[int]`, `decode(ids) -> str`,
and `collapse(ids) -> list[int]` (CTC repeat-collapse then blank-drop, for
raw per-frame argmax sequences — not for `encode`'s output, which never
contains blanks or repeats to collapse).

## 2. Text normalization (source of truth: `src/me2_voicegen/vcm/text.py`)

`normalize_text(text) -> str` guarantees:
- lowercased
- curly apostrophes (`'` U+2018/U+2019, `ʼ` U+02BC) folded to straight `'`
- every other character not in the 29-token alphabet (digits, punctuation,
  non-ASCII letters, symbols) is dropped and replaced by a word boundary
  (space), so adjacent words are never glued together
- whitespace collapsed to single spaces, leading/trailing whitespace
  stripped

Output of `normalize_text` is always encodable by `vcm.alphabet.encode`
unchanged.

## 3. Canonical intent phrases (`INTENT_PHRASES`, in `vcm/text.py`)

20 entries, `<INTENT label>` (as it appears in the `target_commands` bucket
of `test_set/manifest.csv`'s `label` column) -> canonical normalized phrase:

| intent | phrase |
|---|---|
| ALARM | `set alarm` |
| CALL | `call` |
| DIM_DOWN | `dimmer` |
| DIM_UP | `brighter` |
| LIGHT_OFF | `lights off` |
| LIGHT_ON | `lights on` |
| LIST_REMINDERS | `list reminders` |
| MESSAGE | `message` |
| NEXT | `next` |
| PAUSE | `pause` |
| PLAY_MUSIC | `play music` |
| SET_REMINDER | `set reminder` |
| STOP | `stop` |
| TEMP_DOWN | `cooler` |
| TEMP_UP | `warmer` |
| TIME | `time` |
| TIMER | `set timer` |
| VOLUME_DOWN | `volume down` |
| VOLUME_UP | `volume up` |
| WEATHER | `weather` |

Each entry was verified against its own QA report's "expected" column
(`out/conversions/v2/reports/tmp-qa-<INTENT with `_`->`-`>-*.md`) this
session. `tests/test_vcm_text.py` has a `@pytest.mark.slow` drift-guard test
that re-derives this table from those report files on disk and asserts an
exact match against the hardcoded table above — run it whenever the QA
reports change.

## 4. Transcript resolution (source of truth: `vcm.text.resolve_transcript`)

`resolve_transcript(manifest_row) -> str | None` takes one row (dict) of
`out/conversions/v2/test_set/manifest.csv` and joins it back to its own
source dataset's manifest, keyed on `basename(source_relpath)` ==
that manifest's own `filename` column (verified zero misses across all four
sources with real manifests, this session, over the full 2246-row
`test_set/manifest.csv`):

| `source_dataset` | rule |
|---|---|
| `sanitized_clean` (1460 rows) | No manifest exists for this source. `INTENT_PHRASES[label]`. |
| `common_voice_negative` (187 rows) | Has a `transcript` column (per-chunk Whisper transcript). Use `transcript`, **not** `sentence` (that's the whole-clip prompt text, unreliable per-chunk). 10 of 187 rows have an empty transcript — resolves to `""`, not `None`. |
| `youtube_institutional` (218 rows) | Has a `transcript` column. 34 of 218 rows are blank — correctly, these are `ambient`-bucket non-speech rows; resolves to `""`, not `None`. |
| `background_noise` (194 rows) | No transcript column, no speech present. Always resolves to `""`. |
| `filipino_speech_corpus` (187 rows) | Has a `sentence` column, but per decision (B): **all** 187 rows resolve to `None` unconditionally, regardless of whether the row is a whole-clip or `_cNN.wav` chunked row. Excluded from CTC loss entirely; stays in the eval set only as a rejection/false-accept probe. |

Callers must distinguish `""` (real empty transcript / silence — include in
CTC loss as an empty-target sequence) from `None` (excluded from CTC loss
entirely) — they are not interchangeable.

## 5. Log-mel feature contract (Task 02 implements this; not yet built)

- 40 mel bins
- 480-sample analysis window (30ms at the manifest's 16kHz `sample_rate`)
- 160-sample hop (10ms)
- log magnitude with an epsilon floor (no `log(0)`)
- output layout: `(batch, n_mels, frames)`, i.e. `(B, 40, T)`

## 6. Padded-batch convention (Task 02's `collate_fn`; not yet built)

A batch is: `(B, 40, T)` log-mel float tensor (per §5) + `input_lengths`
(per-example valid frame count before padding) + `target_ids` (concatenated
or padded `vcm.alphabet.encode` output per example) + `target_len`
(per-example target token count) — the standard PyTorch CTC-loss batch
shape (`nn.CTCLoss` expects exactly this: padded input + input_lengths +
target + target_lengths).

## 7. Decoder input/output contract (Task 03 implements this; not yet built)

- **Input:** a CTC posterior/log-prob array over the 29-token alphabet
  (§1), shape `(T, 29)` for a single utterance or `(B, T, 29)` for a batch.
- **Output:** `{intent, slots: dict, text, confidence, no_match: bool}` —
  `intent` is one of the 20 `INTENT_PHRASES` keys or `None`, `slots` is
  whatever the grammar for that intent extracted (empty dict if none),
  `text` is the raw decoded/collapsed transcript, `confidence` is the
  decoder's own score, `no_match` is `True` when nothing in the grammar
  accepted the decode (`intent` is then `None`).

## 8. License provenance (CC-BY-NC-SA-4.0)

`out/conversions/v2/background_noise/` is sourced from ESC-50 and is
licensed CC-BY-NC-SA-4.0 (non-commercial, share-alike). It feeds the
`silence` bucket of `test_set/`. Because it is assembled into the same
`test_set/` used to train/eval any VCM checkpoint, **CC-BY-NC-SA-4.0
governs any checkpoint trained on this data** — non-commercial use only,
and any redistribution of the checkpoint or derived work must be shared
under the same license. Every later report or doc (Tasks 04/05/07) that
references a trained VCM checkpoint must carry this note; do not drop it
just because a given experiment's own training subset happened not to
sample any `background_noise` rows.
