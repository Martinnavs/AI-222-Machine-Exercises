"""Full evaluation CLI for the trained toy VCM CTC model: real checkpoint +
`vcm.pipeline`'s whole-clip mode over `out/conversions/v2/test_set/
manifest.csv`'s real `test` split.

Per decision (A) (docs/VCM-CONTRACT.md, ticket .scratch/vcm-toy/tickets/
05-pipeline-evaluate.md): `SPEC_GRAMMAR` and `TOY_GRAMMAR` are evaluated
and reported as two entirely separate sections, never merged. SPEC_GRAMMAR
is the verbatim BNF grammar -- 12 of the 20 `INTENT_PHRASES` intents'
canonical dataset phrases are not accepted by it. Only 2 of those (MESSAGE,
SET_REMINDER) have no SPEC_GRAMMAR rule at all; the other 10 (ALARM,
TIMER, CALL, TIME, WEATHER, LIST_REMINDERS, DIM_UP, DIM_DOWN, and the two
temperature-rule alternatives) DO have a `$CMD_*` BNF rule, but that rule
requires a slot or different phrasing than the dataset's bare canonical
phrase provides -- see `vcm.grammar`'s module docstring's "KNOWN GRAMMAR
LIMITATION" note and `tests/test_vcm_grammar.py::
test_message_and_set_reminder_have_no_spec_grammar_rule` (which asserts
the no-rule-at-all case for exactly those 2 intents, not all 12). So
SPEC_GRAMMAR is *expected* to report `no_match`/`REJECTED` for
essentially all of those 12 intents' clips by grammar construction --
that is a grammar-coverage fact, not a model failure, and both this
module's Markdown/JSON output say so explicitly (and correctly
distinguish the "no rule" vs. "rule needs a slot/different phrasing"
cases).

License (docs/VCM-CONTRACT.md section 8): `out/conversions/v2/
background_noise/` (feeding the `silence` bucket) is ESC-50,
CC-BY-NC-SA-4.0. Any report referencing the trained checkpoint's results
carries that provenance note in its header (this module reuses
`vcm.train.LICENSE_NOTE` verbatim rather than re-deriving separate
wording).

Threshold: `decode()`'s accept/reject cutoff is a mean per-frame
log-probability. This module never hardcodes an operating value -- it
sweeps a fixed candidate grid on the manifest's `val` split (never `test`)
per grammar, picks the value maximizing (target-accept-rate minus
false-accept-rate) on `babble`+`silence`, and reports the full sweep table
alongside the chosen point so the choice is auditable, not asserted.

`filipino_speech_corpus` rows resolve to a `None` transcript (excluded
from CTC training loss, per `vcm.text.resolve_transcript`) but are part of
the `babble` bucket in the manifest and are evaluated here like any other
`babble` row -- no special-casing needed, they are exactly the
rejection/false-accept probes decision (B) intended them to be.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.features import LogMelFeatureExtractor
from me2_voicegen.vcm.grammar import SPEC_GRAMMAR, TOY_GRAMMAR, Grammar
from me2_voicegen.vcm.pipeline import infer_waveform, load_checkpoint
from me2_voicegen.vcm.text import INTENT_PHRASES
from me2_voicegen.vcm.train import LICENSE_NOTE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "out" / "vcm" / "checkpoint.pt"
DEFAULT_SLOT_EVAL_MANIFEST = PROJECT_ROOT / "out" / "vcm" / "slot_eval" / "manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "vcm"

NEG_INF_THRESHOLD = float("-inf")

# Candidate operating points for the val-split sweep, spanning the
# confidence range actually observed for the Task 04 checkpoint (real
# target-command accepts and real babble false-accepts both cluster
# between 0 and about -1.3 mean-log-prob-per-frame; the tail out to -5
# is included so the sweep table isn't silently truncated if a future
# retrained checkpoint's distribution shifts wider).
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.0,
    -0.01,
    -0.02,
    -0.03,
    -0.05,
    -0.075,
    -0.1,
    -0.15,
    -0.2,
    -0.25,
    -0.3,
    -0.35,
    -0.4,
    -0.5,
    -0.75,
    -1.0,
    -1.5,
    -2.0,
    -5.0,
)

TARGET_BUCKET = "target_commands"
REJECT_PROBE_BUCKETS = ("babble", "silence")


@dataclasses.dataclass
class RowResult:
    index: int
    bucket: str
    label: str
    text: str
    intent: str | None
    confidence: float | None
    """`None` iff no grammar terminal was ever reached (equivalent to
    `-inf`, but JSON-serializable)."""


@torch.no_grad()
def decode_split(
    model,
    feature_extractor: LogMelFeatureExtractor,
    dataset: VCMDataset,
    grammar: Grammar,
    beam_width: int,
    device: str | torch.device,
) -> list[RowResult]:
    """Decode every row of `dataset` against `grammar` once, at the most
    permissive possible threshold (`-inf`) -- this captures each row's
    best-reachable intent/confidence independent of any operating
    threshold, so a threshold sweep afterwards is pure arithmetic over
    these cached results rather than re-running the model/decoder per
    candidate threshold."""
    results: list[RowResult] = []
    for i in range(len(dataset)):
        row = dataset.rows[i]
        example = dataset[i]
        decoded = infer_waveform(
            model,
            feature_extractor,
            example.waveform,
            grammar,
            threshold=NEG_INF_THRESHOLD,
            beam_width=beam_width,
            device=device,
        )
        confidence = None if decoded.intent is None else decoded.confidence
        results.append(
            RowResult(
                index=i,
                bucket=row["bucket"],
                label=row["label"],
                text=decoded.text,
                intent=decoded.intent,
                confidence=confidence,
            )
        )
    return results


def _accepted(r: RowResult, threshold: float) -> bool:
    return r.intent is not None and r.confidence is not None and r.confidence >= threshold


def sweep_thresholds(
    results: list[RowResult], thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID
) -> list[dict]:
    """Sweep candidate thresholds against already-decoded `val`-split
    results. `target_accept_rate` is the fraction of `target_commands`
    rows accepted (regardless of whether the accepted intent is correct --
    this sweep is about the accept/reject boundary, not per-intent
    accuracy, which `confusion_counts` reports separately on `test`).
    `false_accept_rate` is the fraction of `babble`+`silence` rows
    accepted (a false trigger)."""
    target_idxs = [i for i, r in enumerate(results) if r.bucket == TARGET_BUCKET]
    reject_idxs = [i for i, r in enumerate(results) if r.bucket in REJECT_PROBE_BUCKETS]

    sweep: list[dict] = []
    for t in thresholds:
        target_accepts = sum(1 for i in target_idxs if _accepted(results[i], t))
        reject_accepts = sum(1 for i in reject_idxs if _accepted(results[i], t))
        target_rate = target_accepts / len(target_idxs) if target_idxs else 0.0
        false_rate = reject_accepts / len(reject_idxs) if reject_idxs else 0.0
        sweep.append(
            {
                "threshold": t,
                "n_target_commands": len(target_idxs),
                "target_accept_rate": target_rate,
                "n_reject_probes": len(reject_idxs),
                "false_accept_rate": false_rate,
                "youden_j": target_rate - false_rate,
            }
        )
    return sweep


def choose_operating_threshold(sweep: list[dict]) -> dict:
    """Documented selection rule: the candidate maximizing Youden's J
    (target_accept_rate - false_accept_rate) on the val-split sweep. Ties
    broken toward the least permissive (highest/least-negative) threshold
    among the tied candidates, since a stricter cutoff is the safer
    default at equal J."""
    best_j = max(row["youden_j"] for row in sweep)
    tied = [row for row in sweep if row["youden_j"] == best_j]
    return max(tied, key=lambda row: row["threshold"])


def confusion_counts(results: list[RowResult], threshold: float) -> dict[str, dict[str, int]]:
    """`target_commands` rows only: true INTENT_PHRASES label ->
    {predicted-intent-or-"REJECTED": count}, at the given threshold."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.bucket != TARGET_BUCKET:
            continue
        predicted = r.intent if _accepted(r, threshold) else "REJECTED"
        counts[r.label][predicted] += 1
    return {label: dict(preds) for label, preds in counts.items()}


def false_accept_stats(results: list[RowResult], threshold: float, bucket: str) -> dict:
    rows = [r for r in results if r.bucket == bucket]
    if not rows:
        return {"n": 0, "false_accepts": 0, "rate": None}
    false_accepts = sum(1 for r in rows if _accepted(r, threshold))
    return {"n": len(rows), "false_accepts": false_accepts, "rate": false_accepts / len(rows)}


def evaluate_grammar(
    model,
    feature_extractor: LogMelFeatureExtractor,
    val_dataset: VCMDataset,
    test_dataset: VCMDataset,
    grammar: Grammar,
    grammar_label: str,
    beam_width: int,
    device: str | torch.device,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> dict:
    val_results = decode_split(model, feature_extractor, val_dataset, grammar, beam_width, device)
    sweep = sweep_thresholds(val_results, threshold_grid)
    chosen = choose_operating_threshold(sweep)
    threshold = chosen["threshold"]

    test_results = decode_split(model, feature_extractor, test_dataset, grammar, beam_width, device)
    confusion = confusion_counts(test_results, threshold)

    target_rows = [r for r in test_results if r.bucket == TARGET_BUCKET]
    n_target = len(target_rows)
    n_accepted = sum(1 for r in target_rows if _accepted(r, threshold))
    n_exact_correct = sum(
        1 for r in target_rows if _accepted(r, threshold) and r.intent == r.label
    )

    return {
        "grammar": grammar_label,
        "threshold_sweep_on_val": sweep,
        "chosen_operating_threshold": threshold,
        "chosen_operating_point_val_stats": chosen,
        "test_split": {
            "n_target_commands": n_target,
            "n_accepted": n_accepted,
            "n_exact_correct": n_exact_correct,
            "accept_rate": n_accepted / n_target if n_target else None,
            "exact_accuracy": n_exact_correct / n_target if n_target else None,
            "per_intent_confusion": confusion,
            "false_accept_rate_babble": false_accept_stats(test_results, threshold, "babble"),
            "false_accept_rate_silence": false_accept_stats(test_results, threshold, "silence"),
        },
    }


# ---------------------------------------------------------------------------
# Optional slot-eval-set section (Task 07's manifest, may not exist -- must
# not hard-fail if it doesn't).
# ---------------------------------------------------------------------------


def _load_slot_eval_rows(manifest_path: Path) -> list[dict]:
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def evaluate_slot_eval_set(
    model,
    feature_extractor: LogMelFeatureExtractor,
    manifest_path: Path,
    grammar: Grammar,
    grammar_label: str,
    threshold: float,
    beam_width: int,
    device: str | torch.device,
) -> dict:
    """Runs Task 07's slot-eval-set clips (if present) through the same
    pipeline, at the already-chosen (`test`-split-reported) operating
    threshold for `grammar_label`. Framed per this ticket's Action section:
    quantifies the KNOWN training gap (the acoustic model saw zero
    slot-word audio during training) -- NOT proof that slot extraction
    works on real audio; `vcm.grammar`'s own synthetic-posterior tests are
    the correctness proof for the grammar/decoder side of slot
    extraction."""
    import torchaudio

    audio_root = manifest_path.parent
    rows = _load_slot_eval_rows(manifest_path)

    per_row: list[dict] = []
    for row in rows:
        wav_path = audio_root / row["path"]
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != int(row.get("sample_rate", sample_rate)):
            waveform = torchaudio.functional.resample(waveform, sample_rate, int(row["sample_rate"]))

        decoded = infer_waveform(
            model, feature_extractor, waveform, grammar, threshold=threshold, beam_width=beam_width, device=device
        )
        per_row.append(
            {
                "filename": row.get("filename"),
                "true_intent": row.get("label"),
                "predicted_intent": decoded.intent,
                "predicted_slots": decoded.slots,
                "text": decoded.text,
                "confidence": None if decoded.no_match else decoded.confidence,
                "accepted": not decoded.no_match,
                "intent_correct": (not decoded.no_match) and decoded.intent == row.get("label"),
            }
        )

    n = len(per_row)
    n_accepted = sum(1 for r in per_row if r["accepted"])
    n_intent_correct = sum(1 for r in per_row if r["intent_correct"])
    return {
        "grammar": grammar_label,
        "threshold_used": threshold,
        "framing": (
            "Quantifies the KNOWN slot-vocabulary training gap: vcm.train's "
            "acoustic model was trained exclusively on the 20 fixed "
            "INTENT_PHRASES canonical command phrases and saw zero "
            "slot-word audio (numbers, artist names, persona names, time "
            "units, am/pm) at training time. Low accept/intent-correct "
            "rates here are an EXPECTED consequence of that training-data "
            "gap, not a grammar/decoder defect -- the grammar/decoder's "
            "own correctness for slot extraction is proven separately by "
            "vcm.grammar's synthetic-posterior unit tests, not by this "
            "real-audio probe."
        ),
        "n_clips": n,
        "n_accepted": n_accepted,
        "n_intent_correct": n_intent_correct,
        "accept_rate": n_accepted / n if n else None,
        "intent_correct_rate": n_intent_correct / n if n else None,
        "rows": per_row,
    }


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# VCM toy CTC model -- evaluation report")
    lines.append("")
    lines.append(f"License: {report['license_note']}")
    lines.append("")
    lines.append(
        f"Checkpoint: `{report['checkpoint_path']}` "
        f"(preset={report['checkpoint_meta'].get('preset')}, "
        f"epoch={report['checkpoint_meta'].get('epoch')}, "
        f"val_loss={report['checkpoint_meta'].get('val_loss')})"
    )
    lines.append(f"Manifest: `{report['manifest_path']}`")
    lines.append(f"Device: `{report['device']}`  Beam width: {report['beam_width']}")
    lines.append("")
    lines.append(
        "Threshold operating points below are chosen SEPARATELY per grammar "
        "by sweeping the candidate grid on the manifest's `val` split "
        "(never `test`) and maximizing Youden's J "
        "(target_accept_rate - false_accept_rate on babble+silence); the "
        "full sweep table is included so the choice is auditable."
    )
    lines.append("")

    for section in report["grammar_sections"]:
        lines.append(f"## {section['grammar']} results")
        lines.append("")
        if section["grammar"] == "SPEC_GRAMMAR":
            lines.append(
                "> **Grammar-coverage note:** `SPEC_GRAMMAR` is the verbatim BNF "
                "grammar. 12 of the 20 `INTENT_PHRASES` intents' canonical "
                "dataset phrases are not accepted by it -- but only 2 of "
                "those (MESSAGE, SET_REMINDER) have no SPEC_GRAMMAR rule at "
                "all; the other 10 (ALARM, TIMER, CALL, TIME, WEATHER, "
                "LIST_REMINDERS, DIM_UP, DIM_DOWN, and the temperature "
                "rule's two intents) DO have a `$CMD_*` BNF rule, it just "
                "requires a slot or different phrasing than the dataset's "
                "bare canonical phrase provides. Those 12 intents are "
                "therefore EXPECTED to report `REJECTED` for essentially "
                "all their clips in this section, by grammar construction, "
                "not because the model failed to recognize them (see the "
                "TOY_GRAMMAR section below for the same clips with the "
                "toy-only bare-phrase aliases available)."
            )
            lines.append("")

        lines.append(
            f"Chosen operating threshold (val sweep): "
            f"**{section['chosen_operating_threshold']}** "
            f"(val target_accept_rate="
            f"{section['chosen_operating_point_val_stats']['target_accept_rate']:.3f}, "
            f"val false_accept_rate="
            f"{section['chosen_operating_point_val_stats']['false_accept_rate']:.3f})"
        )
        lines.append("")
        lines.append("### Val-split threshold sweep")
        lines.append("")
        lines.append("| threshold | val target_accept_rate | val false_accept_rate | Youden J |")
        lines.append("|---|---|---|---|")
        for row in section["threshold_sweep_on_val"]:
            lines.append(
                f"| {row['threshold']} | {row['target_accept_rate']:.3f} | "
                f"{row['false_accept_rate']:.3f} | {row['youden_j']:.3f} |"
            )
        lines.append("")

        ts = section["test_split"]
        lines.append("### Test-split results at the chosen threshold")
        lines.append("")
        lines.append(
            f"- `target_commands`: {ts['n_target_commands']} clips, "
            f"{ts['n_accepted']} accepted "
            f"({ts['accept_rate']:.3f} accept rate), "
            f"{ts['n_exact_correct']} exact-intent-correct "
            f"({ts['exact_accuracy']:.3f} exact accuracy)"
        )
        fab = ts["false_accept_rate_babble"]
        fas = ts["false_accept_rate_silence"]
        fab_rate_str = f"({fab['rate']:.3f})" if fab["rate"] is not None else "(n/a)"
        fas_rate_str = f"({fas['rate']:.3f})" if fas["rate"] is not None else "(n/a)"
        lines.append(
            f"- `babble` false-accept rate: {fab['false_accepts']}/{fab['n']} "
            f"{fab_rate_str} -- includes filipino_speech_corpus rows per decision (B)"
        )
        lines.append(f"- `silence` false-accept rate: {fas['false_accepts']}/{fas['n']} {fas_rate_str}")
        lines.append(
            "- These false-accept rates hold **at the chosen operating "
            "threshold only** -- see the val-split sweep table above for "
            "how false-accept rate degrades at looser (more permissive) "
            "thresholds; it is not an unconditional property of the model."
        )
        lines.append("")
        lines.append("### Per-intent accept/confusion counts (test split)")
        lines.append("")
        lines.append("| true intent | predicted distribution |")
        lines.append("|---|---|")
        for label in sorted(INTENT_PHRASES):
            preds = ts["per_intent_confusion"].get(label, {})
            if not preds:
                lines.append(f"| {label} | (no test-split clips) |")
                continue
            pred_str = ", ".join(f"{k}={v}" for k, v in sorted(preds.items()))
            lines.append(f"| {label} | {pred_str} |")
        lines.append("")

    if report.get("slot_eval_sections"):
        lines.append("## Slot-eval-set (Task 07) results")
        lines.append("")
        lines.append(
            "> Framing: quantifies the KNOWN slot-vocabulary training gap "
            "(the acoustic model saw zero slot-word audio during "
            "training) -- not proof that slot extraction works on real "
            "audio. See each section's own `framing` text below."
        )
        lines.append("")
        for section in report["slot_eval_sections"]:
            lines.append(f"### {section['grammar']}")
            lines.append("")
            lines.append(section["framing"])
            lines.append("")
            lines.append(
                f"- {section['n_clips']} clips, {section['n_accepted']} accepted "
                f"({section['accept_rate']}), {section['n_intent_correct']} "
                f"intent-correct ({section['intent_correct_rate']})"
            )
            lines.append("")
    elif report.get("slot_eval_skipped_reason"):
        lines.append("## Slot-eval-set (Task 07) results")
        lines.append("")
        lines.append(f"Skipped: {report['slot_eval_skipped_reason']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_report(
    checkpoint_path: Path,
    checkpoint_meta: dict,
    manifest_path: Path,
    device: str,
    beam_width: int,
    grammar_sections: list[dict],
    slot_eval_sections: list[dict] | None,
    slot_eval_skipped_reason: str | None,
) -> dict:
    return {
        "license_note": LICENSE_NOTE,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_meta": {
            k: v
            for k, v in checkpoint_meta.items()
            if k != "model_state_dict"
        },
        "manifest_path": str(manifest_path),
        "device": device,
        "beam_width": beam_width,
        "grammar_sections": grammar_sections,
        "slot_eval_sections": slot_eval_sections,
        "slot_eval_skipped_reason": slot_eval_skipped_reason,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument(
        "--slot-eval-manifest",
        type=Path,
        default=DEFAULT_SLOT_EVAL_MANIFEST,
        help="Task 07's optional slot-eval-set manifest; skipped gracefully if missing",
    )
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default=None,
        help="comma-separated float overrides for the val-split sweep grid",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    threshold_grid = DEFAULT_THRESHOLD_GRID
    if args.threshold_grid:
        threshold_grid = tuple(float(x) for x in args.threshold_grid.split(","))

    model, checkpoint_meta = load_checkpoint(args.checkpoint, device=args.device)
    feature_extractor = LogMelFeatureExtractor()

    val_dataset = VCMDataset(args.manifest, split="val", augmenter=None)
    test_dataset = VCMDataset(args.manifest, split="test", augmenter=None)

    grammar_sections = []
    for grammar, label in [(SPEC_GRAMMAR, "SPEC_GRAMMAR"), (TOY_GRAMMAR, "TOY_GRAMMAR")]:
        print(f"evaluating {label} ...")
        section = evaluate_grammar(
            model,
            feature_extractor,
            val_dataset,
            test_dataset,
            grammar,
            label,
            args.beam_width,
            args.device,
            threshold_grid,
        )
        grammar_sections.append(section)
        ts = section["test_split"]
        print(
            f"  {label}: threshold={section['chosen_operating_threshold']} "
            f"accept_rate={ts['accept_rate']:.3f} exact_accuracy={ts['exact_accuracy']:.3f} "
            f"babble_far={ts['false_accept_rate_babble']['rate']:.3f} "
            f"silence_far={ts['false_accept_rate_silence']['rate']}"
        )

    slot_eval_sections: list[dict] | None = None
    slot_eval_skipped_reason: str | None = None
    if not args.slot_eval_manifest.exists():
        slot_eval_skipped_reason = f"{args.slot_eval_manifest} not found (Task 07 clips may not exist yet)"
        print(f"slot-eval-set skipped: {slot_eval_skipped_reason}")
    else:
        try:
            slot_eval_sections = []
            for grammar, label, section in zip(
                [SPEC_GRAMMAR, TOY_GRAMMAR], ["SPEC_GRAMMAR", "TOY_GRAMMAR"], grammar_sections
            ):
                slot_eval_sections.append(
                    evaluate_slot_eval_set(
                        model,
                        feature_extractor,
                        args.slot_eval_manifest,
                        grammar,
                        label,
                        section["chosen_operating_threshold"],
                        args.beam_width,
                        args.device,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - optional section, must not hard-fail
            slot_eval_sections = None
            slot_eval_skipped_reason = f"slot-eval-set present but evaluation failed: {exc!r}"
            print(f"slot-eval-set evaluation failed (non-fatal): {exc!r}")

    report = build_report(
        args.checkpoint,
        checkpoint_meta,
        args.manifest,
        args.device,
        args.beam_width,
        grammar_sections,
        slot_eval_sections,
        slot_eval_skipped_reason,
    )

    json_path = args.out_dir / "eval_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    md_path = args.out_dir / "eval_report.md"
    md_path.write_text(render_markdown(report))

    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
