# Streaming Contract

Shared contract for the `me2_voicegen.vcm.streaming` feature (Tasks
01-04: ring buffer/debounce/policy seam, audio sources, runner loop +
inference backend, CLI). Every task in this feature reads this doc
instead of having the shapes below re-pasted into its prompt. Source of
truth for each section is the module/file named; if this doc and that
module ever disagree, the module wins and this doc is stale and needs
fixing.

Task 01 (this doc's author) owns `RingBuffer`, `Debouncer`, and the
`AcceptancePolicy` seam (`ThresholdPolicy`), plus this document. Tasks
02/03/04 own their own sections below and update them as they land --
they do not need this section re-explained to them, just pointed here.

## 1. `AudioSource` protocol (owned by Task 02)

As actually implemented in `vcm/streaming/sources.py` (this section was
drafted with a `read(n_samples)` shape before Task 02 landed; the shape
below is the real, shipped contract -- update call sites to it, not the
other way around):

```python
class AudioSource(Protocol):
    is_realtime: bool

    def blocks(self) -> Iterator[np.ndarray]:
        """Yields float32 mono blocks at common.features.SAMPLE_RATE.
        Blocks are exactly the source's configured block_samples long
        except possibly the final block (file sources only -- the
        generator simply ends at end-of-stream; a realtime source's
        generator never ends under normal operation, it just blocks
        until audio is available)."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> "AudioSource": ...
    def __exit__(self, *exc_info) -> None: ...
```

Three implementations ship: `RawPcmStreamSource(stream, block_samples)`
(wraps any `.read(n) -> bytes` object -- a subprocess stdout pipe or an
in-memory buffer -- of raw S16_LE mono 16kHz PCM), `WavFileSource(path,
block_samples, realtime=False)` (`is_realtime=False`, loads/downmixes/
resamples via `torchaudio`, same idiom as
`vcm.evaluate.evaluate_slot_eval_set`), and `MicrophoneSource`/
`open_microphone_source(command, block_samples)` (`is_realtime=True`,
spawns `command` -- always a parsed argv list, `shell=False` -- and
wraps its stdout in a `RawPcmStreamSource`; raises
`MicrophoneUnavailableError` if the process is missing or exits on its
own before/without producing audio, but never on a returncode produced
by our *own* `close()`-time `terminate()`).

`is_realtime` selects between two runner-loop modes, because a live
microphone and a pre-recorded file need fundamentally different
backpressure behavior:

- **`is_realtime=True` (mic):** capture happens on its own thread,
  continuously, regardless of whether inference keeps up. If the
  inference loop falls behind, it **skips-and-counts** -- it drains the
  capture queue/buffer to the most recent audio and records how many
  chunks it dropped, rather than blocking capture (an audio driver whose
  callback blocks on inference risks ALSA buffer underruns -- see
  `docs/raw_requirements/streaming_approach.md`'s "Audio Buffer
  Underrun" pitfall). Real time keeps moving whether or not the model
  keeps up; the buffer always reflects "now," not "whenever inference
  last got around to it."
- **`is_realtime=False` (file):** synchronous lockstep, no drops. The
  runner reads exactly as fast as it can process, and every sample in
  the file is guaranteed to reach the ring buffer and be evaluated at
  least once at some stride offset -- reproducibility for tests/eval
  requires this; a file replay that silently drops chunks under load
  would make results depend on host CPU speed.

## 2. `StreamingConfig` (owned by Task 03/04, fields listed here as they are introduced)

A single config object threading window/stride/refractory/threshold and
runner-loop knobs through the CLI, the runner loop, and the
`SlidingWindowPipeline`-equivalent streaming stack, so no task invents a
second, slightly different way to plumb the same handful of numbers.

Landed by Task 03 (`vcm.streaming.config`), frozen dataclass, precedence
`dataclass defaults < JSON config (StreamingConfig.from_json) < explicit CLI
flags (StreamingConfig.merge)`:

- `model: str = "optionc"` -- `MODEL_REGISTRY` name, run-dir path, or direct
  checkpoint/`.onnx` file path (`resolve_model`).
- `backend: str = "onnx"` (`"onnx"` | `"torch"`)
- `onnx_variant: str = "fp32"` (`"fp32"` | `"int8"`)
- `ort_threads: int = 1`
- `policy: str = "threshold"` -- `POLICY_REGISTRY` name (`resolve_policy`).
- `grammar: str = "optionb"` -- `vcm.evaluate.GRAMMAR_REGISTRY` key, reused
  rather than a second registry (`resolve_grammar`).
- `threshold: Optional[float] = None` -- resolved from the resolved model's
  run dir's `metadata/eval_report.json` (`resolve_threshold`) when `None`;
  an explicit value always wins.
- `window_s: float = 2.5`, `stride_s: float = 0.25`,
  `refractory_s: float = 1.5`, `beam_width: int = 25`, `block_s: float =
  0.064`
- `device: str = "cpu"`
- `source: str = "mic"` (`"mic"` | a wav path)
- `realtime: bool = True`
- `mic_command: Optional[str] = None`
- `listen_for: Optional[float] = None`
- `log_all_windows: bool = False`

`MODEL_REGISTRY = {"optionc": out/vcm/optionb-optionc, "default":
out/vcm/optionb}` (`optionc` is the config default -- it beats `default` on
every eval metric, see this ticket's Execution Log). `POLICY_REGISTRY =
{"threshold": ThresholdPolicy}`.

**Security note (both `--model` code paths are trust boundaries):**
`resolve_model`'s result is later fed to either `torch.load` (pickle RCE) or
ONNX Runtime's model parser -- neither is safe on an untrusted file. This is
deliberately NOT run through `fetch_dataset.resolve_under()`-style path
containment (see `config.py`'s and `backends.py`'s module docstrings for
why); only point `--model` at a checkpoint/`.onnx` file you produced
yourself or otherwise trust.

**Security note (`--config` JSON is a trust boundary too, not inert data):**
`StreamingConfig.from_json` validates field names AND, as of the R3-1
hardening pass, per-field types/choices (mirroring `__main__`'s argparse
`choices=`/`type=` constraints) -- but it does not and cannot judge whether
the *values* themselves are safe. A JSON config file can set `model` (the
trust boundary above) or `mic_command` (the argv `MicrophoneSource` spawns
via `subprocess.Popen`, `shell=False` -- see `vcm.streaming.sources`) just
as freely as the equivalent CLI flags can. Treat a `--config` file as
exactly as trusted as the command line itself, never as passive/inert
configuration data safe to copy from an untrusted source.

**Security note (`weights_only=True` is a mitigation, not a guarantee):**
`load_checkpoint`'s `weights_only=True` path defends against naive/accidental
malicious pickles, but this project's pinned `torch==2.3.1` is within the
affected range of CVE-2025-32434, a known bypass of `weights_only=True` on
torch <= 2.5.1 (fixed in 2.6.0). Upgrading torch is out of scope for this
feature (a repo-wide pinned CUDA stack) and is tracked as a separate
backlog concern. Additionally, `load_checkpoint`'s `weights_only=True`
failure path does NOT automatically retry with `weights_only=False` -- a
checkpoint failing the allow-list is itself the attack signature this
mitigation exists to catch, not a benign edge case, so it is refused
outright unless the caller also explicitly passes `allow_unsafe_load=True`.
`TorchBackend` always calls with `allow_unsafe_load=False`.

## 3. `InferenceBackend` adapter interface (owned by Task 03)

Abstracts "torch checkpoint" vs. "ONNX export" behind one call shape so
the runner loop and CLI don't care which backend produced a window's
`logp`:

```python
class InferenceBackend(Protocol):
    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """(samples,) float32 waveform at SAMPLE_RATE -> (T, alphabet_size)
        log-posterior array -- the same contract as
        vcm.pipeline.logp_for_waveform / docs/VCM-CONTRACT.md section 7."""
        ...
```

Two implementations ship (`vcm/streaming/backends.py`):

- `OnnxBackend(model_path, ort_threads=1)`: builds an
  `onnxruntime.InferenceSession` per `benchmark._make_session`'s pattern
  (`SessionOptions`, `intra_op_num_threads=ort_threads`,
  `inter_op_num_threads=1`, `providers=["CPUExecutionProvider"]` -- CPU
  only, the installed ORT wheel offers no other real provider).
  `logp_for_waveform` runs the shared `LogMelFeatureExtractor` on the
  waveform, feeds `{"features": (1, 40, T) float32}` to `session.run`
  (`export_onnx.onnx_vs_pytorch_logits`'s call shape), and applies
  **log-softmax in numpy** to the returned `(1, T, 29)` logits -- ONNX
  emits logits, not log-probs, unlike the torch side which already does
  `log_softmax` internally. Verified within 1e-2 max-abs-diff of
  `TorchBackend` on the checked-in optionc fp32 export
  (`tests/test_vcm_streaming_backends.py::
  test_onnx_backend_matches_torch_backend_on_real_optionc_artifacts`,
  `@pytest.mark.slow`). Preset metadata for a startup banner is NOT carried
  by the `.onnx` file itself -- read it from the run dir's
  `eval_report.json`/`checkpoint_meta` instead.
- `TorchBackend(checkpoint_path, device="cpu")`: wraps
  `vcm.pipeline.load_checkpoint(..., weights_only=True)` +
  `vcm.pipeline.logp_for_waveform`. `weights_only=True` is confirmed to
  load both real checked-in checkpoints (non-tensor metadata included) as
  of torch 2.3.1; `load_checkpoint` falls back to `weights_only=False` with
  an explicit stderr warning if a future checkpoint's metadata trips the
  allow-list, rather than silently regressing.

Cold start is non-trivial for **either** backend -- see section 6, "the
ONNX backend does NOT avoid this."

**Security note:** both backends parse untrusted/arbitrary-file-format
input at construction (`torch.load` pickle deserialization / ONNX Runtime's
model parser) -- see `backends.py`'s and `config.py`'s module docstrings
for the full trust-boundary note and why path-containment is not the right
mitigation here.

## 4. The `AcceptancePolicy` seam (owned by Task 01)

### Protocol signatures

```python
# vcm/streaming/policy.py
@dataclass(frozen=True)
class WindowObservation:
    window_index: int
    samples_seen: int
    result: DecodeResult        # decoded at threshold=-inf

@dataclass(frozen=True)
class PolicyDecision:
    accept: bool
    reason: str                 # surfaced by --log-all-windows

class AcceptancePolicy(Protocol):
    def observe(self, obs: WindowObservation) -> PolicyDecision: ...
    def reset(self) -> None: ...

class ThresholdPolicy:          # the ONE shipped implementation
    """accept iff result.intent is not None and result.confidence >= threshold"""
```

`WindowObservation`/`PolicyDecision` are frozen dataclasses. `observe`/
`reset` are stateful methods, not a pure function of one `DecodeResult`,
because named future cases (multi-window smoothing, N-consecutive
voting, active-region scoring) all need rolling history across
observations; this costs `ThresholdPolicy` nothing since it ignores
history entirely (it's a pure function of the one `WindowObservation`
it's given, it just happens to satisfy the stateful protocol). Likewise
`PolicyDecision` is a dataclass, not a bare `bool`, so a future policy
that needs to influence emission (e.g. a per-intent refractory override)
can gain a field without changing the protocol signature.

### Fixed pipeline order

```
logp -> decode(threshold=-inf) -> DecodeResult -> AcceptancePolicy.observe(...) -> PolicyDecision.accept -> Debouncer (refractory gate) -> TriggerEvent
```

Every window is decoded once, at the most permissive threshold; the
`AcceptancePolicy` decides whether that decode is real evidence; only
*then* does the `Debouncer` decide whether enough time has passed since
the last emission to actually fire a `TriggerEvent`. This order is
fixed -- a future policy or backend must not reorder it (e.g. gating the
decoder call itself on debounce state), because the two questions below
are deliberately independent.

### Evidence vs. emission-rate: why these are two separate seams

`AcceptancePolicy` answers "is this window real evidence?" (a function
of the model's output for this window, possibly with rolling history).
`Debouncer` answers "has enough time passed since the last emission?" (a
function of elapsed samples only, independent of *why* the previous
emission fired). These are different questions, and keeping them as two
separate, independently swappable seams means a future history-based
policy (e.g. N-consecutive-window voting) composes cleanly with
debounce-after: the policy can look back across windows to decide
"real", and the debouncer still independently rate-limits how often a
"real" verdict is allowed to become a `TriggerEvent`, without either one
needing to know the other's internals.

### The `-inf` decode rationale

Every window is decoded via `infer_waveform(..., threshold=NEG_INF)` (or
equivalent) *before* the `AcceptancePolicy` ever sees it -- the policy,
not the decoder, is what applies the operating threshold. This directly
reuses an established precedent, not a new invention:
`vcm.evaluate.decode_split` already decodes every dataset row at
`vcm.evaluate.NEG_INF_THRESHOLD` to capture each row's best-reachable
intent/confidence independent of any operating threshold, so a later
threshold sweep is pure arithmetic over cached decodes rather than
re-running the model per candidate threshold
(`vcm.evaluate.sweep_thresholds`). The streaming seam applies the same
idea for the same reason: decode once, threshold-gate via the swappable
`AcceptancePolicy` afterward, so a different policy (or the
`--log-all-windows` diagnostic below) can see what the decoder actually
found on every window, not just the windows that happened to already
clear one hardcoded threshold. `ThresholdPolicy.observe` then reduces to
exactly `vcm.evaluate._accepted`'s body: `intent is not None and
confidence >= threshold`.

The motivating case for not hardcoding the threshold into the decode
call itself: `vcm.evaluate`'s own confidence/padding-sensitivity finding
(the model's confidence score is measurably sensitive to how much
silence padding surrounds a spoken command in a fixed-length clip) means
a single held-out val-split-chosen threshold is already a compromise,
not a universal cutoff -- keeping the raw decode separate from the
accept/reject decision is what lets that decision evolve (a different
threshold, a smarter policy, a `--log-all-windows` audit trail) without
touching the decoder or re-running inference.

### A concrete future policy: mode-across-a-bounded-listening-period

Two related, real (not hypothetical) failure modes were observed testing
this feature against the live microphone on real hardware, both
consistent with the confidence/padding-sensitivity finding above:

1. **Lingering duplicate triggers on a correctly-spoken phrase.** At the
   shipped defaults (`window_s=2.5`, `stride_s=0.25`, `refractory_s=1.5`),
   a spoken phrase stays inside the sliding window for close to its full
   duration as the window advances, so it decodes correctly across
   roughly ten consecutive windows, not one. `Debouncer` only blocks a
   *second* trigger for `refractory_s` after the first; since the phrase
   can still be sitting in the window once that cooldown expires, it
   fires again on what is mechanically the same utterance. Observed
   directly in manual testing: two `TIME` triggers at `t=2.0s` and
   `t=3.5s` -- a gap of exactly `1.5s`, matching `refractory_s` to the
   decimal -- from a single lingering decode, not two independent
   utterances.
2. **False positives during fast speech/walkthroughs.** Consistent with
   the padding-sensitivity mechanism: confidence is a mean per-frame
   log-prob over the *whole* fixed window, so a short or partially-formed
   phrase is diluted by the surrounding non-speech frames in the same way
   ambient noise is, and can still land just inside the threshold.

Once a wake-word gate exists and defines a bounded "listening period"
(see section 6), the natural fix for both is a policy that consumes
every `WindowObservation` across that whole period and takes the
**mode** (most frequent decode) across it, emitting one consolidated
`TriggerEvent` for the period rather than accepting the first window
that individually clears the threshold. This is exactly the shape
`AcceptancePolicy.observe`/`reset`'s statefulness was reserved for (see
above) -- it needs rolling history across a bounded window of
observations, which `ThresholdPolicy` doesn't use but the protocol
already supports. Implementing it would be a new class in `policy.py`
plus one `POLICY_REGISTRY` line (`vcm/streaming/config.py`); it requires
no change to the runner, the decoder, or `Debouncer`. It would also
resolve failure mode 1 as a side effect -- mode-across-a-bounded-period
naturally collapses "the same phrase decoded ten times in a row" into
one answer, rather than relying on refractory timing to suppress the
repeats. Not implemented here: it is meaningless without the bounded
period a wake-word gate would define, and building it against an
arbitrary/unbounded window would be speculative in exactly the way this
feature otherwise avoids.

## 5. JSONL event schema (owned by Task 04, shape fixed here)

One JSON object per line, one line per emitted `TriggerEvent` (or, under
`--log-all-windows`, one line per *evaluated window* regardless of
accept/reject/suppress -- the diagnostic mode this schema exists to
support):

```json
{
  "event": "trigger" | "window",
  "t_seconds": 12.34,
  "window_index": 42,
  "intent": "CALL",
  "slots": {},
  "text": "call mom",
  "confidence": -0.87,
  "policy_reason": "intent='CALL' confidence=-0.87 >= threshold -1.0"
}
```

- `event`: `"trigger"` for an actually-emitted `TriggerEvent`; `"window"`
  for a non-emitted window only ever logged under `--log-all-windows`
  (rejected by policy, or accepted by policy but suppressed by the
  debouncer -- `policy_reason` and/or a suppression note distinguish
  the two).
- `t_seconds`: wall-clock-equivalent time, i.e. `samples_seen /
  common.features.SAMPLE_RATE`.
- `intent`/`slots`/`text`/`confidence`: straight from the window's
  `DecodeResult` (`None`/`{}`/`""`/`-inf`-equivalent when no grammar
  terminal was reached at all).
- `policy_reason`: `PolicyDecision.reason` verbatim.

## 6. Forward-compat note: wake-word/gate integration (not decided here)

> Cold start is non-trivial (`torch`/`torchaudio` imported at module
> level via `common/features.py`'s `LogMelFeatureExtractor` wrapping
> `torchaudio.transforms.MelSpectrogram` -- the ONNX backend does NOT
> avoid this; plus ORT session construction, `compile_grammar`'s
> 129-phrase trie build, ~0.2s `arecord` device-open). A long-running
> process's ring buffer already holds pre-gate audio vs. a freshly-spawned
> one starting empty -- BUT the planned future product UX (wake-word ->
> deliberate delay -> "Listening..." indicator + audio chime -> user cued
> to speak) is expected to mitigate cold start at the integration layer
> AND substantially neutralizes the pre-gate-audio advantage too (a user
> cued to wait won't have spoken into the pre-gate buffer). Both
> spawn-per-trigger and signal-a-running-process therefore remain viable;
> this is genuinely not decided here -- record it so a future integration
> decision has both facts, not just the risk in isolation.

## 7. Symbols this feature reuses (source of truth: the named module)

- `vcm.pipeline.WINDOW_S` / `STRIDE_S` / `DEFAULT_REFRACTORY_S`,
  `infer_waveform`, `logp_for_waveform`, `load_checkpoint`.
- `common.features.SAMPLE_RATE`, `LogMelFeatureExtractor`.
- `vcm.evaluate.decode_split`, `NEG_INF_THRESHOLD`, `_accepted`.
- `vcm.decoder.DecodeResult`, `decode`.
- `vcm.streaming.buffer.RingBuffer`, `vcm.streaming.debounce.Debouncer`,
  `vcm.streaming.policy.{WindowObservation,PolicyDecision,
  AcceptancePolicy,ThresholdPolicy}` (this feature's own new symbols,
  Task 01).
- `vcm.streaming.sources.{AudioSource,RawPcmStreamSource,WavFileSource,
  MicrophoneSource,open_microphone_source,MicrophoneUnavailableError,
  DEFAULT_MIC_COMMAND}` (this feature's own new symbols, Task 02).
