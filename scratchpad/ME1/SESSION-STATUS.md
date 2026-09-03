# ME1 session status — handoff memo

**RESOLVED — ME1 is fully complete.** This memo was written mid-session as a
handoff snapshot; everything under "What's left to build" below was finished
afterward. See `LEARNINGS-AND-GAPS.md` in this same directory for the actual
closeout doc (final decisions, learnings, gaps vs. the original plan). Kept
below as-is for historical record rather than rewritten. Notably, patch
extraction was further revised *after* this memo: `Tensor.unfold` (mentioned
below) was replaced with hand-built window indices
(`torch.arange`+broadcasting) + plain indexing, following a stricter
TISO-compliance audit — see `layers.py` and `LEARNINGS-AND-GAPS.md` for why.
The three quiz questions listed below are also stale for the same reason:
question 1 (`.unfold(2)` vs `.unfold(3)` ordering) no longer applies since
that code doesn't exist anymore; questions 2 and 3 still apply to the
current implementation unchanged.

Written because the student is having network/power issues and asked to pause
live collaboration; capturing full state here (on the `backup/scratchpad-and-plan`
branch) before continuing solo on the implementation, so nothing is lost if this
session or environment drops.

## Where things stand

We were mid-way through a Socratic design session for ME1 (3-layer CNN on MNIST,
all model layers implemented via einops/einsum, PyTorch tensors only — no
`nn.Conv2d`/`nn.Linear`/etc.). Environment and data loading are done and verified.
The two hardest layers (conv2d, done via manual patch extraction + einsum) are
written and numerically verified against PyTorch's own reference ops. Model
assembly, training, evaluation, visualization, and the final notebook are not yet
built — those are what happens next, done solo per the student's go-ahead, with
me still flagging any real judgment calls rather than deciding everything
unilaterally.

## Decisions locked in so far (all with the student's explicit sign-off)

1. **"3-layer CNN" = 3 layers with learnable parameters**: Conv1 → Conv2 → Linear
   (classifier head). Pooling/ReLU are operations, not counted layers. This
   diverged from the student's original plan (which assumed 3 conv layers +
   a separate dense network) — reconciled explicitly, decision: go with
   2 conv + 1 linear = 3 total. **Must be documented as an explicit callout in
   the final notebook**, per the student's instruction.
2. **Scope of "torch tensors only, no CNN/MLP"**: hand-roll only the *forward*
   math of conv/pool/linear. `torch.autograd`/`.backward()`, `torch.optim`
   (Adam), and `F.cross_entropy` for the loss stay as PyTorch built-ins — they're
   training infrastructure, not the CNN/MLP itself. `nn.Module`/`nn.Parameter`
   may be used, but strictly as parameter *storage* — `forward()` must contain
   only our own einsum/einops math, never `nn.Conv2d`/`nn.Linear`/`nn.Sequential`.
3. **Package management**: conda, not pip. One env per exercise. ME1's env is
   `ai222-me1`, created via
   `conda env create --prefix ~/conda-envs/ai222-me1 -f ME1/environment.yml`
   (spec committed at `ME1/environment.yml` on master). Kept fully user-local:
   `CONDA_PKGS_DIRS=$HOME/.conda/pkgs` during creation, `--prefix` outside any
   shared HPC path — nothing touches `/opt/miniconda3`.
   - **Gotcha hit and fixed**: default channels (`main`, `r`) required
     `conda tos accept --override-channels --channel <url>` before `conda env
     create` would proceed non-interactively. Already accepted for this user.
   - Activation note: `conda activate` fails non-interactively without
     `conda init` (never ran `conda init`, deliberately, to avoid mutating the
     student's shell rc files without asking). Working pattern instead:
     `module load conda` once per shell for the `conda` CLI itself, but for
     running actual scripts, invoke the env's python directly:
     `~/conda-envs/ai222-me1/bin/python <script>.py`.
4. **GPU**: this HPC node has 8 shared, idle A100s. Student explicitly chose
   **CUDA-enabled** PyTorch (not CPU-only) for the env, even though a model
   this small doesn't need it. `environment.yml` pins `pytorch-cuda=12.4` via
   the `pytorch`/`nvidia` channels. Verified working:
   `torch.cuda.is_available()` → `True`.
5. **Data loading**: use `torchvision.datasets.MNIST` (fetch + decode), *not* a
   hand-written raw IDX-file parser — student's call, since raw parsing is pure
   I/O, unrelated to the CNN/einops point of the exercise. Wrapped in a small
   `MNISTData` class (student's explicit request: "lightweight class... general
   loading func and the func to get random images") — see
   `scratchpad/ME1/data.py`. Normalization: `ToTensor()` (÷255) then
   `Normalize(mean=0.1307, std=0.3081)` (MNIST's known stats) — not treated as
   a design point worth debating, standard practice.
6. **Architecture** (final, both conv layers use the *same* config —
   student explicitly rejected varying kernel/stride per layer as
   "anti-pattern" relative to VGG-style consistency):

   | # | Layer | Config | Output shape |
   |---|---|---|---|
   | — | input | — | `(B,1,28,28)` |
   | 1 | Conv1 | 3×3, 1→8 ch, stride 1, padding 1, then ReLU, then 2×2 maxpool | `(B,8,14,14)` |
   | 2 | Conv2 | 3×3, 8→16 ch, stride 1, padding 1, then ReLU, then 2×2 maxpool | `(B,16,7,7)` |
   | — | flatten | `7×7×16 → 784` | `(B,784)` |
   | 3 | Linear | `784→10`, raw logits (no activation) | `(B,10)` |

   - Kernel size 3×3 chosen per lecture guidance (2–4px) and VGG-style
     precedent (stack small kernels rather than vary size).
   - Channel counts (8, 16) are **precedent-based, not derived**: directly
     follows LeNet-5's channel progression for this exact dataset (6, 16
     originally) — explained to student as "heuristic + historical precedent,
     not a formula," with the general heuristic noted (channels tend to
     increase as spatial size shrinks going deeper).
   - Padding formula taught and confirmed understood by student:
     `H_out = floor((H_in + 2·padding − kernel_size)/stride) + 1`; "same"
     padding for odd kernels: `padding = (kernel_size−1)/2`. Also confirmed:
     `F.cross_entropy` operates on **raw logits**, not a layer, out of scope for
     the einops/einsum requirement — do not hand-roll softmax/NLL.

## Conv2d implementation — the actual core of the exercise

Two more scope calls, both with explicit student sign-off, both **important to
preserve** if anyone re-derives this without this memo:

- **Patch extraction**: uses `Tensor.unfold(dim, size, step)` (a *generic*
  sliding-window primitive, not conv-specific) applied once along H then once
  along W, then reshaped via `einops.rearrange` — **deliberately not**
  `torch.nn.functional.unfold`, because that function exists specifically to
  support building convolutions and the student's guardrail is "the logic
  should be implemented by us." `F.pad` for padding was treated as fine
  (generic tensor op, not conv-specific logic).
- **Einsum style**: use `einops.einsum` with full descriptive axis names
  (e.g. `"batch height width patch, out_channels patch -> batch out_channels height width"`),
  **not** raw `torch.einsum` with single-letter axes — student explicitly asked
  for this after finding single-letter axes ("b o h w") unreadable. This
  applies throughout `layers.py` (rearrange/reduce/einsum all use full names).

Implementation lives in `scratchpad/ME1/layers.py`:
`extract_patches`, `conv2d`, `max_pool2d`, `linear`. All three
non-trivial ops (`conv2d`, `max_pool2d`, `linear`) were verified with
`torch.allclose` against PyTorch's own `F.conv2d` / `F.max_pool2d` / `F.linear`
using identical weights — all matched exactly. This check was called out
explicitly to the student as the highest-value test in the whole exercise,
since a shape-correct-but-numerically-wrong bug in the einsum contraction
would otherwise be silent.

`scratchpad/ME1/data.py` has `MNISTData` — `get_loaders()` and
`sample_random_test_images(n)`, both verified working end-to-end (real MNIST
data downloaded successfully, batch shapes confirmed correct).

## Quiz question(s) left unanswered when we paused

**RESOLVED** — walked through all three in a later session, re-derived
against the current `Tensor.unfold`-free implementation. Full answers with
worked numeric examples in `patch-extraction-primer.md` §7 (and §§1–4 for
the underlying fancy-indexing mechanics). Quick summary:

1. Whether `.unfold`-style order matters is moot now (`Tensor.unfold` is
   gone), but the underlying question carries over to `row_index`/
   `col_index`: order doesn't matter in the "which line runs first" sense —
   what matters is each index hitting the axis it was derived from. See
   primer §4 for a non-square case that crashes if you get this wrong.
2. `bias[None, :, None, None]` puts `bias`'s real axis in slot 1, matching
   `out`'s `out_channels` axis. `bias[None, None, None, :]` would target
   `width` instead — crashes outright in the real model, or silently adds
   a per-column offset if sizes ever coincided. Primer §7.
3. A shape-only check misses any bug where two axes being paired/merged/
   reordered coincidentally have matching sizes — right shape, silently
   wrong values (mismatched einsum flatten order, transposed output axes).
   Primer §3 and §7.

## What's left to build (proceeding solo from here, per student's go-ahead)

1. `model.py` — `EinopsCNN(nn.Module)` wrapping `layers.py`'s functions,
   `nn.Parameter` weights/biases, Kaiming/He init for conv weights, zeros for
   biases, straight-line `forward()`.
2. `train.py` — Adam (lr=1e-3), 5 epochs (fixed by the task spec), per-epoch
   train loss/accuracy, then full test-set accuracy after training
   (`torch.no_grad()`).
3. `visualize.py` — sample 16 random test images via `MNISTData`, run
   inference, 4×4 matplotlib grid, titles `gt=X pred=Y`, colored by
   correct/incorrect.
4. Final notebook at `ME1/machine_exercise_1.ipynb` (**not** gitignored —
   the actual submission), narrated top-to-bottom, core einops/einsum layer
   code inlined directly (not hidden behind a module import, since that code
   is the point of the grade), executed end-to-end before saving so committed
   outputs (test accuracy, the 4×4 grid image) are real. Must include the
   explicit "why 3 layers = 2 conv + 1 linear" callout the student asked for.
5. Per the TISO instructions (`scratchpad/ME1/tiso-reference.md`, instruction
   4): a separate learnings/decisions/gaps recap doc, written to
   `scratchpad/ME1/` (not the notebook) — main learnings, implementation
   decisions and why they're defensible, and a recap of gaps between the
   student's original plan (`scratchpad/ME1/me1-anthony-implementation-plan.md`)
   and what we actually built/learned.
6. Commit the finished notebook (+ anything else needed) to `master`, push.
   Periodically re-sync this `backup/scratchpad-and-plan` branch with
   `git add -f scratchpad/ claude-independent-plan/` at reasonable checkpoints
   while working solo, as an extra safety net.

## Files reference

- `claude-independent-plan/ME1-einops-cnn-plan.md` — Claude's solo pre-implementation
  plan (gitignored on master; this is the "if I built it alone" reference).
- `scratchpad/ME1/tiso-reference.md` — the original task spec, verbatim, as the
  student pasted it (gitignored on master).
- `scratchpad/ME1/me1-anthony-implementation-plan.md` — the student's own
  initial plan, with their original open questions (gitignored on master).
- `scratchpad/ME1/data.py`, `scratchpad/ME1/layers.py` — working, verified code
  (gitignored on master; force-tracked on this backup branch).
- `ME1/environment.yml` — committed on master (not gitignored — reproducibility
  artifact for the submission).
