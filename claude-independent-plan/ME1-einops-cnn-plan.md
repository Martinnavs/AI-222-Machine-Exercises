# ME1: Einops/Einsum CNN for MNIST — Independent Engineering Plan

**Author:** Claude, working solo as the implementing engineer for this note.
**Purpose:** This is *my* plan, written before any collaborative/Socratic session with
the student. It is not the artifact the student learns from — it's my own scratch
architecture doc, so that when I switch into "teacher" mode later, I already know where
I'm steering the conversation and why. Kept out of git (see `.gitignore`) because it's
an implementation crutch for me, not a submission deliverable.

## 0. Ground rules I'm setting for myself

- Task requires **all layers/operations** of the CNN to be implemented with
  einops/einsum — not `torch.nn.Conv2d`, `torch.nn.MaxPool2d`, `torch.nn.Linear`, etc.
  `torch` is still fine as the tensor library and autograd engine; what's banned is
  leaning on its pre-built layer modules for the forward math.
- Scope, confirmed explicitly: hand-roll only the model's *forward* layers (conv/pool/
  linear via einsum/einops). `torch.autograd`/`.backward()`, `torch.optim` (Adam), and
  `F.cross_entropy` for the loss stay as-is — they're training infrastructure, not the
  CNN/MLP itself. `nn.Module`/`nn.Parameter` may be used, but strictly as *storage*
  (parameter registration, `.parameters()`) — `forward()` must contain only our own
  einsum/einops math, never `nn.Conv2d`/`nn.Linear`/`nn.Sequential`.
- Package management is conda, not pip — one conda env per machine exercise (e.g.
  `ai222-me1` for ME1), with a committed `environment.yml` per exercise dir so the env
  is reproducible from the repo alone.
- "3-layer CNN" is ambiguous on its own, so I'm fixing a concrete definition now:
  **3 layers with learnable parameters** — Conv1 → Conv2 → Linear(classifier head).
  Pooling and activations are operations, not "layers," in this count. I'll surface this
  as an actual decision point for the student rather than silently imposing it — one of
  the first Socratic questions — but this is what I'd ship if working alone.
- Readability matters as much as satisfying "must use einops/einsum." I will *not* wrap
  every tensor op in its own function. I'll only extract a helper where the einops
  pattern is genuinely non-obvious (patch extraction for conv, the conv contraction
  itself) or where the same block of logic repeats (conv layer used twice). Pooling via
  `einops.reduce` and flatten via `einops.rearrange` are already about as readable as
  a helper call would be, so I'd inline those with a one-line comment rather than hide
  them behind a wrapper.

## 1. Environment setup

- Conda env per exercise: `ai222-me1` for this one, defined in a committed
  `ME1/environment.yml` (env itself is local/not a git object, but the spec is
  reproducible from the repo).
- Dependencies: `python`, `pytorch` + `pytorch-cuda` (this node has 8 shared A100s, so
  CUDA-enabled per the student's call — not strictly needed for a model this small, but
  available), `torchvision` (MNIST dataset + transforms only — not its `nn` layers),
  `einops`, `matplotlib`, `jupyter`/`nbformat` (final notebook assembly), `numpy`.
- Env is created with `--prefix ~/conda-envs/ai222-me1` and `CONDA_PKGS_DIRS` pointed at
  the user's home cache — keeps everything local, no writes to shared HPC conda paths.

## 2. Data pipeline

- Use `torchvision.datasets.MNIST` with `download=True`, which already ships the
  standard 60,000/10,000 train/test split — no need to hand-roll a split.
- Transform: `ToTensor()` (scales to `[0,1]`, shape `(1,28,28)`) plus normalization
  using MNIST's known mean/std (`0.1307`, `0.3081`) — standard, defensible choice, not
  a design point worth spending Socratic time on.
- Wrap in `DataLoader` for train (shuffled, batch size 128) and test (not shuffled,
  batch size e.g. 256, larger since no backward pass).
- Sanity check: print one batch's shape and dtype before building the model. Cheap and
  catches silent shape bugs before they propagate into the einsum layers where they're
  much harder to debug.

## 3. Architecture

Input: `(B, 1, 28, 28)`.

| Layer | Op | Output shape | Params |
|---|---|---|---|
| Conv1 | 3x3 conv, 1→8 channels, stride 1, padding 1 | `(B, 8, 28, 28)` | weight `(8,1,3,3)`, bias `(8,)` |
| — | ReLU | same | — |
| — | 2x2 max pool, stride 2 | `(B, 8, 14, 14)` | — |
| Conv2 | 3x3 conv, 8→16 channels, stride 1, padding 1 | `(B, 16, 14, 14)` | weight `(16,8,3,3)`, bias `(16,)` |
| — | ReLU | same | — |
| — | 2x2 max pool, stride 2 | `(B, 16, 7, 7)` | — |
| — | flatten | `(B, 784)` | — |
| Linear | fully connected → 10 classes | `(B, 10)` | weight `(10,784)`, bias `(10,)` |

Loss: `torch.nn.functional.cross_entropy` on raw logits (combines log-softmax + NLL,
numerically stable). This is a loss function, not a model layer, so it's out of scope
for the einops/einsum requirement — I won't hand-roll softmax+NLL just to prove a
point; that would hurt readability for no requirement-driven reason.

Small channel counts (8, 16) are deliberate: MNIST is easy, and this is a
readability/understanding exercise, not an accuracy-chasing one. Keeps params low and
epochs fast.

## 4. Core einops/einsum primitives (the actual point of the exercise)

Two non-obvious operations to implement, both worth a dedicated helper because the
pattern is easy to get subtly wrong and is reused across both conv layers:

1. **`conv2d(x, weight, bias, padding)`** — implemented as "extract sliding patches,
   then contract with the kernel via einsum," i.e. an im2col-style convolution:
   - Pad spatially.
   - Use `einops.rearrange` (via unfold or manual stride tricks — deciding exact
     mechanism during implementation) to turn `(B, C_in, H, W)` into patches shaped
     `(B, H_out, W_out, C_in*kh*kw)`.
   - `einsum('b h w p, o p -> b o h w', patches, weight_flat)` to contract patches
     against the flattened kernel, producing `(B, C_out, H_out, W_out)`.
   - Add bias via broadcasting (`+ bias[None, :, None, None]`), not einsum — einsum
     for a plain broadcast-add would hurt readability for zero benefit.
2. **`linear(x, weight, bias)`** — trivial but still einsum per the task requirement:
   `einsum('b i, o i -> b o', x, weight) + bias`.

Operations that don't need a dedicated helper — inline with a short comment instead:
- **Max pool**: `einops.reduce(x, 'b c (h k1) (w k2) -> b c h w', 'max', k1=2, k2=2)`
  is already self-documenting.
- **Flatten**: `einops.rearrange(x, 'b c h w -> b (c h w)')`.
- **ReLU**: `torch.clamp(x, min=0)` or `x.relu()` — not an einops concern at all.

I will resist the urge to make a generic `Conv2dLayer` class hierarchy or a config-driven
layer builder — there are exactly two conv layers and one linear layer, hand-writing the
three layers directly is more readable than an abstraction built to support N layers
we don't have.

## 5. Model assembly

Small `nn.Module` (`EinopsCNN`) holding `nn.Parameter` tensors for each layer's weight
and bias, Kaiming/He init for conv weights (fan_in based on `C_in*kh*kw`), zeros for
biases. `forward()` calls the primitives above in sequence — reads like a straight-line
function, which is the point.

## 6. Pre-training sanity check

Before wiring up the training loop: instantiate the model, run one dummy batch through
`forward()`, assert output shape is `(B, 10)`, and diff against a `torch.nn.Conv2d` with
manually-copied weights on one input to confirm the hand-rolled conv math is *actually*
correct (not just shape-correct). This is the single highest-risk bug source in the
whole exercise — silent numerical error in the einsum contraction that still produces
the right shape. Worth an explicit correctness check, not just a shape check.

## 7. Training loop

- Optimizer: Adam, lr=1e-3 (no tuning needed for a task this easy).
- 5 epochs, as specified.
- Per epoch: iterate train loader, forward, loss, `backward()`, `optimizer.step()`,
  `zero_grad()`. Track running average train loss and train accuracy.
- Print one line per epoch (epoch #, train loss, train acc) — no need for a full
  logging framework for a 5-epoch run.

## 8. Evaluation

- After training, run a full pass over the test loader with `torch.no_grad()`.
- Compute and report overall test accuracy (this is the headline number the task asks
  for).

## 9. Qualitative visualization

- Randomly sample 16 images from the *test* set (fixed seed for reproducibility).
- Run inference, get predicted class per image.
- `matplotlib` 4x4 subplot grid: each cell shows the image plus a title
  `"gt=X pred=Y"`, colored (e.g. green/red title) by correct/incorrect so it's visually
  scannable, not just numerically correct.

## 10. Notebook assembly

- Prototype everything as plain `.py` modules first (faster iteration, easier to debug
  stack traces, works well with an agent editing files) inside the gitignored
  `scratchpad/ME1/` dev area.
- Once each piece works, assemble a clean top-to-bottom narrative notebook at
  `ME1/machine_exercise_1.ipynb` (this path is **not** gitignored — it's the actual
  submission): markdown cells explaining each stage, code cells importing from the
  scratchpad modules or with the logic inlined directly (leaning toward inlining the
  core einops layer code directly in the notebook, since the notebook *is* the
  deliverable being graded on the einops/einsum implementation — it shouldn't hide the
  interesting part behind an import from a module the grader won't see rendered).
- Execute end-to-end top-to-bottom before saving, so committed outputs (accuracy number,
  the 4x4 grid image) are real, not stale/hand-edited.

## 11. Submission

- Commit `ME1/machine_exercise_1.ipynb` (and any small supporting files that must ship
  with it) to the repo, push to `origin/master`.
- `claude-independent-plan/` and `scratchpad/` stay local-only via `.gitignore` — this
  plan and the dev-iteration code are not part of what gets graded/submitted.

## Risks / things I'd double check if implementing solo

- Off-by-one errors in patch extraction (stride/padding arithmetic) — verified via the
  correctness check in step 6, not just shape asserts.
- Forgetting `.contiguous()` after a `rearrange`/`unfold` before a view-sensitive op —
  einops generally handles this, but worth being alert to if mixing in raw `.unfold()`.
- Max pool via `einops.reduce` with `'max'` requires the spatial dims to be evenly
  divisible by the pool kernel — true here (28→14→7 all clean), but not a pattern that
  silently generalizes; would need explicit padding logic for odd input sizes.
