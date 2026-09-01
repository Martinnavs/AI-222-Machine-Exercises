# ME1 — Learnings, Implementation Decisions, and Gaps vs. the Original Plan

Written per the TISO instruction 4 detail: "document the main learnings,
implementation decisions (ideally optimal), and a recap of what I need to
know based on what we've talked about (ideally the gaps in my implem plan)
if I were to implement this myself." Kept separate from the submission
notebook (`ME1/machine_exercise_1.ipynb`), which documents the *what* and
*why* inline but isn't the place for a retrospective.

**Result:** 98.25% test accuracy after 5 epochs, 9,098 total learnable
parameters, all three hand-rolled ops (`conv2d`, `max_pool2d`, `linear`)
verified numerically identical to PyTorch's own reference implementations.

## Main learnings

1. **Convolution is "extract overlapping windows, then one big matmul" —
   not a nested loop.** The naive mental model (loop over output pixels,
   loop over filters, dot-product a window against each filter) is
   conceptually correct but never gets written as literal Python loops.
   Instead: (a) a vectorized windowing step extracts every window for every
   output location *simultaneously*, producing a "patches" tensor; (b) a
   single `einsum` contracts that whole tensor against the (flattened)
   filter weights, computing every filter's output at every location in one
   shot. Both the spatial loop and the filter loop get absorbed into tensor
   axes that get parallelized internally, rather than iterated in Python.
   **Revisited under stricter scrutiny (worth recording as its own
   lesson):** the first working version built (a) with `Tensor.unfold`. On
   a later audit ("would a strict grader hold this against the
   requirement?"), we concluded yes — extracting the sliding windows *is*
   part of convolution's actual logic (not incidental plumbing like
   padding), and `unfold`'s genericness defends its *name*, not whether the
   index arithmetic was actually done by us. Replaced it with hand-built
   window indices (`torch.arange` + broadcasting) and plain tensor
   indexing — verified numerically identical, but now with zero reliance on
   any function whose specific purpose is "extract sliding/conv windows."
   The general lesson: "is this generic enough to not count as
   pre-implemented" is a real judgment call worth re-checking adversarially,
   not settling once and moving on.
2. **"Kernel size" and "number of filters" are two independent
   hyperparameters, not one.** Kernel size is the spatial extent of the
   sliding window; number of filters is the output channel count. Easy to
   conflate them as one "size" knob when you haven't implemented a conv
   layer by hand before.
3. **Padding has an exact formula, not a guess:**
   `H_out = floor((H_in + 2·padding − kernel_size)/stride) + 1`. "Same"
   padding for stride 1 and an odd kernel is `padding = (kernel_size−1)/2`.
   This also clarified that there are two *different* "padding" questions
   people conflate: padding raw input data to a uniform size (not needed
   here — MNIST is already fixed `28×28`) vs. padding inside a convolution
   to control output spatial size (the one that actually matters for
   architecture design).
4. **A shape-correct implementation can still be numerically wrong.** The
   single highest-value check in this whole exercise was comparing our
   hand-rolled `conv2d`/`max_pool2d`/`linear` against PyTorch's own
   `F.conv2d`/`F.max_pool2d`/`F.linear` with `torch.allclose` on identical
   weights — not just asserting output shapes. A subtly wrong einsum
   contraction (e.g. a transposed axis) can easily produce the right shape
   with wrong values, and that bug would otherwise be silent.
5. **Channel/layer-width choices are precedent + heuristic, not derived.**
   There's no formula for "how many filters should layer 1 have" — you lean
   on known-working references (here, LeNet-5's channel progression for this
   exact dataset) and the general heuristic that channel count tends to
   increase as spatial resolution shrinks going deeper, then verify with a
   real training run.
6. **Einsum's real usability problem is single-letter axis names, and
   `einops.einsum` fixes it for free.** `torch.einsum('bhwp,op->bohw', ...)`
   and `einops.einsum(..., "batch height width patch, out_channels patch ->
   batch out_channels height width")` compute the *exact same thing* — the
   only difference is whether you have to hold the meaning of each letter in
   your head. Worth defaulting to the named form whenever einops is in play.
7. **`argmax` is invariant to softmax/sigmoid** (both are monotonic
   transforms), so a classifier only needs to apply one of them if you need
   actual probability *values* (calibration, thresholding, reporting
   confidence) — picking the predicted class from raw logits via `argmax`
   gives an identical answer either way. This mattered directly for a gap
   below.

## Implementation decisions (with rationale)

| Decision | Rationale |
|---|---|
| "3-layer CNN" = 3 layers with learnable params (Conv1→Conv2→Linear) | Most literal reading of "a whole CNN *model* with 3 layers"; safer if a grader is counting layers |
| Hand-roll only forward layer math; autograd/optimizer/loss stay PyTorch built-ins | Those are training infrastructure, not the CNN/MLP itself — the guardrail was "the logic should be implemented by us," referring to the model |
| `nn.Module`/`nn.Parameter` used as storage only | Gets `.parameters()`/autograd registration for free without reaching for `nn.Conv2d`/`nn.Linear` |
| `torchvision.datasets.MNIST` for fetch+decode, no hand-written IDX parser | Raw-file parsing is I/O, unrelated to the CNN/einops point of the exercise |
| Hand-built window indices (`torch.arange`+broadcasting) and plain indexing for patch extraction — not `Tensor.unfold`, and not `F.unfold` | Both `unfold` variants exist specifically to extract sliding windows — even the "generic" one hands you the index arithmetic pre-solved. Building the indices ourselves and gathering with ordinary indexing leaves no step of the windowing logic un-implemented by us |
| `einops.einsum`/`rearrange`/`reduce` with full descriptive axis names everywhere | Single-letter axes are the "einsum is arcane" complaint made real; named axes are self-documenting at zero cost |
| Same kernel size/stride/padding (3×3, stride 1, padding 1) in both conv layers; pooling does all downsampling | Standard VGG-style pattern; varying kernel/stride per layer was considered and explicitly rejected as unnecessary complexity here |
| Channel counts 8→16 | LeNet-5 precedent for this exact dataset, not derived |
| Kaiming/He init for conv+linear weights, zero-init biases | Standard for ReLU-activated networks |
| Adam, lr=1e-3, 5 epochs | 5 epochs fixed by the assignment; Adam/lr are untuned defaults since this task doesn't need tuning |
| Conda env per exercise, `--prefix` under `$HOME`, CUDA-enabled | Course convention (per-exercise envs) + shared-HPC etiquette (never write to `/opt/miniconda3`) + explicit student choice to use the node's available GPUs |
| Core layer code inlined directly in the final notebook (not imported from a module) | The notebook is what's graded — the interesting part shouldn't be hidden behind an import the grader has to go find |

## Gaps between the original plan and what we actually built

Comparing against `scratchpad/ME1/me1-anthony-implementation-plan.md`:

1. **Layer count / model size.** The original plan assumed **3 conv layers
   plus a separate dense network after** — a 4-learnable-layer model. What
   we built is 2 conv layers + 1 linear layer = 3 total. This wasn't an
   error in the original plan so much as an unresolved ambiguity in "3-layer
   CNN" itself — worth resolving explicitly *before* writing code next time,
   since it changes the whole shape of the network.
2. **Sigmoid vs. softmax/logits for the classifier output.** The original
   plan said: "since this is a classification problem (with 10 classes), we
   use a sigmoid function and then get the highest probability as the
   classified class." This is the one real conceptual gap worth flagging:
   **sigmoid is for independent per-class probabilities (multi-label, or
   binary classification)** — for mutually-exclusive multi-class
   classification, the standard choice is softmax, and in practice you don't
   even need to compute it explicitly for *prediction* (`argmax` on raw
   logits gives the same class as `argmax` on softmax(logits), since softmax
   is monotonic) — you only need softmax/log-softmax for computing the
   *loss* (which is exactly what `F.cross_entropy` does internally).
   Sigmoid-then-argmax would still "work" in the sense of producing *a*
   answer, but is the wrong tool conceptually and wouldn't produce
   calibrated probabilities across the 10 classes (they wouldn't sum to 1).
3. **Convolution's parallelization mechanism.** The original plan's mental
   model (nested loop: locate window, matmul, stride forward; separately,
   another loop over filters) was conceptually accurate but the plan
   explicitly flagged uncertainty about whether the parallelization needs to
   be done manually or is handled by `einsum`/`torch`. Resolved: neither
   loop is ever written explicitly — a vectorized windowing op collapses the
   spatial loop into tensor axes, and a single `einsum` collapses the filter
   loop into a contraction axis.
4. **Padding, scoped too early.** The original plan raised the padding
   question during the *data-reading* step ("given the MNIST image
   dimension, educate me on how to identify if the logic needs padding or
   not"). Padding doesn't apply at that stage at all (MNIST images are
   already a fixed `28×28`) — it's a convolution-design question, not a
   data-loading one. Worth keeping those two concerns (data shape vs.
   conv-induced shape change) mentally separate from the start.
5. **Kernel size vs. filter count, conflated.** The plan's phrasing ("a
   function that takes a kernel size and filter size") treated these as one
   family of choice; they're independent hyperparameters (kernel = spatial
   window size, filter count = output channels), each with different
   guidance behind picking them.
6. **Intermediate dense-layer sizing, sidestepped rather than answered.**
   The plan asked how to size hidden layers in the "dense network" ("are
   they the same until the pre-output node..."). Once "3 layers total"
   was settled as 2 conv + exactly 1 linear layer, this question dissolved —
   there are no intermediate dense layers to size, since the single linear
   layer goes straight from flattened conv features to the 10 output
   classes. Worth knowing this is a designed-away question, not an
   unanswered one — if a future exercise *does* need multiple dense layers,
   the general answer is still open (commonly: shrink towards the output
   size, e.g. `784 → 128 → 10`, sized by experimentation).
