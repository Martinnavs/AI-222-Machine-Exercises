# AI-222 Machine Exercises — instructions for Claude

This repo holds machine exercises (ME1, ME2, ...) for the AI-222 course. Each
exercise gets its own directory at the repo root (`ME1/`, `ME2/`, ...)
containing the final, graded deliverable — typically a Jupyter notebook plus
a committed `environment.yml`.

## `scratchpad/` is a persistent workpad — do not delete it

`scratchpad/` is gitignored and used as the iteration area for *every*
exercise (e.g. `scratchpad/ME1/`, `scratchpad/ME2/`, ...): prototype `.py`
modules, working notes, downloaded datasets, checkpoints, and the
`build_notebook.py`-style scripts that assemble each exercise's final
notebook. **Once an exercise is "done" and its notebook is committed, do not
delete or clean up its `scratchpad/<ME>/` contents.** Keep it around as
reusable scratch space and a record of how that exercise's code evolved —
future exercises may reuse patterns (e.g. `layers.py` helpers) from earlier
ones, and the student may want to revisit the iteration history.

`claude-independent-plan/` is the same story, one level up in purpose: each
exercise gets a `claude-independent-plan/<ME>-*.md` file — Claude's own
solo "if I built this alone" plan, written *before* the collaborative
session, kept as a private reference for steering that session. Also
gitignored, also not to be deleted.

## Workflow convention for a new machine exercise

This was established during ME1 and should carry forward:

1. Write an AI-only independent implementation plan first, as if working
   solo, in `claude-independent-plan/<ME>-*.md` (gitignored).
2. Run the actual build as a **Socratic, collaborative session** with the
   student — the student designs under guidance, Claude implements and
   teaches; don't just generate the whole thing unprompted. Iterate in
   `scratchpad/<ME>/` (gitignored) as plain `.py` modules first — easier to
   debug and test than notebook cells.
3. Quiz the student after each section is implemented, before moving on.
4. Once everything works, assemble the final notebook into `<ME>/` at the
   repo root (**not** gitignored — this is the actual submission), with the
   core/interesting logic (e.g. the einops/einsum layer implementations)
   inlined directly rather than imported from a scratchpad module the grader
   won't see. Execute the notebook end-to-end (e.g. via
   `jupyter nbconvert --to notebook --execute --inplace`) before committing,
   so outputs are real, not hand-typed.
5. Write a separate learnings/decisions/gaps recap doc into
   `scratchpad/<ME>/` (not the notebook) — main learnings, why each
   implementation decision was made, and a comparison against the student's
   own original plan.
6. Commit the exercise's `environment.yml` and final notebook to `master`.

## Conda environment conventions (shared HPC node)

- One conda env per exercise, created with an explicit `--prefix` under the
  user's home directory — **never** into the shared `/opt/miniconda3/envs`
  (not writable by students, and installing there would affect other HPC
  users). Pattern used for ME1:
  ```
  module load conda
  export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
  conda env create --prefix ~/conda-envs/<env-name> -f <ME>/environment.yml
  ```
- Each exercise commits its own `<ME>/environment.yml` (not gitignored) so
  the env is reproducible from the repo alone.
- First-time gotcha: `conda env create` can fail non-interactively with a
  `CondaToSNonInteractiveError` until the default channels' ToS are
  accepted:
  ```
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  ```
- `conda activate` fails non-interactively (`CondaError: Run 'conda init'
  before 'conda activate'`) unless `conda init` has been run against the
  shell — which we deliberately haven't done, to avoid mutating the
  student's shell rc files without asking. Instead, invoke the env's python
  directly: `~/conda-envs/<env-name>/bin/python script.py`. For Jupyter,
  register a kernel once per env:
  `~/conda-envs/<env-name>/bin/python -m ipykernel install --user --name <env-name>`,
  then execute notebooks with
  `--ExecutePreprocessor.kernel_name=<env-name>`.
- This node has multiple shared, otherwise-idle GPUs. Whether an exercise's
  env should be CPU-only or CUDA-enabled is the *student's* call to make
  explicitly (shared-resource etiquette on an HPC node), not an assumption —
  ask, don't default silently either way.

## Git conventions

- `gh` (already authenticated) is available for repo operations. If plain
  `git push`/`git pull` fails with credential errors (e.g. a stale VS Code
  git-credential-manager socket), run `gh auth setup-git` to point git's
  credential helper at `gh` itself for github.com/gist.github.com — this
  resolved a real failure during ME1.
- There is a `backup/scratchpad-and-plan` branch that force-tracks
  `scratchpad/` and `claude-independent-plan/` (via `git add -f`) as an
  off-master safety net, in case the working environment is lost mid-exercise.
  It is not meant to be merged into `master`.
  - **Gotcha, hit once already — avoid repeating it:** `git checkout` between
    a branch that tracks these gitignored paths and one that doesn't will
    *delete* them from the working tree (git syncs the working tree to the
    target branch's tracked files, regardless of `.gitignore`). If you need
    to update `backup/scratchpad-and-plan` while continuing work on
    `master` in the same working directory, use a **separate `git worktree`**
    for the backup branch instead of `git checkout`-ing back and forth in
    the main one. (ME1 recovered fine via `git show <branch>:<path> > path`
    since everything was already committed to the backup branch before the
    accidental checkout — but avoid needing that recovery step at all.)
