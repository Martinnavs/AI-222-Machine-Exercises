"""Fast CPU-only tests for vcm.train's core plumbing.

No GPU, no real data -- a tiny synthetic 2-example batch, run directly
through `MatchboxNetCTC` + `nn.CTCLoss` + AdamW, proving the
loss/length/padding wiring `vcm.train`'s real loop uses is correct in
isolation (ticket 04's fast-test acceptance criterion).
"""

import torch
from torch import nn

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC
from me2_voicegen.vcm.train import (
    build_arg_parser,
    greedy_decode,
    restore_batchnorm_stats,
    snapshot_batchnorm_stats,
)


def _tiny_batch():
    torch.manual_seed(0)
    n_mels = 40
    lengths = [30, 22]
    words = ["go", "ok"]

    features = [torch.randn(n_mels, length) for length in lengths]
    max_frames = max(lengths)
    padded = torch.zeros(2, n_mels, max_frames)
    for i, f in enumerate(features):
        padded[i, :, : f.shape[-1]] = f

    input_lengths = torch.tensor(lengths, dtype=torch.long)
    ids_per_example = [alphabet.encode(w) for w in words]
    target_ids = torch.cat([torch.tensor(ids, dtype=torch.long) for ids in ids_per_example])
    target_len = torch.tensor([len(ids) for ids in ids_per_example], dtype=torch.long)

    return padded, input_lengths, target_ids, target_len, words


def test_overfit_tiny_synthetic_batch_to_near_zero_ctc_loss():
    features, input_lengths, target_ids, target_len, words = _tiny_batch()

    torch.manual_seed(0)
    config = MatchboxNetConfig(
        n_mels=40, n_blocks=2, channels=32, kernel_sizes=[5, 5], prologue_channels=16, epilogue_channels=32
    )
    model = MatchboxNetCTC(config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    model.train()
    first_loss = None
    last_loss = None
    for _step in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, target_ids, input_lengths, target_len)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())

    assert last_loss < 0.05, f"expected near-zero final loss, got {last_loss}"
    assert last_loss < first_loss / 10

    model.eval()
    with torch.no_grad():
        logits = model(features)
    for i, word in enumerate(words):
        decoded = greedy_decode(logits[i, : input_lengths[i]])
        assert decoded == word, f"expected {word!r}, decoded {decoded!r}"


def test_zero_length_target_row_yields_finite_loss():
    """A silence/no-transcript row (target_len=0) must not blow up CTC loss
    -- Task 02's Established note that zero-length CTC targets are finite
    on this stack, exercised here through the same batch shape train.py
    builds."""
    torch.manual_seed(1)
    n_mels = 40
    features = torch.randn(2, n_mels, 20)
    input_lengths = torch.tensor([20, 20], dtype=torch.long)
    target_ids = torch.tensor(alphabet.encode("hi"), dtype=torch.long)
    target_len = torch.tensor([2, 0], dtype=torch.long)

    config = MatchboxNetConfig(
        n_mels=40, n_blocks=1, channels=16, kernel_sizes=[5], prologue_channels=16, epilogue_channels=16
    )
    model = MatchboxNetCTC(config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    logits = model(features)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss = criterion(log_probs, target_ids, input_lengths, target_len)
    assert torch.isfinite(loss)


def test_nonfinite_training_batch_does_not_poison_batchnorm_running_stats():
    """R1-01 regression: a single non-finite training batch used to leave
    every `nn.BatchNorm1d` layer's running stats permanently non-finite
    (a forward pass updates them as a side effect before the loss is even
    checked), so the model's `eval()`-mode output on later, clean input
    stayed non-finite forever even though `train()`-mode looked healthy.
    This exercises the same snapshot/restore `vcm.train`'s real loop now
    performs around a bad step, and asserts eval-mode output on clean
    input is finite afterward.
    """
    torch.manual_seed(2)
    n_mels = 40
    config = MatchboxNetConfig(
        n_mels=n_mels, n_blocks=2, channels=24, kernel_sizes=[5, 5], prologue_channels=16, epilogue_channels=24
    )
    model = MatchboxNetCTC(config)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    clean_features = torch.randn(2, n_mels, 20)
    input_lengths = torch.tensor([20, 20], dtype=torch.long)
    target_ids = torch.cat(
        [torch.tensor(alphabet.encode(w), dtype=torch.long) for w in ["go", "ok"]]
    )
    target_len = torch.tensor([len(alphabet.encode(w)) for w in ["go", "ok"]], dtype=torch.long)

    # Sanity: model is healthy in eval() mode on clean input before the bad step.
    model.eval()
    with torch.no_grad():
        pre_logits = model(clean_features)
    assert torch.isfinite(pre_logits).all()

    bad_features = clean_features.clone()
    bad_features[0, 0, 0] = float("nan")

    model.train()
    bn_snapshot = snapshot_batchnorm_stats(model)
    optimizer.zero_grad(set_to_none=True)
    logits = model(bad_features)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss = criterion(log_probs, target_ids, input_lengths, target_len)
    assert not torch.isfinite(loss)

    # Mirrors vcm.train's own bad-step handling: restore BN stats, skip
    # the backward/optimizer step, don't just `continue` past the symptom.
    restore_batchnorm_stats(model, bn_snapshot)

    model.eval()
    with torch.no_grad():
        post_logits = model(clean_features)
    assert torch.isfinite(post_logits).all(), (
        "eval()-mode output on clean input is non-finite after a rejected "
        "non-finite training batch -- BatchNorm running stats were poisoned"
    )


def test_batchnorm_snapshot_restore_reverts_all_batchnorm_layers():
    """Directly proves restore_batchnorm_stats() actually reverts every
    BatchNorm1d layer's running_mean/running_var/num_batches_tracked, not
    just some -- this is the mechanism the training loop relies on."""
    torch.manual_seed(3)
    config = MatchboxNetConfig(
        n_mels=40, n_blocks=3, channels=32, kernel_sizes=[5, 5, 5], prologue_channels=16, epilogue_channels=32
    )
    model = MatchboxNetCTC(config)
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    # prologue(1) + block1 main+residual(2, in/out channel mismatch) + block2(1) + block3(1) + epilogue(2)
    assert len(bn_layers) == 7

    model.train()
    snapshot = snapshot_batchnorm_stats(model)
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_layers]

    with torch.no_grad():
        model(torch.randn(2, 40, 20) * 5.0)

    after = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_layers]
    assert any(not torch.equal(b[0], a[0]) or not torch.equal(b[1], a[1]) for b, a in zip(before, after))

    restore_batchnorm_stats(model, snapshot)
    for m, (running_mean, running_var) in zip(bn_layers, before):
        assert torch.equal(m.running_mean, running_mean)
        assert torch.equal(m.running_var, running_var)


def test_onecycle_epochs_defaults_to_max_epochs_not_a_fixed_horizon():
    """Regression test for the LR-freeze bug found in this ticket's real
    training runs: `--onecycle-epochs` used to default to a fixed 20
    regardless of `--max-epochs`, so `OneCycleLR` fully annealed to
    ~0 LR by epoch 20 and then sat frozen there for the rest of a longer
    run (train.py's own loop stops calling `.step()` once `total_steps`
    is exhausted). It must instead default to whatever `--max-epochs`
    is, so the schedule's horizon always tracks the actual epoch budget.
    """
    args = build_arg_parser().parse_args(["--max-epochs", "150"])
    assert args.onecycle_epochs is None  # unresolved until main()'s defaulting step
    if args.onecycle_epochs is None:  # mirrors train.main()'s own defaulting logic
        args.onecycle_epochs = args.max_epochs
    assert args.onecycle_epochs == 150

    args_explicit = build_arg_parser().parse_args(["--max-epochs", "150", "--onecycle-epochs", "20"])
    assert args_explicit.onecycle_epochs == 20  # explicit override still respected


def test_onecycle_schedule_tracking_max_epochs_does_not_freeze_near_zero():
    """Directly demonstrates the bug class: with the OLD fixed 20-epoch
    OneCycleLR sizing, the LR at the same step count a longer run would
    reach fully anneals to ~0 (train.py's guard then just stops stepping
    it, i.e. freezes there). With sizing tracking the real `--max-epochs`
    budget, the LR at that same step count is still meaningfully above
    zero -- the schedule hasn't finished annealing early.
    """
    steps_per_epoch = 91  # matches this ticket's real train-split throughput
    old_fixed_onecycle_epochs = 20
    max_epochs = 150
    max_lr = 1e-4

    def lr_after_n_steps(total_steps: int, n_steps: int) -> float:
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.1
        )
        for _ in range(n_steps):
            optimizer.step()
            scheduler.step()
        return scheduler.get_last_lr()[0]

    n_steps = steps_per_epoch * old_fixed_onecycle_epochs

    old_sizing_lr = lr_after_n_steps(total_steps=n_steps, n_steps=n_steps)
    assert old_sizing_lr < max_lr * 0.01  # fully annealed -- this is the frozen value

    new_sizing_lr = lr_after_n_steps(total_steps=steps_per_epoch * max_epochs, n_steps=n_steps)
    assert new_sizing_lr > max_lr * 0.05  # nowhere near annealed yet at the same step count
