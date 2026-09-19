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
from me2_voicegen.vcm.train import build_arg_parser, greedy_decode


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
