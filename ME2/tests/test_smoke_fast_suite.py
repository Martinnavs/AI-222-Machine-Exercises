"""Cross-cutting smoke test for whole-plan criterion 7: the fast suite (no
`slow` marker) must never require the vendor clone, model weights, or a GPU.

Ticket 03's developer verified this once manually by physically renaming
vendor/ away and re-running --help + the full suite. That's a point-in-time
check, not a regression guard - nothing stops a later change from
reintroducing a module-level `import cosyvoice` or an eager AutoModel/torch.cuda
call. This test asserts the same property in a way that runs on every fast
suite invocation: the real `cosyvoice` vendor package must never end up in
sys.modules as a side effect of importing/using the CLI and factory surface
that a fast, no-vendor, no-GPU environment is supposed to exercise.
"""

from __future__ import annotations

import sys

import pytest


def test_cosyvoice_vendor_package_never_imported_by_fast_surface() -> None:
    assert "cosyvoice" not in sys.modules, (
        "the real vendor `cosyvoice` package got imported somewhere in the fast "
        "suite - criterion 7 requires the fast suite to run with no vendor clone"
    )

    from me2_voicegen import generate_sample
    from me2_voicegen.synthesis import factory

    factory.list_backends()
    factory.get_backend_class("cosyvoice2")

    assert "cosyvoice" not in sys.modules


def test_generate_sample_help_exits_cleanly_without_vendor_or_gpu() -> None:
    from me2_voicegen import generate_sample

    with pytest.raises(SystemExit) as exc_info:
        generate_sample.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "cosyvoice" not in sys.modules


def test_cosyvoice2_backend_module_import_alone_does_not_touch_vendor() -> None:
    """Importing the backend module (as opposed to instantiating the class)
    must not eagerly import the vendor package - only __init__ may do that."""
    from me2_voicegen.synthesis import cosyvoice2_backend  # noqa: F401

    assert "cosyvoice" not in sys.modules
