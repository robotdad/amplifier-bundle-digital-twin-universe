# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for _stop_mock_containers docker-absent guard.

Regression: on Incus-only hosts (no Docker binary) destroy() would raise
FileNotFoundError before reaching incus.stop_container / incus.delete_container,
leaving the Incus container running — a resource leak that required manual cleanup.

Root cause: _stop_mock_containers called subprocess.run(["docker", ...])
unconditionally.  The existing ``if result.returncode != 0: return`` guard
only handles "docker present but errored", not "docker binary missing".

Fix: guard with ``shutil.which("docker") is None`` at the top of the function.

No real Incus daemon, Docker daemon, or container runtime required.
All subprocess calls are mocked.

Run with: uv run pytest tests/unit/test_stop_mock_containers.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_result(stdout: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess with returncode=0."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = ""
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stop_mock_containers_no_docker_binary_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: when docker is absent, _stop_mock_containers must return
    immediately without raising and without calling subprocess.run.

    Before the fix, subprocess.run(["docker", "ps", ...]) would raise
    FileNotFoundError on Incus-only hosts.  That exception propagated out
    of destroy(), aborting before incus.stop_container / incus.delete_container
    ran, so the Incus container was never deleted.
    """
    import amplifier_bundle_digital_twin_universe.engine as engine_mod

    # Simulate Incus-only host: docker binary not on PATH
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _: None)

    # subprocess.run must NOT be called — if it is, the bug is still present
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "subprocess.run must NOT be called when docker binary is absent"
        )

    monkeypatch.setattr(engine_mod.subprocess, "run", _fail_if_called)

    # Must return None without raising FileNotFoundError or AssertionError
    result = engine_mod._stop_mock_containers("test-env-id")
    assert result is None


def test_stop_mock_containers_with_docker_calls_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive: when docker is present, subprocess.run IS invoked (existing
    behaviour preserved — the guard must not suppress the Docker cleanup path).
    """
    import amplifier_bundle_digital_twin_universe.engine as engine_mod

    # Simulate a host that has docker
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _: "/usr/bin/docker")

    subprocess_calls: list[list[str]] = []

    def _record(*args, **kwargs):
        subprocess_calls.append(list(args[0]))
        return _ok_result(stdout="")  # empty stdout → no containers to remove

    monkeypatch.setattr(engine_mod.subprocess, "run", _record)

    engine_mod._stop_mock_containers("test-env-id")

    assert len(subprocess_calls) >= 1, (
        "subprocess.run must be called when docker is present"
    )
    # The first call must be the docker ps inquiry
    assert subprocess_calls[0][0] == "docker"
    assert "ps" in subprocess_calls[0]
