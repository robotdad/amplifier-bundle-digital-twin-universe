# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for REST-based create_container (nested Incus support).

Root cause context:
    ``incus launch`` CLI hangs indefinitely when invoked from inside an Incus
    container via a mounted Unix socket.  The CLI subscribes to
    ``/1.0/events`` via WebSocket for async operation completion; that
    WebSocket upgrade fails silently through a mounted socket.  The REST API
    plain-HTTP long-poll ``/1.0/operations/{id}/wait`` works correctly through
    the same socket.

    ``_should_use_rest_create()`` detects the nested context via two signals:
    (1) ``/dev/incus/sock`` (guest agent socket, present inside ALL Incus
    containers, never on the host) — the primary signal for our scenario where
    ``INCUS_SOCKET`` is not explicitly set in the worker; and (2) an explicit
    ``INCUS_SOCKET`` env-var override pointing to a valid socket.

Live verification (run 2026-06-02 against the host Incus daemon):
    - ``{"type":"image","fingerprint":"8e092cf5415c..."}`` → accepted ✓
    - ``{"type":"image","mode":"pull","server":"https://images.linuxcontainers.org",``
      ``"protocol":"simplestreams","alias":"ubuntu/24.04"}`` → accepted, reused
      cached image (fingerprint 8e092cf5415c5536) ✓
    - ``/dev/incus/sock`` in ``resolve-resovle-dev``: ``srw-rw-rw-`` → present ✓
    - ``INCUS_SOCKET`` env in worker container: NOT SET; socket accessible at
      default path ``/var/lib/incus/unix.socket`` via bind-mount ✓

No real Incus daemon required — all subprocess and socket calls are mocked.

Test groups:
    A — ``_incus_socket_path``, ``_IncusUnixHTTPConnection``, ``_incus_rest_request``
    B — ``_has_daemon_socket``, ``_has_guest_agent_socket``, ``_should_use_rest_create``
    C — ``_build_image_source``
    D — ``create_container`` integration (CLI host path + REST nested path)

Run with: uv run pytest tests/unit/test_incus_rest_create.py -v
"""

from __future__ import annotations

import json
import socket as _socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from amplifier_bundle_digital_twin_universe import incus
from amplifier_bundle_digital_twin_universe.incus import (
    IncusError,
    _IncusUnixHTTPConnection,
    _build_image_source,
    _has_daemon_socket,
    _has_guest_agent_socket,
    _incus_rest_request,
    _incus_socket_path,
    _should_use_rest_create,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ok() -> MagicMock:
    """Mock subprocess result with returncode=0."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = ""
    m.stderr = ""
    return m


def _fail(msg: str = "failed") -> MagicMock:
    """Mock subprocess result with returncode=1."""
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = msg
    return m


@pytest.fixture()
def real_socket(tmp_path: Path) -> str:
    """Create a real bound UNIX socket file; yield its path; clean up."""
    sock_path = str(tmp_path / "test.sock")
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.bind(sock_path)
        yield sock_path
    finally:
        s.close()


# ===========================================================================
# Group A: Connection helpers + _incus_rest_request
# ===========================================================================


def test_socket_path_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns /var/lib/incus/unix.socket when INCUS_SOCKET is not set."""
    monkeypatch.delenv("INCUS_SOCKET", raising=False)
    assert _incus_socket_path() == "/var/lib/incus/unix.socket"


def test_socket_path_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns the INCUS_SOCKET env var value when explicitly set."""
    monkeypatch.setenv("INCUS_SOCKET", "/run/incus-host.socket")
    assert _incus_socket_path() == "/run/incus-host.socket"


def test_unix_http_connection_stores_socket_path() -> None:
    """_IncusUnixHTTPConnection stores the socket path for later connect()."""
    conn = _IncusUnixHTTPConnection("/run/incus-test.sock")
    assert conn._socket_path == "/run/incus-test.sock"


def test_unix_http_connection_connects_via_af_unix(tmp_path: Path) -> None:
    """connect() opens an AF_UNIX connection to the stored path."""
    sock_path = str(tmp_path / "srv.sock")
    server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)
    try:
        conn = _IncusUnixHTTPConnection(sock_path)
        conn.connect()
        conn.close()
    finally:
        server.close()


def _make_mock_conn(read_data: bytes) -> MagicMock:
    """Build a minimal mock HTTPConnection whose getresponse().read() returns read_data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = read_data
    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    return mock_conn


def test_rest_request_sync_response_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """_incus_rest_request returns the parsed JSON dict for sync responses."""
    payload = {"type": "sync", "metadata": {"id": "abc"}, "status": "Success"}
    mock_conn = _make_mock_conn(json.dumps(payload).encode())

    monkeypatch.setattr(incus, "_IncusUnixHTTPConnection", lambda _path: mock_conn)

    result = _incus_rest_request(
        "GET", "/1.0/instances", None, "/var/lib/incus/unix.socket"
    )
    assert result["type"] == "sync"
    assert result["metadata"]["id"] == "abc"


def test_rest_request_async_polls_wait_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For async responses, _incus_rest_request polls /1.0/operations/{id}/wait."""
    async_payload = {
        "type": "async",
        "metadata": {"id": "op-123"},
    }
    wait_payload = {
        "type": "sync",
        "metadata": {"id": "op-123", "status": "Success", "err": ""},
    }

    conns = iter(
        [
            _make_mock_conn(json.dumps(async_payload).encode()),
            _make_mock_conn(json.dumps(wait_payload).encode()),
        ]
    )
    monkeypatch.setattr(incus, "_IncusUnixHTTPConnection", lambda _path: next(conns))

    result = _incus_rest_request(
        "POST", "/1.0/instances", {"name": "x"}, "/sock", op_wait_timeout=30
    )
    assert result["type"] == "sync"


def test_rest_request_op_error_raises_incus_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async operation returning err field raises IncusError."""
    async_payload = {"type": "async", "metadata": {"id": "op-err"}}
    wait_payload = {
        "type": "sync",
        "metadata": {"id": "op-err", "err": "no space left on device"},
    }

    conns = iter(
        [
            _make_mock_conn(json.dumps(async_payload).encode()),
            _make_mock_conn(json.dumps(wait_payload).encode()),
        ]
    )
    monkeypatch.setattr(incus, "_IncusUnixHTTPConnection", lambda _path: next(conns))

    with pytest.raises(IncusError, match="no space left on device"):
        _incus_rest_request("POST", "/1.0/instances", {}, "/sock")


def test_rest_request_error_type_raises_incus_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-level error response raises IncusError."""
    error_payload = {"type": "error", "error": "Not found", "error_code": 404}
    mock_conn = _make_mock_conn(json.dumps(error_payload).encode())
    monkeypatch.setattr(incus, "_IncusUnixHTTPConnection", lambda _path: mock_conn)

    with pytest.raises(IncusError, match="Not found"):
        _incus_rest_request("GET", "/1.0/instances/ghost", None, "/sock")


def test_rest_request_malformed_json_raises_incus_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed JSON response raises IncusError (not JSONDecodeError)."""
    mock_conn = _make_mock_conn(b"not-json{{{")
    monkeypatch.setattr(incus, "_IncusUnixHTTPConnection", lambda _path: mock_conn)

    with pytest.raises(IncusError, match="malformed JSON"):
        _incus_rest_request("GET", "/1.0/instances", None, "/sock")


# ===========================================================================
# Group B: Detection predicates
# ===========================================================================


def test_has_daemon_socket_with_real_socket(real_socket: str) -> None:
    """Returns True when the path is a real unix socket."""
    assert _has_daemon_socket(real_socket) is True


def test_has_daemon_socket_nonexistent_path() -> None:
    """Returns False when the path does not exist."""
    assert _has_daemon_socket("/nonexistent/path/incus.sock") is False


def test_has_daemon_socket_regular_file(tmp_path: Path) -> None:
    """Returns False when the path is a regular file, not a socket."""
    f = tmp_path / "not_a_socket.txt"
    f.write_text("hello")
    assert _has_daemon_socket(str(f)) is False


def test_has_guest_agent_socket_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns False when the guest agent socket path does not exist."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", "/nonexistent/dev/incus.sock")
    assert _has_guest_agent_socket() is False


def test_has_guest_agent_socket_present(
    real_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns True when _GUEST_AGENT_SOCKET points to a real unix socket."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", real_socket)
    assert _has_guest_agent_socket() is True


def test_has_guest_agent_socket_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when _GUEST_AGENT_SOCKET points to a regular file."""
    f = tmp_path / "not_a_sock.txt"
    f.write_text("data")
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", str(f))
    assert _has_guest_agent_socket() is False


def test_should_use_rest_create_guest_agent_present(
    real_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns True when the guest agent socket exists (nested Incus container)."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", real_socket)
    # daemon socket path arg doesn't matter — guest agent signal is sufficient
    assert _should_use_rest_create("/does/not/matter") is True


def test_should_use_rest_create_explicit_incus_socket(
    real_socket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns True when INCUS_SOCKET is explicitly set to a valid socket."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", "/nonexistent/path")
    monkeypatch.setenv("INCUS_SOCKET", real_socket)
    assert _should_use_rest_create(real_socket) is True


def test_should_use_rest_create_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns False when neither guest agent socket nor explicit INCUS_SOCKET exists."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", "/nonexistent/incus.sock")
    monkeypatch.delenv("INCUS_SOCKET", raising=False)
    assert _should_use_rest_create("/var/lib/incus/unix.socket") is False


def test_should_use_rest_create_default_socket_only_no_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns False when on the host: daemon socket at default path, no INCUS_SOCKET env,
    no guest agent socket.  This preserves the existing CLI path on the host."""
    monkeypatch.setattr(incus, "_GUEST_AGENT_SOCKET", "/nonexistent/incus.sock")
    monkeypatch.delenv("INCUS_SOCKET", raising=False)
    # The default daemon socket path (/var/lib/incus/unix.socket) exists on any
    # Incus host but MUST NOT trigger the REST path — that is host-level access.
    assert _should_use_rest_create("/var/lib/incus/unix.socket") is False


# ===========================================================================
# Group C: _build_image_source
# ===========================================================================


def test_build_image_source_images_prefix_remote() -> None:
    """images:ubuntu/24.04 → remote simplestreams pull from linuxcontainers.org."""
    src = _build_image_source("images:ubuntu/24.04")
    assert src == {
        "type": "image",
        "mode": "pull",
        "server": "https://images.linuxcontainers.org",
        "protocol": "simplestreams",
        "alias": "ubuntu/24.04",
    }


def test_build_image_source_images_prefix_strips_prefix() -> None:
    """The 'images:' prefix is stripped from the alias in the REST source dict."""
    src = _build_image_source("images:debian/12")
    assert src["alias"] == "debian/12"
    assert "mode" in src


def test_build_image_source_local_alias() -> None:
    """A plain local alias is passed through as-is."""
    src = _build_image_source("amplifier-cache-python")
    assert src == {"type": "image", "alias": "amplifier-cache-python"}


def test_build_image_source_local_alias_with_colon() -> None:
    """ubuntu:24.04 (not images: prefix) is treated as a local alias."""
    src = _build_image_source("ubuntu:24.04")
    assert src == {"type": "image", "alias": "ubuntu:24.04"}


def test_build_image_source_fingerprint_passthrough() -> None:
    """A plain fingerprint string is passed as a local alias (caller may use fingerprint key)."""
    fingerprint = "8e092cf5415c5536ba869021ba5641bf0bae9efecd28beb8e1484f496718ee25"
    src = _build_image_source(fingerprint)
    assert src == {"type": "image", "alias": fingerprint}


# ===========================================================================
# Group D: create_container integration
# ===========================================================================


def test_create_container_host_uses_cli_not_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the host (_should_use_rest_create=False), the incus launch CLI is used."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=False))

    with patch(
        "amplifier_bundle_digital_twin_universe.incus.subprocess.run",
        return_value=_ok(),
    ) as mock_run:
        incus.create_container("my-dtu", "images:ubuntu/24.04")

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["incus", "launch", "images:ubuntu/24.04"]
    assert cmd[3] == "my-dtu"


def test_create_container_host_cli_passes_config_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI path: --config k=v flags are added for each config entry."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=False))

    with patch(
        "amplifier_bundle_digital_twin_universe.incus.subprocess.run",
        return_value=_ok(),
    ) as mock_run:
        incus.create_container(
            "my-dtu",
            "images:ubuntu/24.04",
            {"security.nesting": "true"},
        )

    cmd = mock_run.call_args[0][0]
    assert "--config" in cmd
    assert "security.nesting=true" in cmd


def test_create_container_host_cli_error_raises_incus_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI path: non-zero returncode raises IncusError."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=False))

    with patch(
        "amplifier_bundle_digital_twin_universe.incus.subprocess.run",
        return_value=_fail("Instance already exists"),
    ):
        with pytest.raises(IncusError, match="Instance already exists"):
            incus.create_container("my-dtu", "images:ubuntu/24.04")


def test_create_container_nested_uses_rest_not_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested context (_should_use_rest_create=True): REST path taken, subprocess NOT called."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    rest_calls: list[dict] = []

    def _mock_rest(
        method: str, path: str, body: dict | None, socket_path: str, **kwargs
    ):
        rest_calls.append({"method": method, "path": path, "body": body})
        return {"type": "sync", "metadata": {}}

    monkeypatch.setattr(incus, "_incus_rest_request", _mock_rest)

    with patch(
        "amplifier_bundle_digital_twin_universe.incus.subprocess.run"
    ) as mock_run:
        incus.create_container("nested-dtu", "images:ubuntu/24.04")

    mock_run.assert_not_called()
    assert len(rest_calls) == 2  # POST create + PUT start


def test_create_container_nested_rest_correct_create_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST path: POST /1.0/instances body has correct name, source, and config."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    rest_calls: list[dict] = []

    def _mock_rest(
        method: str, path: str, body: dict | None, socket_path: str, **kwargs
    ):
        rest_calls.append({"method": method, "path": path, "body": body})
        return {"type": "sync", "metadata": {}}

    monkeypatch.setattr(incus, "_incus_rest_request", _mock_rest)

    incus.create_container(
        "nested-dtu",
        "images:ubuntu/24.04",
        {"security.nesting": "true"},
    )

    create_call = rest_calls[0]
    assert create_call["method"] == "POST"
    assert create_call["path"] == "/1.0/instances"

    body = create_call["body"]
    assert body["name"] == "nested-dtu"
    assert body["config"] == {"security.nesting": "true"}

    src = body["source"]
    assert src["type"] == "image"
    assert src["mode"] == "pull"
    assert src["server"] == "https://images.linuxcontainers.org"
    assert src["protocol"] == "simplestreams"
    assert src["alias"] == "ubuntu/24.04"


def test_create_container_nested_rest_correct_start_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST path: PUT /1.0/instances/{name}/state body has action=start."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    rest_calls: list[dict] = []

    def _mock_rest(
        method: str, path: str, body: dict | None, socket_path: str, **kwargs
    ):
        rest_calls.append({"method": method, "path": path, "body": body})
        return {"type": "sync", "metadata": {}}

    monkeypatch.setattr(incus, "_incus_rest_request", _mock_rest)

    incus.create_container("nested-dtu", "images:ubuntu/24.04")

    start_call = rest_calls[1]
    assert start_call["method"] == "PUT"
    assert start_call["path"] == "/1.0/instances/nested-dtu/state"
    assert start_call["body"]["action"] == "start"


def test_create_container_nested_rest_no_config_sends_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST path: config=None results in an empty config dict in the POST body."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    captured: list[dict] = []

    def _mock_rest(
        method: str, path: str, body: dict | None, socket_path: str, **kwargs
    ):
        captured.append({"method": method, "body": body})
        return {"type": "sync", "metadata": {}}

    monkeypatch.setattr(incus, "_incus_rest_request", _mock_rest)

    incus.create_container("dtu", "images:ubuntu/24.04")  # no config arg

    assert captured[0]["body"]["config"] == {}


def test_create_container_nested_rest_error_raises_incus_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST path: _incus_rest_request raising IncusError propagates as IncusError."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    def _failing_rest(method: str, path: str, body, socket_path: str, **kwargs):
        raise IncusError("Instance already exists")

    monkeypatch.setattr(incus, "_incus_rest_request", _failing_rest)

    with pytest.raises(IncusError, match="Instance already exists"):
        incus.create_container("dtu", "images:ubuntu/24.04")


def test_create_container_nested_local_image_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST path: local image alias (no images: prefix) uses alias source body."""
    monkeypatch.setattr(incus, "_should_use_rest_create", MagicMock(return_value=True))

    captured: list[dict] = []

    def _mock_rest(
        method: str, path: str, body: dict | None, socket_path: str, **kwargs
    ):
        captured.append({"method": method, "body": body})
        return {"type": "sync", "metadata": {}}

    monkeypatch.setattr(incus, "_incus_rest_request", _mock_rest)

    incus.create_container("dtu", "amplifier-cache-python")

    src = captured[0]["body"]["source"]
    assert src == {"type": "image", "alias": "amplifier-cache-python"}
    # No mode/server/protocol for local alias
    assert "mode" not in src
