"""Submit-and-poll layer for driving AiiDA workchains asynchronously.

Every `workflows/*.py` builder function returns a builder you can either run
blocking (`aiida.engine.run_get_node`, used by the CLI/examples) or submit to
the daemon (`aiida.engine.submit`) for a caller that cannot block for the
full job duration -- an MCP tool call, for instance, which has its own
timeout and needs to return control while a multi-minute `pw.x` run is still
going. This module is the generic submit/poll/result half of that, shared by
every workflow instead of re-implemented per MCP tool.

Requires the AiiDA daemon running (`verdi daemon start`) -- `submit()`
queues the process for the daemon to pick up; without a daemon worker it
just sits in the `created` state forever. `status()` surfaces that as a
`daemon_running` flag so callers can give a useful error instead of a
silent timeout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class JobStatus:
    pk: int
    label: str
    process_state: str  # created, waiting, running, finished, excepted, killed
    exit_status: int | None
    is_finished_ok: bool
    is_terminal: bool
    daemon_running: bool


TERMINAL_STATES = {"finished", "excepted", "killed"}


def submit_builder(builder, label: str | None = None) -> int:
    """Submit a builder to the daemon and return the new node's pk.
    Non-blocking: returns as soon as the process is queued, not when it
    finishes."""
    from aiida.engine import submit

    node = submit(builder)
    if label:
        node.label = label
    return node.pk


def is_daemon_running() -> bool:
    from aiida.engine.daemon.client import get_daemon_client

    try:
        return get_daemon_client().is_daemon_running
    except Exception:
        return False


def get_status(pk: int) -> JobStatus:
    from aiida import orm

    node = orm.load_node(pk)
    process_state = node.process_state.value if node.process_state else "unknown"
    return JobStatus(
        pk=pk,
        label=node.label or node.process_label,
        process_state=process_state,
        exit_status=node.exit_status,
        is_finished_ok=bool(node.is_finished_ok),
        is_terminal=process_state in TERMINAL_STATES,
        daemon_running=is_daemon_running(),
    )


def wait_for_job(pk: int, timeout_seconds: float = 60, poll_interval: float = 1.0) -> JobStatus:
    """Poll until the process reaches a terminal state or `timeout_seconds`
    elapses, whichever comes first. Safe to call repeatedly with a short
    timeout from an MCP tool that itself has a call-timeout budget."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    status = get_status(pk)
    while not status.is_terminal and time.monotonic() < deadline:
        time.sleep(poll_interval)
        status = get_status(pk)
    return status


def get_results(pk: int) -> dict:
    """Return a plain-dict view of a finished node's output namespace.
    Raises if the node hasn't reached a terminal state, or finished with a
    non-zero exit status (the caller should check `get_status` first)."""
    from aiida import orm
    from aiida.common import LinkType

    node = orm.load_node(pk)
    if node.process_state is None or node.process_state.value not in TERMINAL_STATES:
        raise RuntimeError(f"pk={pk} has not finished yet (process_state={node.process_state})")
    if not node.is_finished_ok:
        raise RuntimeError(f"pk={pk} finished with exit status {node.exit_status}: {node.exit_message}")

    # `node.outputs` is a NodeLinksManager (attribute-style access per link
    # label, e.g. `node.outputs.output_parameters`) -- it isn't a mapping and
    # has no `.items()`. The outgoing-links API is what actually enumerates
    # them; filtering to RETURN links excludes CALL_CALC/CALL_WORK links to
    # sub-processes (e.g. a restarted calculation's `iteration_01`), and
    # `.nested()` groups namespaced outputs (e.g. `base.pw.x`) into nested
    # dicts instead of flattening dotted labels.
    results = {}
    for key, value in node.base.links.get_outgoing(link_type=LinkType.RETURN).nested().items():
        if isinstance(value, dict):  # a nested output namespace, not a single node
            continue
        if isinstance(value, orm.Dict):
            results[key] = value.get_dict()
        elif isinstance(value, (orm.Float, orm.Int, orm.Str, orm.Bool)):
            results[key] = value.value
        else:
            # Structures, folders, arrays etc.: identify by pk/uuid rather than
            # trying to serialize potentially large binary/array content.
            results[key] = {"node_type": type(value).__name__, "pk": value.pk, "uuid": value.uuid}
    return results
