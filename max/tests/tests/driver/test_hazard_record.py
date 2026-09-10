# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Host<->device hazard tracking, on everything a host can reach without a GPU.

Tracking now covers staging memory only, and staging requires a non-host
device, so every buffer reachable here answers "untracked". That makes this
file about two things: the answers themselves, which are what keep the
narrowing honest, and the machinery that runs regardless of them -- the
completion object, the realization bridge, and the absence of a Python shadow
in front of any host access point.

The ordering guarantees live in ``test_hazard_gpu.py``, which has staging
memory to order.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from max._core.driver import HostHazardCompletion, HostHazardError
from max.driver import CPU, Accelerator, Buffer, Usage, accelerator_count
from max.driver import buffer as driver_buffer
from max.driver._hazard import render_producer_error, stamp_device_write
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.executor import UnsupportedGraphError
from max.experimental.tensor import Tensor, realization_context


def _cpu_queue():  # noqa: ANN202
    return CPU().default_queue


@pytest.fixture
def accelerator() -> Accelerator:
    if accelerator_count() == 0:
        pytest.skip("requires an accelerator")
    return Accelerator()


def test_a_recorded_completion_drains_its_event() -> None:
    HostHazardCompletion.record_on(_cpu_queue()).wait()


def test_completions_compare_by_identity() -> None:
    """Two tokens with equal fields are still distinct submissions."""
    queue = _cpu_queue()
    a = HostHazardCompletion.record_on(queue)
    b = HostHazardCompletion.record_on(queue)

    assert a is not b
    assert len({a, b}) == 2


def test_a_poisoned_completion_never_blocks_and_then_raises() -> None:
    comp = HostHazardCompletion.poisoned(_cpu_queue(), "producer blew up")

    assert comp.is_ready(), "a poisoned token must never block a waiter"
    with pytest.raises(HostHazardError, match="producer blew up"):
        comp.wait()


def test_poisoning_a_recorded_completion_takes_effect() -> None:
    """Poison outranks the event, so a failed producer cannot be waited on."""
    comp = HostHazardCompletion.record_on(_cpu_queue())
    comp.wait()  # fine before

    comp.poison("executor failed")
    with pytest.raises(HostHazardError, match="executor failed"):
        comp.wait()


def test_a_rendered_producer_error_keeps_the_traceback_as_text() -> None:
    """The stack still reaches the user, just not as live frames."""
    try:
        raise ValueError("producer blew up")
    except ValueError as producer_error:
        text = render_producer_error(producer_error)

    assert "Traceback (most recent call last)" in text
    assert "test_a_rendered_producer_error_keeps_the_traceback_as_text" in text
    assert "producer blew up" in text


def test_a_rendered_error_holds_no_reference_to_its_frames() -> None:
    """Text cannot reach a buffer, which is the point: `Buffer` has no
    ``tp_traverse``, so a cycle through a live traceback would leak it."""
    try:
        raise ValueError("boom")
    except ValueError as producer_error:
        text = render_producer_error(producer_error)

    assert isinstance(text, str)
    assert not hasattr(text, "__traceback__")


_CONSTRUCTION_ROUTES: list[tuple[str, Callable[[], Buffer]]] = [
    ("allocates", lambda: Buffer(DType.float32, (4,), CPU())),
    (
        "allocates staging on a host device",
        lambda: Buffer(DType.float32, (4,), CPU(), Usage.STAGING),
    ),
    (
        "allocates and fills",
        lambda: Buffer.zeros((4,), DType.float32, CPU()),
    ),
    (
        "wraps a numpy array",
        lambda: Buffer.from_numpy(np.zeros(4, dtype=np.float32)),
    ),
    ("slices", lambda: Buffer(DType.float32, (8,), CPU())[0:4]),
    (
        "reinterprets",
        lambda: Buffer(DType.float32, (4,), CPU()).view(DType.uint8),
    ),
    (
        "imports through dlpack",
        lambda: Buffer.from_dlpack(np.zeros(4, dtype=np.float32)),
    ),
    ("copies", lambda: Buffer(DType.float32, (4,), CPU()).copy()),
]


@pytest.mark.parametrize(
    ("build",),
    [(build,) for _, build in _CONSTRUCTION_ROUTES],
    ids=[name for name, _ in _CONSTRUCTION_ROUTES],
)
def test_nothing_reachable_from_a_host_device_is_tracked(
    build: Callable[[], Buffer],
) -> None:
    """Staging means pinned host memory allocated *against a device*, so a
    host device cannot produce it however the buffer is built."""
    assert not build()._host_hazard_tracked


def test_staging_on_a_host_device_is_not_pinned_either() -> None:
    """The request is honored only where a transfer can be asynchronous, and
    the predicate agrees with what the allocation actually did."""
    buf = Buffer(DType.float32, (4,), CPU(), Usage.STAGING)

    assert not buf.pinned
    assert not buf._host_hazard_tracked


def test_untracked_usage_is_only_meaningful_with_staging() -> None:
    plain = Buffer(DType.float32, (4,), CPU())
    opted_out = Buffer(
        DType.float32, (4,), CPU(), Usage.STAGING | Usage.UNTRACKED
    )

    assert not plain._host_hazard_tracked
    assert not opted_out._host_hazard_tracked


def test_a_view_reports_its_parents_usage() -> None:
    """The predicate reads storage usage, and every view path carries it, so
    slicing cannot silently flip a buffer's tracking."""
    parent = Buffer(DType.float32, (8,), CPU(), Usage.STAGING)

    assert parent[0:4].usage == parent.usage
    assert parent.view(DType.uint8).usage == parent.usage
    assert parent[0:4]._host_hazard_tracked == parent._host_hazard_tracked


def test_stamping_an_untracked_buffer_is_a_no_op() -> None:
    buf = Buffer(DType.float32, (4,), CPU())
    comp = HostHazardCompletion.record_on(_cpu_queue())

    buf._stamp_write(comp)
    buf._stamp_read(comp)

    # There is no record for the token to land in, so the absence of a raise
    # is the whole assertion.
    assert not buf._host_hazard_tracked


def test_stamp_device_write_records_nothing_when_all_are_untracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event no record can hold is pure overhead on the path least able to
    afford it, so the bridge skips the group entirely."""
    recorded = 0
    original = HostHazardCompletion.record_on

    def counting(queue):  # noqa: ANN001, ANN202
        nonlocal recorded
        recorded += 1
        return original(queue)

    monkeypatch.setattr(HostHazardCompletion, "record_on", counting)

    stamp_device_write(
        [Buffer(DType.float32, (4,), CPU())],
        [Buffer(DType.float32, (4,), CPU())],
    )

    assert recorded == 0


def test_stamp_device_write_accepts_a_poison_without_a_queue_to_record_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure path must never record an event: a poisoned token never
    consults one, and the submission it would stand for is the failed one."""
    monkeypatch.setattr(
        HostHazardCompletion,
        "record_on",
        lambda queue: pytest.fail("the failure path recorded an event"),
    )

    stamp_device_write([Buffer(DType.float32, (4,), CPU())], error="boom")


# A Python function on any of these would mean a wait moved back out of the
# binding, where every C++ caller bypasses it.
#
# `inplace_copy_from` and `view` are deliberately absent: their wrappers
# predate hazard tracking and still do argument work the binding does not,
# validation and shape deduction respectively.
_BINDING_OWNED = (
    "__dlpack__",
    "__getitem__",
    "__setitem__",
    "copy",
    "item",
    "to",
    "zeros",
)


@pytest.mark.parametrize("name", _BINDING_OWNED)
def test_the_host_access_points_are_the_raw_binding(name: str) -> None:
    attr = Buffer.__dict__.get(name)
    assert attr is not None, f"Buffer no longer publishes {name}"

    underlying = getattr(attr, "__func__", attr)
    assert getattr(underlying, "__module__", "") != driver_buffer.__name__, (
        f"Buffer.{name} is a Python shadow again. Waiting and stamping belong"
        " in the binding; a shadow here is bypassed by every C++ caller."
    )


def test_the_driver_module_publishes_no_python_hazard_state() -> None:
    """The record is owned by the binding and never handed to Python. A module
    attribute that looks like one means the split leaked back."""
    leaked = [
        name
        for name, value in vars(driver_buffer).items()
        if isinstance(value, type) and name.endswith("Record")
    ]
    assert not leaked, f"{leaked} should live in the binding, not in Python"


def test_the_buffer_exposes_no_record_slot() -> None:
    buf = Buffer(DType.float32, (4,), CPU())
    assert not hasattr(buf, "_host_hazard_record")
    assert not hasattr(buf, "_host_hazard_self_allocated")


def test_the_hazard_module_is_only_the_realization_bridge() -> None:
    """Everything else moved; what is left has one producer and one reason."""
    from max.driver import _hazard

    published = {
        name
        for name, value in vars(_hazard).items()
        if not name.startswith("__") and not isinstance(value, types.ModuleType)
    }
    assert published <= {
        "Buffer",
        "DeviceQueue",
        "HostHazardCompletion",
        "HostHazardError",
        "QueueKey",
        "Sequence",
        "annotations",
        "render_producer_error",
        "stamp_device_write",
        "traceback",
    }, f"unexpected names in the hazard bridge: {sorted(published)}"


# The eager interpreter declines some graphs and falls back to compiling. The
# stamping is the same code either way; these cases cover the compiled path.
_compiled_fallback = pytest.mark.filterwarnings(
    "ignore:The eager interpreter failed on this graph"
)


class _FailingExecutor:
    """Fails after execution would have started, so work may be in flight."""

    def execute(
        self, graph: object, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        raise RuntimeError("executor exploded")


class _RefusingExecutor:
    """Declines the graph, which its contract says happens before execution."""

    def execute(
        self, graph: object, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        raise UnsupportedGraphError("executor declined")


@pytest.fixture
def stamp_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int, str | None]]:
    """Records ``(written, read, error)`` shapes the bridge was handed.

    The buffers themselves are untracked on a host device, so what a CPU run
    can check is that the bridge is called at all, with the right sets and the
    right error, rather than what landed in a record.
    """
    seen: list[tuple[int, int, str | None]] = []
    original = rc.stamp_device_write

    def recording(written, read=(), error=None):  # noqa: ANN001, ANN202
        seen.append((len(written), len(read), error))
        return original(written, read, error)

    monkeypatch.setattr(rc, "stamp_device_write", recording)
    return seen


def _stage_a_pair() -> tuple[Tensor, Tensor]:
    """Stages the one graph shape every case below starts from.

    The compiled-model cache is keyed by the graph, so every distinct shape a
    test introduces costs seconds of real compilation on a contended worker.
    """
    return (
        Tensor.zeros([4], device=CPU()) + 1.0,
        Tensor.zeros([4], device=CPU()) + 2.0,
    )


def _realize_a_pair() -> tuple[Tensor, Tensor]:
    """Two realized CPU tensors, both written by one submission."""
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        left, right = _stage_a_pair()
    return left, right


@_compiled_fallback
def test_eager_realization_reaches_the_bridge(
    stamp_calls: list[tuple[int, int, str | None]],
) -> None:
    """`realize_all` is the one producer the binding cannot stamp for itself."""
    _realize_a_pair()

    assert stamp_calls, "realize_all must hand its outputs to the bridge"
    assert all(error is None for _, _, error in stamp_calls)
    written, read, _ = stamp_calls[-1]
    assert written > 0 and read > 0, (
        "outputs are written and inputs are read, in one call"
    )


@_compiled_fallback
def test_eager_realization_still_produces_the_right_values() -> None:
    """An order-sensitive op, so inputs fed in the wrong order show up here."""
    left, right = _realize_a_pair()

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        result = left - right

    np.testing.assert_array_equal(
        result.driver_tensor.to_numpy(), np.full(4, -1.0, dtype=np.float32)
    )


@_compiled_fallback
def test_a_failed_realization_hands_the_bridge_a_rendered_error(
    stamp_calls: list[tuple[int, int, str | None]],
) -> None:
    """Work already enqueued can land in a mutated source, so a later host
    read must fail rather than trust unfinished bytes."""
    destination, source = _realize_a_pair()
    stamp_calls.clear()

    failing = rc.EagerRealizationContext(executor=_FailingExecutor())
    with (
        pytest.raises(RuntimeError, match="executor exploded"),
        failing,
        realization_context(failing),
    ):
        F.buffer_store(destination, source)

    assert stamp_calls, "the failure path must still stamp"
    _, read, error = stamp_calls[-1]
    assert error is not None and "executor exploded" in error
    assert read == 0, "read-only inputs stay clean; nothing wrote them"


@_compiled_fallback
def test_a_declined_realization_stamps_nothing(
    stamp_calls: list[tuple[int, int, str | None]],
) -> None:
    """A refusal is not a failure: nothing ran, and poison is terminal."""
    destination, source = _realize_a_pair()
    stamp_calls.clear()

    refusing = rc.EagerRealizationContext(executor=_RefusingExecutor())
    with (
        pytest.raises(UnsupportedGraphError, match="executor declined"),
        refusing,
        realization_context(refusing),
    ):
        F.buffer_store(destination, source)

    assert not stamp_calls, (
        "poisoning a declined graph would be terminal for buffers -- including"
        " long-lived signal buffers -- that nothing ever wrote"
    )


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="the engine takes staging memory only for a CPU graph input, and "
    "`Tensor` derives its device from its storage, so staging cannot be an "
    "eager input; the bridge fires when a graph can produce staging output",
)
@pytest.mark.filterwarnings("ignore:The eager interpreter failed on this graph")
def test_eager_realization_stamps_a_staging_input(
    accelerator: Accelerator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staging graph input is the producer the binding cannot stamp itself."""
    staging = Buffer(
        dtype=DType.float32, shape=[4], device=accelerator, usage=Usage.STAGING
    )
    assert staging._host_hazard_tracked

    recorded = 0
    original = HostHazardCompletion.record_on

    def counting(queue):  # noqa: ANN001, ANN202
        nonlocal recorded
        recorded += 1
        return original(queue)

    monkeypatch.setattr(HostHazardCompletion, "record_on", counting)

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        out = Tensor(storage=staging) + 1.0

    assert out is not None
    assert recorded, "the bridge recorded nothing for a tracked input"
