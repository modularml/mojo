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
"""Integration tests for ``StructuredOutputOverlapState``.

These tests exercise the async host-func plumbing the overlap
text-generation pipeline uses. They mirror the style of
``test_async_host_func_gpu.py`` (pytest fixtures, ``pytest.skip`` on
non-CUDA backends, ``accelerator.synchronize()`` between phases).

All tests require a CUDA accelerator -- :class:`CompletionFlag` and
``__unsafe_enqueue_async_py_host_func`` are CUDA-only.
"""

import threading
import time

import numpy as np
import pytest
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kernels import inplace_memcpy, wait_host_value_with_dep
from max.pipelines.lib.pipeline_variants.structured_output_overlap import (
    StructuredOutputOverlapState,
)


def _spin_until_signaled(state: StructuredOutputOverlapState) -> None:
    """Busy-wait until the bitmask flag is signalled (acquire-ordered load).

    The trampoline + worker pattern signals the flag from an AsyncRT
    worker thread once ``fn()`` returns. The kickoff host node lands
    on the device's default stream, so without other GPU work the
    driver has no reason to flush the stream and the host node may
    sit indefinitely. Force the stream to flush via ``synchronize()``
    before spinning so the trampoline fires; the worker then runs
    asynchronously and the spin sees the release-store.
    """
    state.device.default_queue.synchronize()
    deadline = time.monotonic() + 5.0
    while state.bitmask_flag.load() == 0:
        if time.monotonic() > deadline:
            raise TimeoutError("bitmask flag was not signalled within 5s")
        time.sleep(0.001)


# Small shape so the pinned allocation stays cheap across tests.
_MAX_BATCH = 2
_NUM_POSITIONS = 3
_VOCAB = 128


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator()


@pytest.fixture
def cpu() -> CPU:
    return CPU()


@pytest.fixture
def state(accelerator: Accelerator, cpu: CPU) -> StructuredOutputOverlapState:
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "StructuredOutputOverlapState requires a CUDA/HIP accelerator"
        )
    return StructuredOutputOverlapState(
        device=accelerator,
        cpu=cpu,
        max_batch_size=_MAX_BATCH,
        num_positions=_NUM_POSITIONS,
        vocab_size=_VOCAB,
    )


def test_async_enqueue_round_trip(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """Callback writes pinned memory; flag signals on completion.

    Verifies ``enqueue_async_callback`` runs the closure on an
    AsyncRT worker, observes the worker's pinned writes from the
    main thread once the flag is signalled.
    """
    # Pre-condition: flag is zero (just constructed).
    assert state.bitmask_flag.load() == 0

    pinned_np = state.pinned_bitmask.to_numpy()
    pinned_np[...] = False

    def cb() -> None:
        # Write a deterministic pattern from the worker thread.
        view = state.pinned_bitmask.to_numpy()
        view[...] = True

    state.enqueue_async_callback(cb)
    _spin_until_signaled(state)

    assert state.bitmask_flag.load() == 1
    assert bool(state.pinned_bitmask.to_numpy().all())


def test_callback_exception_does_not_deadlock(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """A raising callback still signals the flag via the trampoline."""

    def cb() -> None:
        raise RuntimeError("intentional test failure")

    state.enqueue_async_callback(cb)
    # Should not hang; the trampoline's except clause sets the flag.
    _spin_until_signaled(state)

    assert state.bitmask_flag.load() == 1


def test_prime_signals_flag_for_first_replay(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """``prime`` writes pinned memory and drops the flag to 1."""
    assert state.bitmask_flag.load() == 0

    src = np.ones(state.max_bitmask_shape, dtype=np.int32)
    state.prime(src)

    assert state.bitmask_flag.load() == 1
    assert bool(state.pinned_bitmask.to_numpy().all())


def test_reuse_across_iterations(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """N enqueue/wait cycles confirm the trampoline auto-resets the flag."""
    counter = [0]

    def cb() -> None:
        counter[0] += 1

    for _ in range(3):
        state.enqueue_async_callback(cb)
        _spin_until_signaled(state)
        assert state.bitmask_flag.load() == 1

    assert counter[0] == 3


def test_payload_shape_matches_wait_host_value(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """``wait_payload`` is CPU int64[2] = [flag._unsafe_ptr, 1].

    Matches the contract in
    ``max.nn.kernels.wait_host_value_with_dep``'s docstring.
    """
    payload = state.wait_payload
    assert payload.dtype == DType.int64
    assert tuple(payload.shape) == (2,)
    # Read via the pinned numpy view -- ``DevicePinnedBuffer`` is what
    # backs ``wait_payload`` (so that input bindings stay pinned -> CPU
    # graph input and avoid pageable HtoD copies).
    payload_np = payload.to_numpy()
    assert int(payload_np[0]) == state.bitmask_flag._unsafe_ptr
    assert int(payload_np[1]) == 1


def _build_overlap_smoke_graph(
    accelerator: Accelerator, shape: list[int]
) -> Graph:
    """Builds a minimal graph mirroring the Eagle K2.5 overlap structure:

      wait_host_value_with_dep(payload, scratch)
      inplace_memcpy(scratch, pinned_bitmask)

    The wait threads ``scratch`` through as a fake mutable operand so
    the graph compiler / cuGraph capture sees an explicit chain edge
    between the wait and the memcpy (both ops mutate ``scratch``).
    Without that edge the two ``inplace_custom`` ops are independent
    and may be reordered or parallelised.

    The pinned input is declared on the CPU per the engine's input
    binding rule ("Pinned tensors can only be used in place of CPU
    graph inputs"), even though the runtime ``DevicePinnedBuffer``'s
    ``.device`` is the accelerator.

    Both pinned and scratch use int32 because the bitmask is packed:
    one bit per vocab token stored in 32-bit words.
    """
    device_ref = DeviceRef.from_device(accelerator)
    with Graph(
        "overlap_h2d_smoke",
        input_types=[
            TensorType(DType.int32, shape, device=DeviceRef.CPU()),
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
            BufferType(DType.int32, shape, device=device_ref),
        ],
    ) as graph:
        pinned = graph.inputs[0].tensor
        payload = graph.inputs[1].buffer
        scratch = graph.inputs[2].buffer
        wait_host_value_with_dep(payload, scratch, device=device_ref)
        inplace_memcpy(scratch, pinned)
        graph.output()
    return graph


def test_in_graph_h2d_preserves_pinned_row_layout(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """End-to-end: pinned -> in-graph H2D -> device scratch preserves rows.

    This is the exact op composition the Eagle K2.5 overlap path uses
    in ``unified_eagle_model.py`` just before the acceptance sampler:

      wait_host_value(wait_payload, device=device_ref)
      inplace_memcpy(device_bitmask_scratch, pinned_bitmask)

    The test writes a distinct per-(row, position) pattern into the
    pinned bitmask, signals the completion flag, runs the graph, and
    asserts the device scratch matches the pinned source byte-for-byte.

    Catches:
      - DevicePinnedBuffer / graph-input device mismatch (would make
        inplace_memcpy lower as the wrong direction and produce
        uninitialized device contents).
      - Row reordering between pinned and device.
      - The wait_host_value not actually gating on the flag (would
        race the inplace_memcpy against the pinned write).
    """
    shape = list(state.max_bitmask_shape)
    graph = _build_overlap_smoke_graph(accelerator, shape)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    # Distinct per-(batch, position) pattern: row b position p is
    # all-True iff (b * num_positions + p) is even. Easy to inspect
    # by eye and easy to detect if rows are swapped or shifted.
    expected = np.zeros(state.max_bitmask_shape, dtype=np.int32)
    for b in range(state.max_batch_size):
        for p in range(state.num_positions):
            expected[b, p, :] = ((b * state.num_positions + p) % 2) == 0
    state.pinned_bitmask.to_numpy()[...] = expected

    # Pre-signal the flag so the in-graph wait_host_value passes
    # immediately (no host callback in this test).
    state.bitmask_flag.signal(1)

    # Zero the device scratch so a stale value can't masquerade as a
    # correct copy.
    state.device_bitmask_scratch.to_numpy()[...] = False

    model.execute(
        state.pinned_bitmask,
        state.wait_payload,
        state.device_bitmask_scratch,
    )
    accelerator.synchronize()

    actual = state.device_bitmask_scratch.to_numpy()
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            "Device scratch contents do not match pinned source. "
            "Possible causes: pinned-buffer/graph-input device mismatch, "
            "in-graph H2D row reordering, or wait_host_value not gating "
            "the inplace_memcpy against the pinned write."
        ),
    )


def test_callback_writes_propagate_to_device(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """End-to-end with the async callback in the loop.

    Enqueues a callback that writes a distinct per-row pattern into
    pinned, then runs the smoke graph -- the model stream's
    wait_host_value should gate the in-graph H2D on the callback's
    completion, and the device scratch should match what the callback
    wrote.
    """
    shape = list(state.max_bitmask_shape)
    graph = _build_overlap_smoke_graph(accelerator, shape)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    expected = np.zeros(state.max_bitmask_shape, dtype=np.int32)
    for b in range(state.max_batch_size):
        for p in range(state.num_positions):
            expected[b, p, :] = ((b + p) % 2) == 0

    def cb() -> None:
        state.pinned_bitmask.to_numpy()[...] = expected

    state.device_bitmask_scratch.to_numpy()[...] = False
    state.enqueue_async_callback(cb)

    model.execute(
        state.pinned_bitmask,
        state.wait_payload,
        state.device_bitmask_scratch,
    )
    accelerator.synchronize()

    actual = state.device_bitmask_scratch.to_numpy()
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            "Device scratch does not match the callback's pinned writes. "
            "Either wait_host_value_with_dep isn't gating on the flag, "
            "the host callback isn't reaching pinned, or the in-graph "
            "H2D is reading the wrong region."
        ),
    )


def test_stale_flag_no_race_with_default_queue_callback(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """Mirrors the production pattern: callback on the device default
    stream, captured graph with ``wait_host_value_with_dep`` +
    ``inplace_memcpy``. With a stale ``flag=1`` left from a prior iter
    and stale pinned data, dispatching the graph must not race the
    callback: stream ordering puts the trampoline's
    ``flag.reset()`` strictly before the wait poll, so the wait
    blocks until the worker writes fresh pinned + signals 1.

    Regression test for the cross-stream race that motivated moving
    the host func onto the device default stream (instead of a side
    stream). If a future refactor moves it back to a side stream
    without adding an explicit ordering edge to the captured-graph
    wait, this test should fail with stale data in scratch.
    """
    shape = list(state.max_bitmask_shape)
    graph = _build_overlap_smoke_graph(accelerator, shape)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    # Pre-condition: leftover flag=1 + stale pinned, as if from a
    # prior iteration's callback.
    stale = np.zeros(state.max_bitmask_shape, dtype=np.int32)
    state.pinned_bitmask.to_numpy()[...] = stale
    state.bitmask_flag.signal(1)
    state.device_bitmask_scratch.to_numpy()[...] = False
    accelerator.synchronize()

    # The "real" callback writes fresh data, deliberately slow so a
    # broken (non-gated) graph would visibly race past it.
    fresh = np.ones(state.max_bitmask_shape, dtype=np.int32)

    def cb() -> None:
        time.sleep(0.05)
        state.pinned_bitmask.to_numpy()[...] = fresh

    state.enqueue_async_callback(cb)

    model.execute(
        state.pinned_bitmask,
        state.wait_payload,
        state.device_bitmask_scratch,
    )
    accelerator.synchronize()

    actual = state.device_bitmask_scratch.to_numpy()
    np.testing.assert_array_equal(
        actual,
        fresh,
        err_msg=(
            "Device scratch contains stale data: the wait observed the "
            "leftover flag=1 instead of waiting for the trampoline's "
            "reset + worker re-signal. The host func is probably no "
            "longer on the model stream, or the wait + memcpy lost "
            "their shared mutable operand on ``device_bitmask_scratch``."
        ),
    )


def _build_overlap_smoke_graph_with_output(
    accelerator: Accelerator, shape: list[int]
) -> Graph:
    """Variant of :func:`_build_overlap_smoke_graph` whose graph emits a
    tensor output derived from ``device_bitmask_scratch``.

    ``model.execute(...)`` waits for outputs to be ready, so this
    forces the model stream to drain through
    ``wait_host_value_with_dep`` + ``inplace_memcpy`` + the
    ``buffer_load`` of scratch before returning. A no-output graph
    might let ``execute`` return as soon as work is queued, which would
    defeat the timing-based hoist detection in
    :func:`test_in_graph_h2d_is_gated_by_wait_host_value`.

    Both pinned and scratch use int32 because the bitmask is packed:
    one bit per vocab token stored in 32-bit words.
    """
    device_ref = DeviceRef.from_device(accelerator)
    with Graph(
        "overlap_h2d_smoke_with_output",
        input_types=[
            TensorType(DType.int32, shape, device=DeviceRef.CPU()),
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
            BufferType(DType.int32, shape, device=device_ref),
        ],
    ) as graph:
        pinned = graph.inputs[0].tensor
        payload = graph.inputs[1].buffer
        scratch = graph.inputs[2].buffer
        wait_host_value_with_dep(payload, scratch, device=device_ref)
        inplace_memcpy(scratch, pinned)
        graph.output(scratch[...])
    return graph


def test_overlap_smoke_graph_ir_dump(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """Dump the smoke graph IR so chain wiring between
    ``mo.wait_host_value_with_dep`` and ``mo.inplace_memcpy`` is
    auditable.

    The two ops share ``scratch`` as a mutable operand, so the
    captured cuGraph must serialise the memcpy after the wait. If the
    IR shows them as independent (no shared mutation), the compiler is
    free to reorder or parallelise them, and the in-graph H2D can run
    before the wait.

    The IR text goes to pytest's captured stdout; inspect with
    ``--capture=no`` or check the test log.
    """
    shape = list(state.max_bitmask_shape)
    graph = _build_overlap_smoke_graph_with_output(accelerator, shape)
    ir = repr(graph)
    print("===== overlap smoke graph IR =====")
    print(ir)
    print("===== end IR =====", flush=True)

    assert "wait_host_value_with_dep" in ir, (
        f"wait_host_value_with_dep op not found in IR:\n{ir}"
    )
    assert "mo.inplace_memcpy" in ir or "inplace_memcpy" in ir, (
        f"inplace_memcpy op not found in IR:\n{ir}"
    )


def test_in_graph_h2d_is_gated_by_wait_host_value(
    accelerator: Accelerator, state: StructuredOutputOverlapState
) -> None:
    """The captured in-graph H2D must not run before ``wait_host_value``
    passes.

    If the graph compiler hoists ``inplace_memcpy`` ahead of
    ``wait_host_value`` -- e.g. because the chain dependency between
    the two ``inplace_custom`` ops is lost during MOGG lowering or
    cuGraph capture, or because the captured graph executes
    parallel paths concurrently -- ``model.execute`` will return
    before the flag is signalled, because the H2D + scratch read
    can complete without ever needing the flag.

    The test:

      1. Constructs a graph where ``model.execute`` blocks on
         outputs derived from ``device_bitmask_scratch``.
      2. Leaves ``bitmask_flag`` at 0 (wait expects 1) so the wait
         genuinely blocks.
      3. Runs ``execute`` in a daemon thread; sleeps 200ms.
      4. Asserts the thread is still alive (wait gated the rest of
         the graph). If it returned, the H2D bypassed the wait.
      5. Signals the flag and joins the thread.

    Failure mode: ``thread_alive_before_signal == False`` ⇒ the
    in-graph H2D was reordered ahead of (or made parallel with) the
    wait. Fix is to add an explicit data dependency between the two
    ops (e.g. wait_host_value returning a token consumed by
    inplace_memcpy) or to teach the lowering / capture path to
    respect the device chain.
    """
    shape = list(state.max_bitmask_shape)
    graph = _build_overlap_smoke_graph_with_output(accelerator, shape)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    # Sentinel pattern. Pinned all-True is what the H2D would copy.
    state.pinned_bitmask.to_numpy()[...] = True
    state.device_bitmask_scratch.to_numpy()[...] = False
    accelerator.synchronize()

    # Flag is at 0 from construction; wait expects 1. Wait should block.
    assert state.bitmask_flag.load() == 0

    exc: list[BaseException] = []

    def run() -> None:
        try:
            outputs = model.execute(
                state.pinned_bitmask,
                state.wait_payload,
                state.device_bitmask_scratch,
            )
            # ``model.execute`` is async-dispatch: it returns once work
            # has been queued, not once the GPU has completed. To
            # observe whether the wait actually gated the rest of the
            # graph we need to force a host-visible sync. Materialize
            # the output to host via ``to_numpy()``; that issues a
            # blocking D2H on the model stream, so if the captured
            # graph is genuinely waiting on the flag, this call will
            # block until the flag is signalled.
            if outputs:
                _ = outputs[0].to_numpy()
            else:
                accelerator.synchronize()
        except BaseException as e:
            exc.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # Give the model stream a chance to enqueue the wait + downstream
    # ops. 200ms is long compared to a single replay's host time but
    # short enough to keep the test snappy.
    time.sleep(0.2)

    thread_alive_before_signal = t.is_alive()

    # Release the wait so the thread can drain and join cleanly.
    state.bitmask_flag.signal(1)

    t.join(timeout=5.0)
    assert not t.is_alive(), "model.execute() did not finish after signal"
    if exc:
        raise exc[0]

    assert thread_alive_before_signal, (
        "model.execute() returned before the flag was signalled. "
        "Either wait_host_value did not block, or the in-graph "
        "inplace_memcpy + scratch read ran ahead of the wait "
        "(compiler reorder, lost chain dependency, or parallel "
        "execution in the captured cuGraph)."
    )


# ------------------------- several verify widths -------------------------- #

_WIDTHS = (1, 2, 4)


@pytest.fixture
def multi_width_state(
    accelerator: Accelerator, cpu: CPU
) -> StructuredOutputOverlapState:
    """A state serving several verify widths, as a width schedule asks for."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "StructuredOutputOverlapState requires a CUDA/HIP accelerator"
        )
    return StructuredOutputOverlapState(
        device=accelerator,
        cpu=cpu,
        max_batch_size=_MAX_BATCH,
        num_positions=_WIDTHS,
        vocab_size=_VOCAB,
    )


def test_priming_one_width_does_not_disturb_another(
    multi_width_state: StructuredOutputOverlapState,
) -> None:
    """Widths are independent buffers, so one width cannot corrupt another.

    This is the property the single shared buffer could not provide, and the
    reason a single shared buffer could not serve several widths.
    """
    packed = multi_width_state.packed_vocab_size
    narrow, wide = _WIDTHS[0], _WIDTHS[-1]

    multi_width_state.prime(
        np.full((_MAX_BATCH, wide, packed), -1, dtype=np.int32)
    )
    multi_width_state.prime(
        np.zeros((_MAX_BATCH, narrow, packed), dtype=np.int32)
    )

    wide_rows = multi_width_state.pinned_for(wide).to_numpy()
    assert (wide_rows[:_MAX_BATCH, :wide, :] == -1).all(), (
        "priming the narrow width overwrote the wide width's rows"
    )
    narrow_rows = multi_width_state.pinned_for(narrow).to_numpy()
    assert (narrow_rows[:_MAX_BATCH, :narrow, :] == 0).all()
