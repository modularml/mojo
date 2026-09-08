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
"""Tests for the InferenceSession compile/init API split.

These tests exercise the public ``compile()`` / ``init()`` / ``init_all()``
methods that expose the two halves of ``load()`` as separate steps. They
also confirm ``load()`` / ``load_all()`` still behave the same way after
the internal refactor.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_count,
    get_virtual_cpu_target,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import CompiledModel, InferenceSession, Model
from max.engine._compilation_stats import collect_compilation_stats
from max.experimental.nn._compilation_timer import CompilationTimer
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn import Signals


@dataclass
class Unity:
    """Graph body that returns a scalar 1.0 (input is unused)."""

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.constant(1.0, dtype=DType.float32, device=DeviceRef.CPU())


def _unity_graph(name: str = "unity", module: Module | None = None) -> Graph:
    return Graph(
        name,
        forward=Unity(),
        input_types=[
            TensorType(DType.float32, ["batch", "dim"], device=DeviceRef.CPU())
        ],
        module=module,
    )


def test_compile_returns_compiled_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    assert isinstance(compiled, CompiledModel)
    # A CompiledModel is not executable: it has no `execute` attribute.
    assert not hasattr(compiled, "execute")


def test_compiled_model_repr() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    assert repr(compiled) == "CompiledModel()"


def test_init_produces_runnable_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())
    model = session.init(compiled)

    assert isinstance(model, Model)
    output = model.execute(np.zeros((1, 4), dtype=np.float32))
    assert len(output) == 1
    assert output[0].to_numpy().item() == pytest.approx(1.0)


def test_init_all_returns_single_model() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())
    models = session.init_all(compiled)

    assert len(models) == 1
    assert isinstance(next(iter(models.values())), Model)


def test_compile_init_matches_load() -> None:
    """compile()+init() should produce the same output as load()."""
    session = InferenceSession(devices=[CPU()])
    x = np.zeros((1, 4), dtype=np.float32)

    via_load = session.load(_unity_graph()).execute(x)
    compiled = session.compile(_unity_graph())
    via_compile = session.init(compiled).execute(x)

    assert via_load[0].to_numpy().item() == via_compile[0].to_numpy().item()


def test_compiled_model_can_be_initialized_twice() -> None:
    """A single compiled artifact can produce multiple independent Models."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    model_a = session.init(compiled)
    model_b = session.init(compiled)

    assert isinstance(model_a, Model)
    assert isinstance(model_b, Model)
    assert model_a is not model_b

    x = np.zeros((1, 4), dtype=np.float32)
    assert (
        model_a.execute(x)[0].to_numpy().item()
        == model_b.execute(x)[0].to_numpy().item()
    )


def test_init_rejects_non_contiguous_weights() -> None:
    """init() applies the same contiguity check that load() does today."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())

    weight_tensor = torch.randn(4, 4).t()
    assert not weight_tensor.is_contiguous()

    with pytest.raises(ValueError, match="non-contiguous tensors"):
        session.init(
            compiled, weights_registry={"extra": weight_tensor.numpy()}
        )


def test_load_after_refactor_still_works() -> None:
    """Regression: load() still returns a single executable Model."""
    session = InferenceSession(devices=[CPU()])
    model = session.load(_unity_graph())

    assert isinstance(model, Model)
    output = model.execute(np.zeros((1, 4), dtype=np.float32))
    assert output[0].to_numpy().item() == pytest.approx(1.0)


def test_load_all_after_refactor_still_works() -> None:
    """Regression: load_all() still returns a dict of executable models."""
    session = InferenceSession(devices=[CPU()])
    models = session.load_all(_unity_graph())

    assert len(models) == 1
    assert isinstance(next(iter(models.values())), Model)


def test_collector_records_compile_only_for_compile() -> None:
    """compile() updates compile_seconds; init_seconds stays at zero."""
    session = InferenceSession(devices=[CPU()])
    with collect_compilation_stats() as stats:
        session.compile(_unity_graph())

    assert stats.compile_seconds > 0.0
    assert stats.init_seconds == 0.0
    assert stats.build_seconds == 0.0
    assert stats.num_phases == 0


def test_collector_records_init_only_for_init_all() -> None:
    """init_all() updates init_seconds; compile_seconds stays at zero."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_unity_graph())
    with collect_compilation_stats() as stats:
        session.init_all(compiled)

    assert stats.init_seconds > 0.0
    assert stats.compile_seconds == 0.0


def test_collector_records_compile_and_init_for_load_all() -> None:
    """load_all() should populate both compile_seconds and init_seconds."""
    session = InferenceSession(devices=[CPU()])
    with collect_compilation_stats() as stats:
        session.load_all(_unity_graph())

    assert stats.compile_seconds > 0.0
    assert stats.init_seconds > 0.0


def test_compilation_timer_populates_outer_collector() -> None:
    """CompilationTimer should fill build/compile/init in the outer stats."""
    session = InferenceSession(devices=[CPU()])
    with collect_compilation_stats() as stats:
        with CompilationTimer("unity") as timer:
            graph = _unity_graph()
            timer.mark_build_complete()
            session.load_all(graph)

    assert stats.build_seconds >= 0.0
    assert stats.compile_seconds > 0.0
    assert stats.init_seconds > 0.0
    assert stats.num_phases == 1


@pytest.fixture
def virtual_device_mode() -> Iterator[None]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_80")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def test_accelerator_constructs_in_virtual_device_mode(
    virtual_device_mode: None,
) -> None:
    """Virtual-device mode must apply to device creation, not just counting.

    The virtual-device settings are process-wide state in the MLRT driver. If
    any image in the process (the ``_core`` extension, libmax, the Mojo
    bindings dylib) ends up with its own copy of that state, the
    ``set_virtual_device_*`` setters and ``accelerator_count()`` see one copy
    while ``Accelerator()`` construction sees another — so construction takes
    the real-hardware path even though the count reports virtual devices.
    That split broke CPU-only Linux runners ('No supported "gpu" device
    available') when the extension statically linked the driver
    implementation, and macOS for any device id beyond the physical GPU
    (DRIV-209, Mach-O two-level namespace binding the two halves of the API
    to different dylibs).

    Requesting id 1 with a virtual count of 2 is the discriminating check:
    no CI machine class is guaranteed two physical GPUs, so this only
    succeeds if creation consults the same virtual-device state the setters
    wrote.
    """
    set_virtual_device_count(2)
    assert accelerator_count() == 2
    # Must not raise, even on machines with zero or one physical GPU.
    Accelerator(0)
    Accelerator(1)


def test_init_all_in_virtual_device_mode_returns_dict(
    virtual_device_mode: None,
) -> None:
    """Cross-compilation: init_all() returns a subscriptable dict keyed by
    graph name even when virtual devices are in use, so multi-graph callers
    like the Kimi K2.5 vision+language pipeline can do
    ``models[vision_graph.name]``.
    """
    session = InferenceSession(devices=[CPU()])
    module = Module()
    encoder = _unity_graph(name="encoder", module=module)
    decoder = _unity_graph(name="decoder", module=module)
    compiled = session.compile(module)
    models = session.init_all(compiled)

    assert set(models.keys()) == {encoder.name, decoder.name}


def test_signal_buffers_skip_allocation_in_virtual_device_mode(
    virtual_device_mode: None,
) -> None:
    """Signal-buffer allocation degrades to nothing on a virtual device."""
    set_virtual_device_count(2)
    signals = Signals([DeviceRef.GPU(0), DeviceRef.GPU(1)])

    assert signals.buffers() == []


def test_weight_cast_keeps_dtype_and_payload_in_step_in_virtual_device_mode(
    virtual_device_mode: None,
) -> None:
    """A virtual-device cast converts no values, but the dtype it reports
    still describes its payload: loaders validate a weight against the dtype
    of its DLPack buffer, not against this metadata.
    """
    weight = WeightData.from_numpy(
        np.ones((3, 4), dtype=np.float32), "norm.weight"
    )

    converted = weight.astype(DType.bfloat16)

    assert converted.dtype == DType.bfloat16
    assert Buffer.from_dlpack(converted).dtype == DType.bfloat16


def test_nested_collectors_both_observe_phases() -> None:
    """Inner CompilationTimer's local stats and the outer collector both see
    compile and init events; nesting no longer shadows."""
    session = InferenceSession(devices=[CPU()])
    with collect_compilation_stats() as outer:
        with collect_compilation_stats() as inner:
            session.load_all(_unity_graph())

        assert inner.compile_seconds > 0.0
        assert inner.init_seconds > 0.0

    assert outer.compile_seconds >= inner.compile_seconds
    assert outer.init_seconds >= inner.init_seconds


def test_virtual_cpu_target_roundtrip() -> None:
    """The virtual CPU target setter/getter round-trips and defaults empty."""
    assert get_virtual_cpu_target() == ""
    try:
        set_virtual_cpu_target("x86-64-v3")
        assert get_virtual_cpu_target() == "x86-64-v3"
        set_virtual_cpu_target("generic")
        assert get_virtual_cpu_target() == "generic"
    finally:
        set_virtual_cpu_target("")
    assert get_virtual_cpu_target() == ""
