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
"""Lifetime tests for the engine binding split.

Verifies that the ``nb::keep_alive`` annotations on the new
``CompiledModels`` / ``ModelMetadata`` types correctly retain parent objects
when handed out to Python.

``InferenceSession.compile_from_*`` returns an ``AsyncValue`` holding the
binding-level ``CompiledModels``. ``async_value.get()`` returns the typed
``CompiledModels`` wrapper backed by the AsyncRef storage; the
``type_hook<AsyncValue>`` specialization in MLRT's nanobind layer dispatches
to the right Python class via the held payload's ``TypeID``.
"""

from __future__ import annotations

import asyncio
import gc

import numpy as np
from max._core.engine import CompiledModels, ModelMetadata
from max._core.mlrt import AsyncValue
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import CompiledModel, InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue


def _make_graph() -> Graph:
    """A trivial single-input single-output identity-add graph."""

    def body(x: TensorValue) -> TensorValue:
        return x + 1.0

    return Graph(
        "identity_add",
        body,
        input_types=(TensorType(DType.float32, (4,), DeviceRef.CPU()),),
    )


def _handle(compiled: CompiledModel) -> AsyncValue[CompiledModels]:
    handle = compiled._compiled
    assert isinstance(handle, AsyncValue)
    return handle


def test_compile_returns_compiled_models() -> None:
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_make_graph())
    handle = _handle(compiled).result()
    assert isinstance(handle, CompiledModels)
    assert len(handle) >= 1


def test_compiled_models_len_and_indexing() -> None:
    session = InferenceSession(devices=[CPU()])
    handle = _handle(session.compile(_make_graph())).result()
    assert len(handle) == 1
    spec = handle[0]
    assert isinstance(spec, ModelMetadata)
    assert spec.name == "identity_add"


def test_compiled_models_iteration() -> None:
    session = InferenceSession(devices=[CPU()])
    handle = _handle(session.compile(_make_graph())).result()
    specs = list(handle)
    assert len(specs) == 1
    assert all(isinstance(s, ModelMetadata) for s in specs)


def test_compiled_models_names() -> None:
    session = InferenceSession(devices=[CPU()])
    handle = _handle(session.compile(_make_graph())).result()
    assert handle.names == ["identity_add"]


def test_model_metadata_input_output_metadata() -> None:
    session = InferenceSession(devices=[CPU()])
    handle = _handle(session.compile(_make_graph())).result()
    spec = handle[0]
    assert len(spec.input_metadata) == 1
    assert spec.input_metadata[0].dtype == DType.float32
    assert len(spec.output_metadata) == 1


def test_compiled_models_outlives_session() -> None:
    """``keep_alive<0, 1>`` on ``compile_from_*`` keeps the session alive
    via the returned ``AsyncValue``."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_make_graph())
    del session
    gc.collect()
    handle = _handle(compiled).result()
    assert len(handle) == 1
    assert handle.names == ["identity_add"]


def test_model_metadata_outlives_compiled_models() -> None:
    """``keep_alive<0, 1>`` on ``__getitem__`` keeps the parent CompiledModels alive."""
    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(_make_graph())
    spec = _handle(compiled).result()[0]
    del compiled
    del session
    gc.collect()
    assert spec.name == "identity_add"
    assert len(spec.input_metadata) == 1


def test_await_compiled_models() -> None:
    """``InferenceSession.compile`` returns an ``AsyncValue[CompiledModels]``
    — awaiting it gives the resolved ``CompiledModels`` handle."""

    async def go() -> object:
        session = InferenceSession(devices=[CPU()])
        return await _handle(session.compile(_make_graph()))

    handle = asyncio.run(go())
    assert isinstance(handle, CompiledModels)
    assert handle.names == ["identity_add"]


def test_loaded_model_outlives_session_and_compiled() -> None:
    """Each returned post-load Model holds an nb::ref<InferenceSession>
    internally, so it stays valid after dropping the session."""
    session = InferenceSession(devices=[CPU()])
    [model] = session.load_all(_make_graph()).values()
    assert isinstance(model, Model)
    del session
    gc.collect()
    input_buf = Buffer.from_numpy(np.zeros(4, dtype=np.float32))
    output = model.execute(input_buf)
    assert len(output) == 1
