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

"""Tests reusing precompiled MEFs across sessions, including the guard.

The point of matching artifacts by path rather than by compile key is that a
divergence is loud, so the mismatch cases matter more here than the happy path:
they are what stops a caller from quietly reverting to compiling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import support
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType


def _graph(name: str = "add_one", *, width: int = 4) -> Graph:
    dtype = TensorType(DType.float32, [width], device=DeviceRef.CPU())
    with Graph(name, input_types=[dtype]) as graph:
        graph.output(graph.inputs[0].tensor + 1.0)
    return graph


def _execute(model: Any, width: int = 4) -> np.ndarray:
    outputs = model.execute(np.zeros(width, dtype=np.float32))
    return outputs[0].to_numpy()


def test_exports_then_reuses_without_recompiling(tmp_path: Path) -> None:
    exported = InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph()
    )
    np.testing.assert_allclose(_execute(exported), np.ones(4))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [entry["name"] for entry in manifest["graphs"]] == ["add_one"]
    assert (tmp_path / manifest["graphs"][0]["key"]).is_file()

    reused = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path).load(
        _graph()
    )
    np.testing.assert_allclose(_execute(reused), np.ones(4))


def test_reusing_a_differently_shaped_graph_raises(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph(width=4)
    )

    # A shape divergence is what a pipeline sizing itself from device memory
    # produces, and it must not be papered over with a stale artifact.
    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    with pytest.raises(RuntimeError, match="no precompiled artifact"):
        session.load(_graph(width=8))


def test_reusing_a_renamed_graph_raises(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(
        _graph("add_one")
    )

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    with pytest.raises(RuntimeError, match="no precompiled artifact"):
        session.load(_graph("something_else"))


def test_compiling_a_graph_twice_reuses_one_artifact(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(_graph())
    assert len(list(tmp_path.glob("*.mef"))) == 1

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    for _ in range(2):
        np.testing.assert_allclose(_execute(session.load(_graph())), np.ones(4))


def test_a_graph_with_no_artifact_raises(tmp_path: Path) -> None:
    InferenceSession(devices=[CPU()], export_mefs=tmp_path).load(_graph())

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    session.load(_graph())
    with pytest.raises(RuntimeError, match="no precompiled artifact"):
        session.load(_graph("a_graph_that_was_never_exported"))


def test_artifacts_are_matched_regardless_of_order(tmp_path: Path) -> None:
    # The consuming run rarely compiles in the producing run's order -- it may
    # run a subset, or reach the same graphs by another path.
    exporting = InferenceSession(devices=[CPU()], export_mefs=tmp_path)
    exporting.load(_graph("first"))
    exporting.load(_graph("second", width=8))

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    np.testing.assert_allclose(
        _execute(session.load(_graph("second", width=8)), width=8),
        np.ones(8),
    )
    np.testing.assert_allclose(
        _execute(session.load(_graph("first"))), np.ones(4)
    )


def test_same_name_different_shapes_get_their_own_artifacts(
    tmp_path: Path,
) -> None:
    # What `Module.compile` produces: it names every graph after the module
    # class, so one model compiled at several shapes shares one name.
    exporting = InferenceSession(devices=[CPU()], export_mefs=tmp_path)
    exporting.load(_graph("shared", width=4))
    exporting.load(_graph("shared", width=8))
    assert len(list(tmp_path.glob("*.mef"))) == 2

    session = InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)
    np.testing.assert_allclose(
        _execute(session.load(_graph("shared", width=8)), width=8),
        np.ones(8),
    )


def test_reusing_a_directory_with_no_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no manifest\.json"):
        InferenceSession(devices=[CPU()], precompiled_mefs=tmp_path)


def test_exporting_and_reusing_at_once_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most one of"):
        InferenceSession(
            devices=[CPU()],
            precompiled_mefs=tmp_path,
            export_mefs=tmp_path,
        )


def test_is_inert_by_default() -> None:
    model = InferenceSession(devices=[CPU()]).load(_graph())
    np.testing.assert_allclose(_execute(model), np.ones(4))


def _linear() -> Module[[Tensor], Tensor]:
    """A one-op ModuleV3 model, pinned to CPU as the rest of this file is."""

    @module_dataclass
    class Linear(Module[[Tensor], Tensor]):
        weight: Tensor

        def forward(self, x: Tensor) -> Tensor:
            return x @ self.weight.T

    module = Linear(
        weight=Tensor.zeros([4, 4], dtype=DType.float32, device=CPU())
    )
    module.to(CPU())
    return module


def _module_input(rows: int = 4) -> Tensor:
    return Tensor.zeros([rows, 4], dtype=DType.float32, device=CPU())


def _module_input_type(rows: int = 4) -> TensorType:
    return TensorType(DType.float32, [rows, 4], device=DeviceRef.CPU())


def test_module_exports_then_reuses_without_recompiling(
    tmp_path: Path,
) -> None:
    # Module.compile keeps the artifact and the plumbing it derived rather than
    # handing the graph to `load`, so it needs its own coverage: without the
    # store wired into that path the export below writes nothing.
    with support.set_export_mefs(tmp_path):
        exported = _linear().compile(_module_input_type())
        expected = exported(_module_input()).to_numpy()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["graphs"]) == 1
    assert (tmp_path / manifest["graphs"][0]["key"]).is_file()

    with support.set_precompiled_mefs(tmp_path):
        reused = _linear().compile(_module_input_type())
        np.testing.assert_allclose(reused(_module_input()).to_numpy(), expected)


def test_module_reusing_a_differently_shaped_graph_raises(
    tmp_path: Path,
) -> None:
    with support.set_export_mefs(tmp_path):
        _linear().compile(_module_input_type())

    with support.set_precompiled_mefs(tmp_path):
        with pytest.raises(RuntimeError, match="no precompiled artifact"):
            _linear().compile(_module_input_type(rows=8))


def test_setting_mef_dirs_is_undone_on_scope_exit(tmp_path: Path) -> None:
    with support.set_export_mefs(tmp_path):
        _linear().compile(_module_input_type())
    recorded = json.loads((tmp_path / "manifest.json").read_text())["graphs"]

    # Back outside the scope the session records nothing, so the manifest is
    # unchanged by a further compile.
    _linear().compile(_module_input_type())
    after = json.loads((tmp_path / "manifest.json").read_text())["graphs"]
    assert after == recorded


def _graph_adding(name: str, addend: float, *, width: int = 4) -> Graph:
    """A graph whose name and signature say nothing about what it computes."""
    dtype = TensorType(DType.float32, [width], device=DeviceRef.CPU())
    with Graph(name, input_types=[dtype]) as graph:
        graph.output(graph.inputs[0].tensor + addend)
    return graph


def test_exporting_one_graph_twice_is_allowed(tmp_path: Path) -> None:
    # The other side of the collision guard: re-exporting a graph rewrites
    # bytes that describe the same computation, which is what a caller reaching
    # the same graph twice does and must keep being allowed to do.
    session = InferenceSession(devices=[CPU()], export_mefs=tmp_path)
    session.load(_graph_adding("repeated", 1.0))
    session.load(_graph_adding("repeated", 1.0))

    assert len(list(tmp_path.glob("*.mef"))) == 1


def test_exporting_two_graphs_under_one_name_raises(tmp_path: Path) -> None:
    # Nothing makes a graph's name unique, so a name and signature can describe
    # two different computations. Matching them by that pair would hand the
    # consumer whichever artifact was written last -- silently the wrong one, on
    # a path whose whole purpose is to avoid recompiling. Refuse instead, and
    # let the caller tell them apart by naming them differently.
    session = InferenceSession(devices=[CPU()], export_mefs=tmp_path)
    session.load(_graph_adding("collide", 1.0))

    with pytest.raises(RuntimeError, match="already exported"):
        session.load(_graph_adding("collide", 2.0))


def test_reuses_artifacts_split_across_directories(tmp_path: Path) -> None:
    # What a fragmented producer leaves behind: one directory per build action,
    # each with its own manifest. Nothing merges them.
    first, second = tmp_path / "a", tmp_path / "b"
    InferenceSession(devices=[CPU()], export_mefs=first).load(_graph("first"))
    InferenceSession(devices=[CPU()], export_mefs=second).load(
        _graph("second", width=8)
    )

    session = InferenceSession(
        devices=[CPU()], precompiled_mefs=[first, second]
    )
    np.testing.assert_allclose(
        _execute(session.load(_graph("second", width=8)), width=8), np.ones(8)
    )
    np.testing.assert_allclose(
        _execute(session.load(_graph("first"))), np.ones(4)
    )


def test_naming_no_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one directory"):
        InferenceSession(devices=[CPU()], precompiled_mefs=[])


def test_reusing_colliding_artifacts_from_two_directories_raises(
    tmp_path: Path,
) -> None:
    # The same-name collision the exporting session refuses, but split across
    # the per-action directories a fragmented producer leaves behind: neither
    # export sees the other, so the clash only becomes visible when the
    # manifests are unioned. Picking one silently would hand a graph the other
    # graph's code.
    first, second = tmp_path / "a", tmp_path / "b"
    InferenceSession(devices=[CPU()], export_mefs=first).load(
        _graph_adding("collide", 1.0)
    )
    InferenceSession(devices=[CPU()], export_mefs=second).load(
        _graph_adding("collide", 2.0)
    )

    with pytest.raises(RuntimeError, match="describe different graphs"):
        InferenceSession(devices=[CPU()], precompiled_mefs=[first, second])


def test_one_graph_exported_to_two_directories_is_allowed(
    tmp_path: Path,
) -> None:
    # Two build actions that both reach the same graph agree about it, so the
    # union has nothing to disambiguate and the duplicate is just a duplicate.
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        InferenceSession(devices=[CPU()], export_mefs=directory).load(
            _graph_adding("shared", 1.0)
        )

    session = InferenceSession(
        devices=[CPU()], precompiled_mefs=[first, second]
    )
    np.testing.assert_allclose(
        _execute(session.load(_graph_adding("shared", 1.0))), np.ones(4)
    )
