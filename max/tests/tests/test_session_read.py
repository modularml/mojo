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
"""``max.engine.read`` loads exported ``.mef`` artifacts.

Round-trips ``CompiledModel.export_mef`` through ``read`` from both a path
and a binary file-like object, without re-invoking the graph compiler.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


@pytest.fixture(scope="module")
def mef_path(session: InferenceSession, tmp_path_factory) -> Path:  # noqa: ANN001
    with Graph(
        "addself",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as g:
        (x,) = g.inputs
        g.output(x.tensor + x.tensor)

    path = tmp_path_factory.mktemp("mef") / "addself.mef"
    session.compile(g).export_mef(str(path))
    return path


def _check_model(session: InferenceSession, compiled) -> None:  # noqa: ANN001
    model = session.init(compiled)
    (out,) = model(np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(out.to_numpy()), np.full(4, 2.0, dtype=np.float32)
    )


def test_read_from_path(session: InferenceSession, mef_path: Path) -> None:
    _check_model(session, engine.read(mef_path))


def test_read_from_str_path(session: InferenceSession, mef_path: Path) -> None:
    _check_model(session, engine.read(str(mef_path)))


def test_read_from_file_like(session: InferenceSession, mef_path: Path) -> None:
    # The buffer is consumed before read() returns; the model must stay
    # executable with no backing file reachable by path.
    _check_model(session, engine.read(io.BytesIO(mef_path.read_bytes())))


def test_read_missing_path(session: InferenceSession, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        engine.read(tmp_path / "nonexistent.mef")


def test_read_invalid_artifact(session: InferenceSession) -> None:
    with pytest.raises(Exception):
        engine.read(io.BytesIO(b"this is not a mef"))


def test_free_read_needs_no_session(
    session: InferenceSession, mef_path: Path
) -> None:
    _check_model(session, engine.read(mef_path))


def test_free_read_from_bytes(
    session: InferenceSession, mef_path: Path
) -> None:
    _check_model(session, engine.read(io.BytesIO(mef_path.read_bytes())))


def test_free_read_survives_source_deletion(
    session: InferenceSession, mef_path: Path, tmp_path: Path
) -> None:
    # The artifact is mmapped; unlinking the file must not invalidate it.
    copy = tmp_path / "unlinked.mef"
    copy.write_bytes(mef_path.read_bytes())
    compiled = engine.read(copy)
    copy.unlink()
    _check_model(session, compiled)


def test_free_read_repeat_init(
    session: InferenceSession, mef_path: Path
) -> None:
    compiled = engine.read(mef_path)
    _check_model(session, compiled)
    _check_model(session, compiled)


def test_free_read_invalid_artifact() -> None:
    with pytest.raises(RuntimeError):
        engine.read(io.BytesIO(b"this is not a mef"))
