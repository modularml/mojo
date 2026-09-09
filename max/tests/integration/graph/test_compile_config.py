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

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession, LogLevel
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def compile_config_ops_path() -> Path:
    return Path(os.environ["MODULAR_COMPILE_CONFIG_OPS_PATH"])


@pytest.mark.skip
def test_compile_config_split_k_reduction_scheme(
    session: InferenceSession, compile_config_ops_path: Path
) -> None:
    tensor_type = TensorType(
        dtype=DType.int32, shape=[1], device=DeviceRef.CPU()
    )
    with Graph(
        "graph", input_types=[], custom_extensions=[compile_config_ops_path]
    ) as graph:
        graph.output(
            ops.custom(
                "use_splitk_reduction_scheme",
                device=tensor_type.device,
                values=[],
                out_types=[tensor_type],
            )[0]
        )

    session.set_split_k_reduction_precision("ACCUM")
    model = session.load(graph)
    output = model.execute()[0]
    assert isinstance(output, Buffer)
    result = output.to_numpy()
    assert result == [1]

    session.set_split_k_reduction_precision("OUTPUT")
    model = session.load(graph)
    output = model.execute()[0]
    assert isinstance(output, Buffer)
    result = output.to_numpy()
    assert result == [2]


def test_compile_config_use_logger(
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
    session: InferenceSession,
    compile_config_ops_path: Path,
) -> None:
    tensor_type = TensorType(
        dtype=DType.int32, shape=[1], device=DeviceRef.CPU()
    )
    with Graph(
        "graph", input_types=[], custom_extensions=[compile_config_ops_path]
    ) as graph:
        graph.output(
            ops.custom("use_logger", DeviceRef.CPU(), [], [tensor_type])[0]
        )

    session.set_mojo_log_level(LogLevel.DEBUG)
    model = session.load(graph)
    output = model.execute()
    assert isinstance(output[0], Buffer)
    result = output[0].to_numpy()

    # On the Mojo side the logger level is set to DEBUG, so the result should be 10.
    assert result == [20]

    # On the Mojo side, the logger has printed "I'm a custom Mojo function!"
    # We need to capture the output and check that it matches.
    captured = capfd.readouterr()
    assert (
        "ERROR" in captured.out
        and "::: I'm a custom Mojo function!\n" in captured.out
    )


# I'm really not sure why this is. The kernel is in `max/tests/integration/graph/testdata/compile_config_ops/__init__.mojo`
@pytest.mark.skip(
    reason="TODO(GEX-2134): Could not find a mojo kernel registered for add_one_custom with function mogg.execute"
)
@pytest.mark.skipif(
    accelerator_api() != "cuda",
    reason="This test is checking if the PTX output is correct, it will be the "
    "same logic for HIP but we need to generalize the asserts.",
)
def test_compile_config_dump_asm(
    session: InferenceSession, compile_config_ops_path: Path
) -> None:
    rows = 5
    columns = 10
    dtype = DType.float32

    graph = Graph(
        "addition",
        forward=lambda x: (
            ops.custom(
                name="add_one_custom",
                device=DeviceRef.CPU(),
                values=[x],
                out_types=[
                    TensorType(
                        dtype=x.dtype,
                        shape=x.tensor.shape,
                        device=DeviceRef.CPU(),
                    )
                ],
            )[0].tensor
        ),
        input_types=[
            TensorType(dtype, shape=[rows, columns], device=DeviceRef.CPU()),
        ],
    )

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "kernel.ptx"
    session._dump_gpu_asm(output_path)

    model = session.load(graph, custom_extensions=compile_config_ops_path)

    x_values = np.random.uniform(size=(rows, columns)).astype(np.float32)

    x = Buffer.from_numpy(x_values).to(Accelerator())

    result = model.execute(x)[0]
    assert isinstance(result, Buffer)
    assert (result.to_numpy() == x_values + np.ones_like(x_values)).all()
    assert output_path.exists()
    assert "algorithm_functional" in output_path.read_text()

    temp_dir.cleanup()
