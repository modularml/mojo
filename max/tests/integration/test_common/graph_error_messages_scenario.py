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
"""Subprocess script: run a GPU OOB gather scenario and print JSON result.

Reads the scenario name from the _GRAPH_ERROR_SCENARIO environment variable.
Prints a single JSON line with the captured error (or lack thereof).

GPU OOB gather faults poison the CUDA context, so each scenario must run
in its own subprocess.
"""

from __future__ import annotations

import enum
import json
import os
import sys

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


class Scenario(enum.Enum):
    UNFUSED_OOB_GATHER = "unfused_oob_gather"
    FUSED_OOB_GATHER = "fused_oob_gather"
    FUSED_OOB_GATHER_DUPLICATE_SOURCES = "fused_oob_gather_duplicate_sources"
    FUSED_OOB_GATHER_UNROLLED_SOURCES = "fused_oob_gather_unrolled_sources"


def _run(scenario: Scenario) -> None:
    element_count = 4
    gpu = DeviceRef.GPU(0)

    data_type = TensorType(DType.float32, shape=[element_count], device=gpu)
    indices_type = TensorType(DType.int32, shape=[element_count], device=gpu)
    b_type = TensorType(DType.float32, shape=[element_count], device=gpu)

    if scenario is Scenario.UNFUSED_OOB_GATHER:
        with Graph(
            "oob_gather_unfused",
            input_types=(data_type, indices_type),
        ) as graph:
            data, indices = (inp.tensor for inp in graph.inputs)
            gathered = ops.gather(data, indices, axis=0)
            graph.output(gathered)

    elif scenario is Scenario.FUSED_OOB_GATHER:
        with Graph(
            "oob_gather_fused",
            input_types=(data_type, indices_type, b_type),
        ) as graph:
            data, indices, b = (inp.tensor for inp in graph.inputs)
            gathered = ops.gather(data, indices, axis=0)
            summed = ops.add(gathered, b)
            difference = ops.sub(gathered, b)
            product = ops.mul(summed, difference)
            graph.output(product)

    elif scenario is Scenario.FUSED_OOB_GATHER_DUPLICATE_SOURCES:
        with Graph(
            "oob_gather_fused_large",
            input_types=(data_type, indices_type, b_type),
        ) as graph:
            data, indices, b = (inp.tensor for inp in graph.inputs)
            out = ops.gather(data, indices, axis=0)
            for _ in range(11):
                summed = ops.add(out, b)
                difference = ops.sub(out, b)
                out = ops.mul(summed, difference)
            graph.output(out)

    elif scenario is Scenario.FUSED_OOB_GATHER_UNROLLED_SOURCES:
        with Graph(
            "oob_gather_fused_unrolled",
            input_types=(data_type, indices_type, b_type),
        ) as graph:
            data, indices, b = (inp.tensor for inp in graph.inputs)
            out = ops.gather(data, indices, axis=0)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            summed = ops.add(out, b)
            difference = ops.sub(out, b)
            out = ops.mul(summed, difference)
            graph.output(out)

    session = InferenceSession(devices=[Accelerator(0)])
    model = session.load(graph)

    data_buf = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(Accelerator(0))
    bad_indices_buf = Buffer.from_numpy(
        np.array([999999, 1000000, 1000001, 1000002], dtype=np.int32)
    ).to(Accelerator(0))
    b_buf = Buffer.from_numpy(
        np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    ).to(Accelerator(0))

    inputs = (
        (data_buf, bad_indices_buf)
        if scenario is Scenario.UNFUSED_OOB_GATHER
        else (data_buf, bad_indices_buf, b_buf)
    )

    try:
        outputs = model.execute(*inputs)
        outputs[0].to(CPU()).to_numpy()
        print(json.dumps({"no_error": True}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    name = os.environ.get("_GRAPH_ERROR_SCENARIO")
    if not name:
        print(
            json.dumps({"error": "No _GRAPH_ERROR_SCENARIO set"}),
            file=sys.stderr,
        )
        sys.exit(1)
    _run(Scenario(name))
