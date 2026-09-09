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
"""Kernel-level timing sanity for row_mean_of_squares vs ops.mean(x*x).

Times the device-side execution of a single-op graph with CUDA events. Not a
correctness test; correctness lives in test_row_mean_of_squares_gpu.py.
"""

from __future__ import annotations

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import row_mean_of_squares

_SHAPES = [(16, 1536), (16, 256)]
_ITERS = 200
_WARMUP = 50


def _build(use_custom: bool, rows: int, cols: int) -> Graph:
    device_ref = DeviceRef.GPU()
    with Graph(
        "bench",
        input_types=(
            TensorType(DType.bfloat16, [rows, cols], device=device_ref),
        ),
    ) as graph:
        (x,) = graph.inputs
        if use_custom:
            out = row_mean_of_squares(x.tensor)
        else:
            xf = ops.cast(x.tensor, DType.float32)
            out = ops.mean(xf * xf, axis=-1)
        graph.output(out)
    return graph


def _time(compiled: Model, x_buf: Buffer) -> float:
    for _ in range(_WARMUP):
        compiled.execute(x_buf)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        compiled.execute(x_buf)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / _ITERS * 1e3  # us per execute


def main() -> None:
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    for rows, cols in _SHAPES:
        x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cpu")
        x_buf = Buffer.from_dlpack(x).to(device)

        custom = session.load(_build(True, rows, cols))
        baseline = session.load(_build(False, rows, cols))

        t_custom = _time(custom, x_buf)
        t_base = _time(baseline, x_buf)
        print(
            f"shape=({rows},{cols})  row_mean_of_squares={t_custom:7.2f} us  "
            f"ops.mean(x*x)={t_base:7.2f} us  speedup={t_base / t_custom:5.2f}x"
        )


def test_bench_row_mean_of_squares() -> None:
    main()


if __name__ == "__main__":
    main()
