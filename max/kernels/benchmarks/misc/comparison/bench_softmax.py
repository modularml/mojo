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

# Softmax benchmark comparing MAX against PyTorch baseline.
# Run via kbench: kbench bench_softmax_comparison.yaml

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from typing import Any

import torch
from bencher_utils import Bench, ThroughputMeasure

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"),
)
from bench import bench_kineto
from max.driver import Accelerator
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType, ops


def _softmax_bytes(shape: list[int], dtype: torch.dtype) -> int:
    """Softmax reads input and writes output: 2 * N * element_size bytes."""
    n_elements = 1
    for d in shape:
        n_elements *= d
    bytes_per_element = torch.finfo(dtype).bits // 8
    return 2 * n_elements * bytes_per_element


def bench_torch_softmax(
    shape: list[int],
    axis: int,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int]:
    x = torch.randn(*shape, dtype=dtype, device="cuda")

    def run_kernel() -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=axis)

    # PyTorch uses softmax_warp_forward for small last-dim and
    # cunn_SoftMaxForward(Reg) for large last-dim.
    for kernel_name in ("softmax", "SoftMax"):
        try:
            time_s = bench_kineto(
                run_kernel,
                kernel_names=kernel_name,
                num_tests=num_iters,
                suppress_kineto_output=True,
                with_multiple_kernels=True,
            )
            assert isinstance(time_s, float)
            return time_s, _softmax_bytes(shape, dtype)
        except RuntimeError:
            continue
    raise RuntimeError(f"torch softmax kernel not found for shape={shape}")


def bench_max_softmax(
    shape: list[int],
    axis: int,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int]:
    max_dtype = torch_dtype_to_max(dtype)
    in_type = TensorType(max_dtype, shape=shape, device=DeviceRef.GPU())

    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "softmax_max",
        forward=partial(ops.softmax, axis=axis),
        input_types=[in_type],
    )
    model = session.load(graph)

    x = torch.randn(*shape, dtype=dtype, device="cuda")

    def run_kernel() -> Any:
        return model.execute(x.detach())

    # Short inner axes (<=32) use the warp kernel (softmax_warp_*);
    # longer rows use the block/online kernel (softmax_temperature_*).
    for kernel_name in ("softmax_temperature", "softmax_warp"):
        try:
            time_s = bench_kineto(
                run_kernel,
                kernel_names=kernel_name,
                num_tests=num_iters,
                suppress_kineto_output=True,
                with_multiple_kernels=True,
            )
            assert isinstance(time_s, float)
            return time_s, _softmax_bytes(shape, dtype)
        except RuntimeError:
            continue
    raise RuntimeError(f"MAX softmax kernel not found for shape={shape}")


def bench_softmax(
    shape: list[int],
    axis: int,
    dtype: torch.dtype,
    engine: str,
    num_iters: int,
) -> tuple[float, int] | None:
    print("=" * 80)
    print(
        f"Softmax Benchmark  shape={'x'.join(str(d) for d in shape)}"
        f"  axis={axis}  dtype={dtype}  engine={engine}"
    )
    print("=" * 80)

    try:
        if engine == "torch":
            return bench_torch_softmax(shape, axis, dtype, num_iters)
        elif engine == "modular_max":
            return bench_max_softmax(shape, axis, dtype, num_iters)
    except Exception as e:
        print(f"{engine} benchmark failed: {e}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Softmax benchmark: PyTorch vs MAX"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="32x128256",
        help="Input tensor shape as 'DxD...' (e.g. 32x128256, 4x32x512)",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        help="Reduction axis (default: -1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Input dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=["torch", "modular_max"],
        help="Backend to benchmark",
    )
    parser.add_argument(
        "--num_iters",
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output CSV path",
    )
    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    shape = [int(d) for d in args.shape.split("x")]
    dtype = dtype_map[args.dtype]

    result = bench_softmax(
        shape=shape,
        axis=args.axis,
        dtype=dtype,
        engine=args.engine,
        num_iters=args.num_iters,
    )

    if result and args.num_iters > 1:
        met_sec, total_bytes = result
        bw_measure = ThroughputMeasure(Bench.bytes, total_bytes)
        name = (
            "Softmax"
            f"/shape={args.shape}"
            f"/axis={args.axis}"
            f"/dtype={dtype}"
            f"/engine={args.engine}"
        )
        b = Bench(name, iters=1, met=met_sec, metric_list=[bw_measure])
        b.dump_report(output_path=args.output)
