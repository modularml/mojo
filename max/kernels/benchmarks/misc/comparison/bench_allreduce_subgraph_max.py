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

# AllReduce implementation in Modular MAX Graph with Subgraphs.
# This example demonstrates how to implement a multi-GPU allreduce operation
# using MAX Graph's subgraph.
# Run the benchmarks using kbench:
#    kbench bench_allreduce_subgraph_max.yaml


from __future__ import annotations

import argparse

import numpy as np
from bench import bench_kineto_with_cupti_warmup
from bencher_utils import Bench, ThroughputMeasure

# from benchmark import benchmark, main
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn import Allreduce, Signals


def check_available_devices() -> int:
    # Check for number of available GPUs, with maximum of 8
    available_gpus = accelerator_count()
    return min(available_gpus, 8)


def create_devices(num_gpus: int) -> tuple[list[DeviceRef], Signals]:
    # Create device references for each GPU
    devices = [DeviceRef.GPU(id=id) for id in range(num_gpus)]

    # Create signal for each GPU
    signals = Signals(devices=devices)

    return devices, signals


def build_and_compile_graph(
    device_refs: list[DeviceRef],
    signals: Signals,
    num_elements: int,
    dtype: DType = DType.float32,
) -> Model:
    # Get number of GPUs
    num_gpus = len(device_refs)

    # Create input types for each device
    input_types = [
        TensorType(dtype=dtype, shape=[num_elements], device=device_refs[i])
        for i in range(num_gpus)
    ]

    # Combine tensor types and buffer types
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "allreduce",
        input_types=all_input_types,
    ) as graph:
        # Get tensor inputs and apply scaling
        tensor_inputs = []
        for i in range(num_gpus):
            assert isinstance(graph.inputs[i], TensorValue)
            # Scale each input by (i + 1)
            scaled_input = graph.inputs[i].tensor * (i + 1)
            tensor_inputs.append(scaled_input)

        allreduce = Allreduce(num_accelerators=num_gpus)
        allreduce_outputs = allreduce(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_gpus:]],
        )

        graph.output(*allreduce_outputs)

    host = CPU()

    # Create device objects
    devices: list[Device]
    devices = [Accelerator(i) for i in range(num_gpus)]

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    return compiled


def synchronize_devices(devices: list[Accelerator]) -> None:
    for dev in devices:
        dev.synchronize()


def execute_graph(
    compiled: Model, input_tensors: list[Buffer], signal_buffers: list[Buffer]
) -> None:
    compiled.execute(*input_tensors, *signal_buffers)


def bench_allreduce_modular_max(
    num_gpus: int, num_bytes: int, dtype: DType, num_iters: int = 1
) -> tuple[float, float] | None:
    assert num_bytes % dtype.size_in_bytes == 0
    num_elements: int = num_bytes // dtype.size_in_bytes
    # Check for number of available GPUs, between 2 and 8
    num_gpus_available = check_available_devices()

    if num_gpus < 2:
        raise RuntimeError(
            f"Insufficient GPUs available for benchmark: found {num_gpus}, but "
            f"requires at least 2."
        )

    if num_gpus > num_gpus_available:
        raise RuntimeError(
            f"Number of requested GPUs for benchmark {num_gpus} are larger than available GPUs {num_gpus_available}."
        )

    # Create device refs and signals
    device_refs, signals = create_devices(num_gpus)

    # Create device objects
    devices = [Accelerator(i) for i in range(num_gpus)]

    # Build and compile the graph
    # with kepler.time("graph-compile"):
    compiled = build_and_compile_graph(device_refs, signals, num_elements)

    # Create input tensors
    a_np = np.ones(num_elements).astype(np.float32)

    # Create tensors & signal buffers on each device
    input_tensors = [Buffer.from_numpy(a_np).to(device) for device in devices]
    signal_buffers = signals.buffers()

    # Run warmup iteration
    synchronize_devices(devices)

    def run_kernel() -> None:
        execute_graph(compiled, input_tensors, signal_buffers)

    # run_kernel()
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="allreduce",
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )

    assert isinstance(time_s, float)
    bus_bw = 2 * num_bytes * (num_gpus - 1) / num_gpus
    return time_s, bus_bw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AllReduce benchmark: Modular MAX"
    )
    parser.add_argument(
        "--num-gpus",
        "--num_gpus",
        type=int,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--num-bytes",
        "--num_bytes",
        type=int,
        default=16 * 1024,
        help="Buffer size in bytes to test",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="modular_max",
        choices=["modular_max"],
        help="Select backend engine",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )

    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": DType.float16,
        "bfloat16": DType.bfloat16,
        "float32": DType.float32,
    }

    # All ranks must call run_benchmark for distributed benchmarks
    result = bench_allreduce_modular_max(
        num_gpus=args.num_gpus,
        num_bytes=args.num_bytes,
        dtype=dtype_map[args.dtype],
    )

    name = "bench_allreduce_subgraph_max"
    met_sec, bytes = result or [0, 0]
    bytes = args.num_bytes
    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    b = Bench(
        name,
        iters=1,
        met=met_sec,
        metric_list=[bytes_per_sec],
    )

    b.dump_report(output_path=args.output)


if __name__ == "__main__":
    main()
