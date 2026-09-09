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

from std.math import iota
from std.random import rand
from std.sys import (
    argv,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.bit import log2_floor
from max.gpu.host import DeviceBuffer, DeviceContext
from kernels.matrix_multiplication import MatrixMultiplication
from kernels.tensor_core_mma import TensorCoreMMA
from kernels.top_k import TopK
from layout.int_tuple import product, to_index_list
from extensibility import (
    IOSpec,
    ManagedTensorSlice,
    StaticTensorSpec,
    get_row_major_tensor_spec_static,
)


# Wrap a ManagedTensorSlice and DeviceBuffer as an owning Tensor
@fieldwise_init
struct Tensor[
    dtype: DType,
    rank: Int,
    //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank, _],
](ImplicitlyCopyable):
    comptime size = product(Self.static_spec.shape_tuple)

    var slice: ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]
    var buffer: DeviceBuffer[Self.dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.size)

        self.slice = ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_spec
        ](
            self.buffer.unsafe_ptr(),
            to_index_list[Self.rank](Self.static_spec.shape_tuple),
            to_index_list[Self.rank](Self.static_spec.strides_tuple),
        )

    def rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.as_span())
            return self

    def iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.as_span())
            return self

    def fill(self, value: Scalar[Self.dtype]) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            for i in range(Self.size):
                host_buffer[i] = value
            return self

    def custom_fill_a(self, M: Int, K: Int) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            for i in range(M):
                for j in range(K):
                    host_buffer[i * K + j] = Scalar[Self.dtype](i)
            return self

    def custom_fill_b(self, K: Int, N: Int) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            for i in range(K):
                for j in range(N):
                    host_buffer[i * N + j] = Scalar[Self.dtype](j)
            return self


def top_k() raises:
    print("Running top-k benchmark...")
    comptime batch_size = 30_000
    comptime K = 32
    comptime els = batch_size * K
    comptime rank = 2
    comptime val_dtype = DType.float32
    comptime idx_dtype = DType.int32

    comptime val_spec = get_row_major_tensor_spec_static[
        val_dtype, rank, batch_size, K
    ]()
    comptime idx_spec = get_row_major_tensor_spec_static[
        idx_dtype, rank, batch_size, K
    ]()

    var cpu_ctx = DeviceContext(api="cpu")

    var in_vals = Tensor[IOSpec.Input, val_spec](cpu_ctx).rand()
    var out_vals = Tensor[IOSpec.Output, val_spec](cpu_ctx).rand()
    var out_idxs = Tensor[IOSpec.Output, idx_spec](cpu_ctx).rand()

    var b = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, els * log2_floor(K))
    var elements = ThroughputMeasure(BenchMetric.elements, els)
    var metrics: List = [flops, elements]

    def top_k_cpu() raises {imm}:
        TopK.execute[K=K, target="cpu"](
            out_vals.slice, out_idxs.slice, in_vals.slice, cpu_ctx
        )

    b.bench_function(top_k_cpu, BenchId("top_k_custom", "cpu"), metrics)

    comptime if has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext()

        var out_vals_dev = Tensor[IOSpec.Output, val_spec](gpu_ctx).rand()
        var out_idxs_dev = Tensor[IOSpec.Output, idx_spec](gpu_ctx).rand()
        var in_vals_dev = Tensor[IOSpec.Input, val_spec](gpu_ctx).rand()

        def top_k_gpu() raises {imm}:
            TopK.execute[K=K, target="gpu"](
                out_vals_dev.slice,
                out_idxs_dev.slice,
                in_vals_dev.slice,
                gpu_ctx,
            )

        b.bench_function(top_k_gpu, BenchId("top_k_custom", "gpu"), metrics)
    b.config.verbose_metric_names = False
    print(b)


def matmul() raises:
    print("Running matmul benchmark...")
    comptime M = 1028
    comptime K = 1028
    comptime N = 1028

    comptime rank = 2
    comptime dtype = DType.float32

    comptime FLOPS = M * N * (2 * K - 1)

    comptime a_spec = get_row_major_tensor_spec_static[dtype, rank, M, K]()
    comptime b_spec = get_row_major_tensor_spec_static[dtype, rank, K, N]()
    comptime c_spec = get_row_major_tensor_spec_static[dtype, rank, M, N]()

    var cpu_ctx = DeviceContext(api="cpu")

    var a = Tensor[IOSpec.Input, a_spec](cpu_ctx).rand()
    var b = Tensor[IOSpec.Input, b_spec](cpu_ctx).rand()
    var c = Tensor[IOSpec.Output, c_spec](cpu_ctx).rand()

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics: List = [flops, elements]

    def matmul_cpu() raises {imm}:
        MatrixMultiplication["naive"].execute[target="cpu"](
            c.slice, a.slice, b.slice, cpu_ctx
        )

    bench.bench_function(matmul_cpu, BenchId("cpu", "naive"), metrics)

    comptime if (
        has_amd_gpu_accelerator()
        or has_apple_gpu_accelerator()
        or has_nvidia_gpu_accelerator()
    ):
        var gpu_ctx = DeviceContext()
        var a_dev = Tensor[IOSpec.Input, a_spec](gpu_ctx).rand()
        var b_dev = Tensor[IOSpec.Input, b_spec](gpu_ctx).rand()
        var c_dev = Tensor[IOSpec.Output, c_spec](gpu_ctx).rand()

        def bench_matmul_kernel[impl: StaticString]() raises {mut bench, imm}:
            def bench_gpu() raises {imm}:
                MatrixMultiplication[impl].execute[target="gpu"](
                    c_dev.slice, a_dev.slice, b_dev.slice, gpu_ctx
                )

            bench.bench_function(
                bench_gpu, BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()

        comptime if not has_apple_gpu_accelerator():
            bench_matmul_kernel["tensor_core"]()

    bench.config.verbose_metric_names = False
    print(bench)


def tensor_core_mma() raises:
    print("Running tensor core mma benchmark...")
    comptime M = 4096
    comptime N = 4096
    comptime K = 4096

    comptime rank = 2
    comptime dtype = DType.float16

    comptime FLOPS = M * N * (2 * K - 1)

    comptime a_spec = get_row_major_tensor_spec_static[dtype, rank, M, K]()
    comptime b_spec = get_row_major_tensor_spec_static[dtype, rank, K, N]()
    comptime c_spec = get_row_major_tensor_spec_static[
        DType.float32, rank, M, N
    ]()

    var cpu_ctx = DeviceContext(api="cpu")

    var a = Tensor[IOSpec.Input, a_spec](cpu_ctx).rand()
    var b = Tensor[IOSpec.Input, b_spec](cpu_ctx).rand()
    var c = Tensor[IOSpec.Output, c_spec](cpu_ctx).rand()

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)
    var metrics: List = [flops, elements]

    comptime perform_validation = False

    comptime if perform_validation:
        bench.config.max_iters = 1
        bench.config.max_batch_size = 1
        bench.config.num_repetitions = 1

    # TODO: Add NVIDIA GPU support
    comptime if has_amd_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev = Tensor[IOSpec.Input, a_spec](gpu_ctx).rand()
        var b_dev = Tensor[IOSpec.Input, b_spec](gpu_ctx).rand()
        var c_dev = Tensor[IOSpec.Output, c_spec](gpu_ctx).rand()

        def bench_matmul_kernel[impl: StaticString]() raises {mut bench, imm}:
            def bench_gpu() raises {imm}:
                TensorCoreMMA[impl].execute[target="gpu", M=M, N=N, K=K](
                    c_dev.slice,
                    a_dev.slice,
                    b_dev.slice,
                    perform_validation,
                    gpu_ctx,
                )

            bench.bench_function(
                bench_gpu, BenchId("gpu", String(impl)), metrics
            )

        bench_matmul_kernel["naive_tensor"]()
        bench_matmul_kernel["basic_shared_mem"]()
        bench_matmul_kernel["multi_block_tiled"]()
        bench_matmul_kernel["scheduler_hints"]()
        bench_matmul_kernel["double_buffer"]()
        bench_matmul_kernel["mma_tile_buffers"]()

    bench.config.verbose_metric_names = False
    print(bench)


def main() raises:
    var args = argv()
    if len(args) == 1:
        top_k()
        matmul()
    else:
        for arg in argv():
            if arg == "--top-k":
                top_k()
            if arg == "--matmul":
                matmul()
            if arg == "--tensor-core-mma":
                tensor_core_mma()
