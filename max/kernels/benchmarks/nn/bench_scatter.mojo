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

from std.random import rand, randint

from std.benchmark import *
from max.gpu.host import DeviceContext
from std.memory import alloc, dealloc
from nn.gather_scatter import scatter_elements
from extensibility import DynamicTensor

from std.utils.index import Index


def bench_scatter(mut m: Bench, spec: ScatterSpec) raises:
    @always_inline
    def bench_scatter_wrapper(
        mut b: Bencher, concrete_spec: ScatterSpec
    ) raises {}:
        bench_scatter(b, concrete_spec)

    m.bench_with_input(
        bench_scatter_wrapper, BenchId("scatter", String(spec)), spec
    )


def bench_scatter(mut bencher: Bencher, spec: ScatterSpec) raises:
    var index_rand_min = 0
    var index_rand_max = spec.m1 - 1

    var input_shape = Index(spec.m1, spec.m2)
    var indices_shape = Index(spec.n1, spec.n2)

    var data_alloc = alloc[Float32](
        {count = input_shape.flattened_length()}
    ).into_managed()
    rand(data_alloc.unsafe_span())
    var data_tensor = DynamicTensor[.float32, 2](
        data_alloc.unsafe_ptr(), input_shape
    )

    var indices_alloc = alloc[Int32](
        {count = indices_shape.flattened_length()}
    ).into_managed()
    randint(
        indices_alloc.unsafe_span(),
        index_rand_min,
        index_rand_max,
    )
    var indices_tensor = DynamicTensor[.int32, 2](
        indices_alloc.unsafe_ptr(), indices_shape
    )

    var updates_alloc = alloc[Float32](
        {count = indices_shape.flattened_length()}
    ).into_managed()
    rand(updates_alloc.unsafe_span())
    var updates_tensor = DynamicTensor[.float32, 2](
        updates_alloc.unsafe_ptr(), indices_shape
    )

    var output_alloc = alloc[Float32](
        {count = input_shape.flattened_length()}
    ).into_managed()
    var output_tensor = DynamicTensor[.float32, 2](
        output_alloc.unsafe_ptr(), input_shape
    )

    @always_inline
    def bench_fn() raises {mut output_tensor, imm}:
        @always_inline
        def reduce_fn[
            _dtype: DType, width: SIMDLength
        ](
            input_val: SIMD[_dtype, width], update_val: SIMD[_dtype, width]
        ) -> SIMD[_dtype, width]:
            return input_val + update_val

        scatter_elements[reduce_fn=reduce_fn](
            data_tensor,
            indices_tensor,
            updates_tensor,
            spec.axis,
            output_tensor,
            DeviceContext(api="cpu"),
        )

    bencher.iter(bench_fn)

    _ = data_tensor
    _ = indices_tensor
    _ = updates_tensor
    _ = output_tensor

    dealloc(data_alloc^)
    dealloc(indices_alloc^)
    dealloc(updates_alloc^)
    dealloc(output_alloc^)


@fieldwise_init
struct ScatterSpec(ImplicitlyCopyable, Writable):
    var axis: Int
    var m1: Int
    var m2: Int
    var n1: Int
    var n2: Int

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the scatter spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "axis=",
            self.axis,
            ";Dim=(",
            self.m1,
            ",",
            self.m2,
            ")(",
            self.n1,
            ",",
            self.n2,
            ")",
        )


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))
    bench_scatter(m, ScatterSpec(axis=1, m1=400, m2=400, n1=200, n2=200))
    bench_scatter(m, ScatterSpec(axis=1, m1=1000, m2=1000, n1=200, n2=200))
    m.dump_report()
