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
from layout import Coord, TileTensor, row_major
from nn.gather_scatter import gather_elements

from std.utils.index import Index


def bench_gather(mut m: Bench, spec: GatherSpec) raises:
    @always_inline
    def bench_gather_wrapper(
        mut b: Bencher, concrete_spec: GatherSpec
    ) raises {}:
        bench_gather(b, concrete_spec)

    m.bench_with_input(
        bench_gather_wrapper, BenchId("gather", String(spec)), spec
    )


def bench_gather(mut bencher: Bencher, spec: GatherSpec) raises:
    var index_rand_min = 0
    var index_rand_max = spec.m1 - 1

    var input_shape = Index(spec.m1, spec.m2)
    var indices_shape = Index(spec.n1, spec.n2)

    var data_ptr = List(length=input_shape.flattened_length(), fill=Float32(0))
    rand(data_ptr)
    var data_tensor = TileTensor(data_ptr, row_major(Coord(input_shape)))

    var indices_ptr = List(
        length=indices_shape.flattened_length(), fill=Int32(0)
    )
    randint(
        indices_ptr,
        index_rand_min,
        index_rand_max,
    )
    var indices_tensor = TileTensor(
        indices_ptr, row_major(Coord(indices_shape))
    )

    var output_ptr = List(
        length=indices_shape.flattened_length(), fill=Float32(0)
    )
    var output_tensor = TileTensor(output_ptr, row_major(Coord(indices_shape)))

    @always_inline
    def bench_fn() raises {
        var data_tensor,
        var indices_tensor,
        var output_tensor,
        imm,
    }:
        gather_elements(
            data_tensor,
            indices_tensor,
            spec.axis,
            output_tensor,
            DeviceContext(api="cpu"),
        )

    bencher.iter(bench_fn)


@fieldwise_init
struct GatherSpec(ImplicitlyCopyable, Writable):
    var axis: Int
    var m1: Int
    var m2: Int
    var n1: Int
    var n2: Int

    # fmt: off
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the gather spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "axis=", self.axis,
            ";Dim=(", self.m1, ",", self.m2, ")",
            "(", self.n1, ",", self.n2, ")",
        )
    # fmt: on


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))
    bench_gather(m, GatherSpec(axis=1, m1=400, m2=400, n1=200, n2=200))
    bench_gather(m, GatherSpec(axis=1, m1=1000, m2=1000, n1=200, n2=200))
    m.dump_report()
