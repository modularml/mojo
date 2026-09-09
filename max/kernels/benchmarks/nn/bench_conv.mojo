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

from std.math import align_up, ceildiv
from std.memory import alloc, dealloc, Layout as AllocLayout
from std.random import rand
from std.sys import simd_width_of, size_of
from std.sys.defines import get_defined_int, get_defined_string

from std.benchmark import *
from std.benchmark import keep
from layout import Coord, Layout, LayoutTensor, RuntimeLayout
from nn.conv.conv import ConvDirectNHWC, ConvInfoStatic
from nn.conv.conv_utils import (
    ConvShape,
    extend_shape,
    get_direct_conv_micro_kernel_width,
)

from std.utils import IndexList
from std.utils.index import Index


def bench_conv(mut m: Bench, spec: ConvSpec) raises:
    comptime input_type = spec.static_info.input_type
    comptime filter_type = spec.static_info.filter_type
    comptime output_type = spec.static_info.output_type

    # Alignment in terms of number of elmements.
    comptime alignment = 64
    comptime input_align = alignment // size_of[input_type]()
    comptime filter_align = alignment // size_of[filter_type]()
    comptime output_align = alignment // size_of[output_type]()

    comptime simd_size = simd_width_of[filter_type]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    var f_per_group = spec.f // spec.num_groups

    var output_dims = IndexList[spec.static_info.rank](1)

    comptime for i in range(spec.static_info.rank):
        output_dims[i] = (
            spec.input_dims[i]
            + spec.pad[2 * i]
            + spec.pad[2 * i + 1]
            - spec.dilation[i] * (spec.filter_dims[i] - 1)
            - 1
        ) // spec.stride[i] + 1

    var packed_filter_shape = IndexList[spec.static_info.rank + 3](1)

    comptime for i in range(spec.static_info.rank):
        packed_filter_shape[i + 1] = output_dims[i]
    packed_filter_shape[0] = spec.num_groups * ceildiv(
        f_per_group, micro_kernel_f_size
    )
    packed_filter_shape[spec.static_info.rank + 1] = spec.c
    packed_filter_shape[spec.static_info.rank + 2] = micro_kernel_f_size

    # Input and output shape, sizes
    var input_shape = extend_shape(spec.input_dims, spec.n, spec.c)
    var input_alloc_size = align_up(input_shape.flattened_length(), input_align)
    var filter_alloc_size = align_up(
        packed_filter_shape.flattened_length(), filter_align
    )
    var output_shape = extend_shape(output_dims, spec.n, spec.f)
    var output_alloc_size = align_up(
        output_shape.flattened_length(), output_align
    )

    # Set the total buffer allocation to be 4x L3 cache.
    comptime MB = 1024 * 1024
    comptime L3_cache = get_defined_int["L3SIZE", 24]() * MB
    var size_per_copy = (
        input_alloc_size * size_of[input_type]()
        + filter_alloc_size * size_of[filter_type]()
    )
    var num_copies = ceildiv(4 * L3_cache, size_per_copy)

    # Allocate input and output buffers.
    var input_alloc = alloc(
        AllocLayout[Scalar[input_type]].aligned[alignment](
            count=num_copies * input_alloc_size
        )
    ).into_managed()
    var input_ptr = input_alloc.unsafe_ptr()
    var filter_alloc = alloc(
        AllocLayout[Scalar[filter_type]].aligned[alignment](
            count=num_copies * filter_alloc_size
        )
    ).into_managed()
    var filter_ptr = filter_alloc.unsafe_ptr()
    var output_alloc = alloc(
        AllocLayout[Scalar[output_type]].aligned[alignment](
            count=num_copies * output_alloc_size
        )
    ).into_managed()
    var output_ptr = output_alloc.unsafe_ptr()

    rand(input_alloc.unsafe_span())
    rand(filter_alloc.unsafe_span())

    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

    comptime if spec.static_info.rank == 1:
        pad_w = Index(spec.pad[0], spec.pad[1])
    elif spec.static_info.rank == 2:
        pad_h = Index(spec.pad[0], spec.pad[1])
        pad_w = Index(spec.pad[2], spec.pad[3])
    elif spec.static_info.rank == 3:
        pad_d = Index(spec.pad[0], spec.pad[1])
        pad_h = Index(spec.pad[2], spec.pad[3])
        pad_w = Index(spec.pad[4], spec.pad[5])

    var conv_shape = ConvShape[spec.static_info.rank](
        n=spec.n,
        input_dims=Coord(spec.input_dims),
        output_dims=Coord(output_dims),
        filter_dims=Coord(spec.filter_dims),
        c=spec.c,
        f=spec.f,
        stride=Coord(spec.stride),
        dilation=Coord(spec.dilation),
        pad_d=Coord(pad_d),
        pad_h=Coord(pad_h),
        pad_w=Coord(pad_w),
        num_groups=spec.num_groups,
    )

    @always_inline
    def bench_conv_wrapper(
        mut b: Bencher, concrete_spec: ConvSpec[spec.static_info]
    ) raises {imm}:
        # Count the iteration to decide which input copy to use.
        var counter = 0

        @always_inline
        def bench_fn() {mut counter, imm}:
            comptime layout_2 = Layout.row_major[spec.static_info.rank + 2]()
            comptime layout_3 = Layout.row_major[spec.static_info.rank + 3]()
            var input = LayoutTensor[input_type, layout_2](
                input_ptr.unsafe_offset(
                    (counter % num_copies) * input_alloc_size
                ),
                RuntimeLayout[layout_2].row_major(input_shape),
            )
            var filter = LayoutTensor[filter_type, layout_3](
                filter_ptr.unsafe_offset(
                    (counter % num_copies) * filter_alloc_size
                ),
                RuntimeLayout[layout_3].row_major(packed_filter_shape),
            )
            var output = LayoutTensor[output_type, layout_2](
                output_ptr.unsafe_offset(
                    (counter % num_copies) * output_alloc_size
                ),
                RuntimeLayout[layout_2].row_major(output_shape),
            )

            try:
                ConvDirectNHWC[
                    layout_2,
                    layout_3,
                    layout_2,
                    input_type,
                    filter_type,
                    output_type,
                    True,
                    ConvInfoStatic[spec.static_info.rank](),
                ].run(
                    output,
                    input,
                    filter,
                    conv_shape,
                )

                counter += 1

            except e:
                print(e)

            keep(output.ptr)

        b.iter(bench_fn)

    m.bench_with_input(
        bench_conv_wrapper,
        BenchId("Conv", String(spec)),
        spec,
        # TODO: Pick relevant benchmetric.
        [ThroughputMeasure(BenchMetric.elements, spec.flops())],
    )

    dealloc(input_alloc^)
    dealloc(filter_alloc^)
    dealloc(output_alloc^)


@fieldwise_init
struct ConvSpecStatic(ImplicitlyCopyable):
    # Conv rank, 1d, 2d, or 3d. The input rank is rank + 2.
    var rank: Int
    var input_type: DType
    var filter_type: DType
    var output_type: DType


@fieldwise_init
struct ConvSpec[static_info: ConvSpecStatic](ImplicitlyCopyable, Writable):
    var n: Int
    var input_dims: IndexList[Self.static_info.rank]
    var c: Int
    var filter_dims: IndexList[Self.static_info.rank]
    var f: Int
    var stride: IndexList[Self.static_info.rank]
    var dilation: IndexList[Self.static_info.rank]
    var pad: IndexList[2 * Self.static_info.rank]
    var num_groups: Int

    # fmt: off
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "n=", self.n,
            ";input=", self.input_dims,
            ";c=", self.c,
            ";f=", self.f,
            ";filter=", self.filter_dims,
            ";stride=", self.stride,
            ";padding=", self.pad,
        )
    # fmt: on

    def flops(self) -> Int:
        var output_dims = IndexList[Self.static_info.rank](1)

        comptime for i in range(Self.static_info.rank):
            output_dims[i] = (
                self.input_dims[i]
                + self.pad[2 * i]
                + self.pad[2 * i + 1]
                - self.dilation[i] * (self.filter_dims[i] - 1)
                - 1
            ) // self.stride[i] + 1

        return (
            2
            * self.n
            * output_dims.flattened_length()
            * self.filter_dims.flattened_length()
            * self.c
            * self.f
        )


def main() raises:
    var m = Bench(BenchConfig())

    comptime fp32_1d = ConvSpecStatic(
        rank=1,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    def rebind1d(idx: IndexList[1]) -> IndexList[fp32_1d.rank]:
        return rebind[IndexList[fp32_1d.rank]](idx)

    @always_inline
    def rebind1d_pad(idx: IndexList[2]) -> IndexList[2 * fp32_1d.rank]:
        return rebind[IndexList[2 * fp32_1d.rank]](idx)

    # fmt: off
    @always_inline
    def spec1d(N: Int, W: Int, C: Int, S: Int, F: Int, st: Int, di: Int, \
        pa: IndexList[2], ng: Int
    ) -> ConvSpec[fp32_1d]:
        return (
            ConvSpec[fp32_1d](
                n=N,
                input_dims=rebind1d(Index(W)),
                c=C,
                filter_dims=rebind1d(Index(S)),
                f=F,
                stride=rebind1d(Index(st)),
                dilation=rebind1d(Index(di)),
                pad=rebind1d_pad(pa),
                num_groups=ng,
            )
        )
    # fmt: on

    comptime fp32_2d = ConvSpecStatic(
        rank=2,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    def rebind2d(idx: IndexList[2]) -> IndexList[fp32_2d.rank]:
        return rebind[IndexList[fp32_2d.rank]](idx)

    @always_inline
    def rebind2d_pad(idx: IndexList[4]) -> IndexList[2 * fp32_2d.rank]:
        return rebind[IndexList[2 * fp32_2d.rank]](idx)

    @always_inline
    def spec2d(
        N: Int,
        H: Int,
        W: Int,
        C: Int,
        R: Int,
        S: Int,
        F: Int,
        st: IndexList[2],
        di: IndexList[2],
        pa: IndexList[4],
        ng: Int,
    ) -> ConvSpec[fp32_2d]:
        return ConvSpec[fp32_2d](
            n=N,
            input_dims=rebind2d(Index(H, W)),
            c=C,
            filter_dims=rebind2d(Index(R, S)),
            f=F,
            stride=rebind2d(st),
            dilation=rebind2d(di),
            pad=rebind2d_pad(pa),
            num_groups=ng,
        )

    comptime fp32_3d = ConvSpecStatic(
        rank=3,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    def rebind3d(idx: IndexList[3]) -> IndexList[fp32_3d.rank]:
        return rebind[IndexList[fp32_3d.rank]](idx)

    @always_inline
    def rebind3d_pad(idx: IndexList[6]) -> IndexList[3 * fp32_3d.rank]:
        return rebind[IndexList[3 * fp32_3d.rank]](idx)

    # 1D benchmarks for wavlm
    comptime if get_defined_string["model", "walvm"]() == "wavlm":
        bench_conv(m, spec1d(2, 16000, 1, 10, 512, 5, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 3199, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 1599, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 799, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 399, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 199, 512, 2, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 99, 512, 2, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 49, 1024, 128, 1024, 1, 1, Index(64, 64), 16))
    # fmt: off
    # 2D benchmarks for resnet
    elif get_defined_string["model", "wavlm"]() == "resnet50":
        bench_conv(m, spec2d(1, 14, 14, 256, 3, 3, 256, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 56, 56,  64, 3, 3,  64, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 28, 28, 128, 3, 3, 128, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 7, 7,   512, 3, 3, 512, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 224, 224, 3, 7, 7,  64, Index(2, 2), Index(1, 1), Index(3, 3, 3, 3), 1))
        bench_conv(m, spec2d(1, 56, 56, 128, 3, 3, 128, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 28, 28, 256, 3, 3, 256, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 14, 14, 512, 3, 3, 512, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 56, 56,  256, 1, 1, 512, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
        bench_conv(m, spec2d(1, 28, 28,  512, 1, 1, 1024, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
        bench_conv(m, spec2d(1, 14, 14, 1024, 1, 1, 2048, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
    # fmt: on

    m.dump_report()
