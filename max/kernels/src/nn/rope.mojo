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

from std.collections import OptionalReg
from std.math import gcd
from std.sys.info import _current_target, simd_width_of

from max.algorithm.functional import elementwise
from std.complex import ComplexSIMD
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from layout import (
    Coord,
    CoordLike,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
    coord,
    coord_to_index_list,
)
from nn._ragged_utils import get_batch_from_row_offsets

from std.utils import IndexList


@always_inline
def _rope[
    dtype: DType,
    freq_dtype: DType,
    width: SIMDLength,
](val: SIMD[dtype, width], freq: SIMD[freq_dtype, width]) -> SIMD[dtype, width]:
    var x_re, x_im = val.cast[freq_dtype]().deinterleave()
    var f_re, f_im = freq.deinterleave()
    var r = ComplexSIMD(x_re, x_im) * ComplexSIMD(f_re, f_im)
    return rebind[SIMD[dtype, width]](r.re.interleave(r.im).cast[dtype]())


# In GGUF, weights are organized as real, imag, real, imag, real, imag, …,
# while in safetensors, the data is stored as real, …, real, imag, …, imag.
# This function return the indices for the real and imaginary part.
@always_inline
def get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> Tuple[Int, Int]:
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
def get_identity_rope_coeff[width: Int, dtype: DType]() -> SIMD[dtype, width]:
    # Creates a SIMD vector with real parts set to 1 and imaginary parts to
    # 0, effectively making the RoPE transformation an identity operation.
    return rebind[SIMD[dtype, width]](
        SIMD[dtype, width // 2](1).interleave(SIMD[dtype, width // 2](0))
    )


@always_inline
def apply_rope[
    dtype: DType,
    freq_dtype: DType,
    width: SIMDLength,
    //,
    *,
    interleaved: Bool,
    alignment: Int,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[3], val: SIMD[dtype, width]
    ) -> None,
](
    x: TileTensor[dtype, ...],
    idx: IndexList[3],
    freq_val: SIMD[freq_dtype, width],
    output_fn: OutputFn,
):
    comptime rank = 3
    comptime assert rank - 1 >= 0
    var indices = get_safetensors_idx(idx[rank - 1], x.static_shape[rank - 1])
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    comptime width_2 = width // 2

    var val: SIMD[dtype, width]

    comptime if interleaved:
        var coord = Coord(idx)
        val = x.load[width=width, alignment=1](coord)
    else:
        var re_coord = Coord(pos_re)
        var im_coord = Coord(pos_im)
        val = rebind[SIMD[dtype, width]](
            x.load[width=width_2, alignment=1](re_coord).interleave(
                x.load[width=width_2, alignment=1](im_coord)
            )
        )

    var res = _rope(val, freq_val)

    comptime if interleaved:
        output_fn[alignment=alignment](idx, res)
    else:
        var output_re, output_im = res.deinterleave()
        output_fn[alignment=alignment](pos_re, output_re)
        output_fn[alignment=alignment](pos_im, output_im)


@always_inline
def rope_ragged[
    dtype: DType,
    freq_dtype: DType,
    *,
    interleaved: Bool,
    target: StaticString,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[3], val: SIMD[dtype, width]
    ) -> None,
    rope_first: Bool = False,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    x: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[.uint32, ...],
    start_pos: TileTensor[.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    context: DeviceContext,
    output_fn: OutputFn,
    position_ids: OptionalReg[
        TileTensor[.uint32, PositionIdsLayoutType, ImmutAnyOrigin]
    ] = None,
) raises where (
    input_row_offsets.flat_rank == 1
    and start_pos.flat_rank == 1
    and freqs_cis.flat_rank == 2
):
    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static rope_dim for freqs_cis"
    comptime head_size = Int(x.static_shape[2])
    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime unroped_dim = head_size - rope_dim
    comptime has_nope = unroped_dim > 0
    # `rope_first` puts the rotated columns at the front of each head (the
    # DSA Indexer layout, where Q and K are chunked as `pe, nope`) rather than
    # at the back (the MLA layout). It only changes which side of the head the
    # passthrough columns sit on, so it also moves where `freqs_cis` starts
    # being indexed.
    comptime freq_col_offset = 0 if rope_first else unroped_dim
    # Non-interleaved RoPE pairs column j with j + head_size // 2
    # (`get_safetensors_idx`), a rotate-half split spanning the whole head.
    # That only lines up with the roped region when the region is the
    # trailing half, so a leading roped region would rotate roped columns
    # against passthrough ones.
    comptime assert interleaved or not (
        rope_first and has_nope
    ), "rope_ragged: rope_first partial RoPE requires interleaved layout"

    # Extract the position_ids raw pointer + row stride into primitive locals so
    # they can be `@__copy_capture`'d into the device kernel closure. Capturing
    # the `OptionalReg[TileTensor]` directly does not marshal its device pointer
    # into the Metal kernel arg struct -- the closure then reads a null/stale
    # pointer, so every token reads freqs row 0 and RoPE collapses to identity
    # (producing incoherent FLUX latents -> noise). Mirrors the validated
    # `rope_split_store` path. When position_ids is absent, capture a dummy
    # pointer (never dereferenced; guarded by `has_position_ids`) reusing
    # start_pos' (uint32) storage so the captured value has the same type in
    # both branches.
    var has_position_ids: Bool = Bool(position_ids)
    comptime PtrType = type_of(position_ids.value().ptr.as_imm())
    var pos_ids_ptr: PtrType
    var pos_ids_stride: Int
    if has_position_ids:
        pos_ids_ptr = rebind[PtrType](position_ids.value().ptr.as_imm())
        # Row stride from the layout, not `dim[1]()`: the kernel steps whole
        # position_ids rows (`section_idx * pos_ids_stride`), so use the actual
        # dim-0 stride rather than assuming a row-major-contiguous layout where
        # it happens to equal `dim[1]`.
        pos_ids_stride = Int(position_ids.value().layout.stride[0]().value())
    else:
        pos_ids_ptr = rebind[PtrType](start_pos.ptr.as_imm())
        pos_ids_stride = 0

    @always_inline
    def rope_fn[width: Int, alignment: Int = 1](idx_arg: Coord) {var}:
        comptime assert idx_arg.rank == 3, "Invalid rank passed to rope kernel"
        comptime assert freqs_cis.flat_rank >= 2

        comptime if width == 1:
            assert False, (
                "RoPE kernel called with simd width = 1, We should never be"
                " here. This is indicative of an uneven last dimension of"
                " the rope tensor. Ensure the model's head_size is"
                " divisible by the simd width of your target hardware."
            )
            return
        else:
            var idx = rebind[IndexList[3]](coord_to_index_list(idx_arg))

            var global_token_idx = idx[0]

            var batch_idx: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var head_dim_idx = idx[2]

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = start_pos[batch_idx] + UInt32(token_idx)

            var position_ids_idx = Int(post_seq_idx)
            if has_position_ids:
                comptime if mrope_section:
                    var section_idx = 0

                    comptime for i in range(len(mrope_section.value())):
                        comptime val = Int(mrope_section.value()[i].value())
                        if head_dim_idx < val:
                            section_idx = i
                            break
                    position_ids_idx = Int(
                        pos_ids_ptr[
                            section_idx * pos_ids_stride + global_token_idx
                        ]
                    )
                else:
                    position_ids_idx = Int(pos_ids_ptr[global_token_idx])

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_unroped_region: Bool
            comptime if rope_first:
                is_unroped_region = head_dim_idx >= rope_dim
            else:
                is_unroped_region = head_dim_idx < unroped_dim

            var f_c_temp: SIMD[freq_dtype, width]

            comptime if has_nope:
                if is_unroped_region:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    f_c_temp = freqs_cis.load[width=width, alignment=1](
                        (
                            Scalar[freqs_cis.linear_idx_type](position_ids_idx),
                            Scalar[freqs_cis.linear_idx_type](
                                head_dim_idx - freq_col_offset
                            ),
                        )
                    )
            else:
                f_c_temp = freqs_cis.load[width=width, alignment=1](
                    (
                        Scalar[freqs_cis.linear_idx_type](position_ids_idx),
                        Scalar[freqs_cis.linear_idx_type](head_dim_idx),
                    )
                )
            apply_rope[
                interleaved=interleaved,
                alignment=alignment,
            ](x, idx, f_c_temp, output_fn)

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, rope_dim)

    comptime if mrope_section:
        comptime for i in range(len(mrope_section.value())):
            comptime assert (
                mrope_section.value()[i].static_value % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    comptime assert (
        kernel_simd_width >= 2 and rope_dim % kernel_simd_width == 0
    ), (
        "Rope kernel simd width must be between 2 and rope_dim and divisible by"
        " rope_dim. Ensure the model's head_size is divisible by the simd width"
        " of your target hardware."
    )

    comptime if is_cpu[target]():
        elementwise[simd_width=kernel_simd_width, target=target](
            rope_fn, x.layout.shape_coord(), context
        )
    else:
        elementwise[
            simd_width=kernel_simd_width,
            target=target,
            _trace_description="rope",
        ](rope_fn, x.layout.shape_coord(), context)
