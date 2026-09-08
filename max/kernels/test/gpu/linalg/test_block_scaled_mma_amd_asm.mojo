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
"""Pins the register footprint of the CDNA4 `f8f6f4` block-scaled MFMA operands.

`cdna4_block_scaled_mfma` sizes operand fragments in powers of two, so FP6
travels in a 32-byte fragment despite carrying only 24 payload bytes. One FP6
MFMA (`cbsz:2 blgp:2`, accumulator width 16) on gfx950 measures:

    operand     wrapper         A/B span  VGPRs  bytes
    <6 x i32>   @always_inline   6         17     48
    <8 x i32>   @always_inline   6         17     48
    <6 x i32>   @no_inline       6         31     48
    <8 x i32>   @no_inline       6         33     64

The padding is free only while InstCombine narrows the operand and the wrapper
stays inlined. The span reads 6 in every row, so the tests also pin whole-kernel
counts and require no call.

Cross-compiles to gfx950, so it needs no AMD GPU.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, lane_id
from std.testing import assert_equal, assert_true
from std.utils import StaticTuple

from max.gpu.host.compile import _compile_code, get_gpu_target

from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _single_mfma_kernel[
    matrix_format: CDNA4F8F6F4MatrixFormat,
    accum_width: Int,
](
    a_in: ImmPointer[UInt8, ImmutAnyOrigin],
    b_in: ImmPointer[UInt8, ImmutAnyOrigin],
    scales: ImmPointer[Int32, ImmutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    comptime fragment_width = matrix_format.simd_width()
    var lane = Int(lane_id())

    var a_frag = a_in.load[width=fragment_width](lane * fragment_width)
    var b_frag = b_in.load[width=fragment_width](lane * fragment_width)
    var acc = SIMD[.float32, accum_width](0.0)

    cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
        acc, a_frag, b_frag, scales[0], scales[1]
    )

    dst.store(lane * accum_width, acc)


def _directive_value(asm: String, directive: String) raises -> Int:
    var start = asm.find(directive)
    assert_true(start >= 0, "assembly is missing " + directive)
    var line_end = asm.find("\n", start)
    var line = asm[byte=start:line_end]
    var colon = line.find(":")
    return Int(line[byte = colon + 1 :].strip())


def _register_span(operand: StringSlice) raises -> Int:
    """Returns how many registers a `v[lo:hi]` instruction operand covers."""
    var text = String(operand).strip()
    var open_bracket = text.find("[")
    var colon = text.find(":")
    var close_bracket = text.find("]")
    assert_true(
        open_bracket > 0 and colon > open_bracket and close_bracket > colon,
        "expected a v[lo:hi] register range, got " + text,
    )
    var low = Int(text[byte = open_bracket + 1 : colon])
    var high = Int(text[byte = colon + 1 : close_bracket])
    return high - low + 1


def _assert_register_footprint[
    matrix_format: CDNA4F8F6F4MatrixFormat,
    accum_width: Int,
    expected_registers: Int,
    expected_vgprs: Int,
]() raises:
    comptime kernel = _single_mfma_kernel[matrix_format, accum_width]
    var asm = _compile_code[kernel, target=get_gpu_target["gfx950"]()]().asm

    var mfma = String("")
    var matches = 0
    for line in asm.splitlines():
        if "v_mfma_scale_f32_" in line:
            mfma = String(String(line).strip())
            matches += 1
    assert_equal(
        matches, 1, "expected exactly one block-scaled MFMA in the assembly"
    )

    # Operand 0 is the mnemonic plus the accumulator; 1 and 2 are A and B.
    var operands = mfma.split(",")
    assert_true(
        len(operands) >= 3,
        "MFMA line did not split into a mnemonic and three operands",
    )
    assert_equal(_register_span(operands[1]), expected_registers)
    assert_equal(_register_span(operands[2]), expected_registers)

    assert_true(
        "s_swappc" not in asm,
        "the MFMA wrapper must be inlined into the kernel",
    )
    assert_equal(_directive_value(asm, ".vgpr_count"), expected_vgprs)
    assert_equal(_directive_value(asm, ".vgpr_spill_count"), 0)
    assert_equal(_directive_value(asm, ".sgpr_spill_count"), 0)


def test_fp8_operands_use_eight_registers() raises:
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3, 16, 8, 26]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3, 4, 8, 22]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2, 16, 8, 26]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2, 4, 8, 22]()


def test_fp6_operands_use_six_registers() raises:
    """The 32-byte FP6 fragment must not turn into an eight-register operand.

    The VGPR counts stay well below the FP8 cases above, which is what shows FP6
    is not paying for its two registers of padding.
    """
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3, 16, 6, 17]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3, 4, 6, 18]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2, 16, 6, 17]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2, 4, 6, 18]()


def test_fp4_operands_use_four_registers() raises:
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1, 16, 4, 17]()
    _assert_register_footprint[CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1, 4, 4, 14]()


def main() raises:
    print("== test_fp8_operands_use_eight_registers")
    test_fp8_operands_use_eight_registers()
    print("== test_fp6_operands_use_six_registers")
    test_fp6_operands_use_six_registers()
    print("== test_fp4_operands_use_four_registers")
    test_fp4_operands_use_four_registers()
    print("PASS")
