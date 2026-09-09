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

from std.math import ceildiv
from std.sys.info import align_of, size_of

from quantization import Q4sym
from std.testing import assert_true
from layout import TileTensor, row_major

from std.utils import IndexList


def _run_test_quant[group_size: Int, tolerance: Float32]() -> Bool:
    var uniform = SIMD[.float32, group_size]()
    for i in range(group_size):
        uniform[i] = Float32(i)
    uniform -= Float32(group_size // 2)

    var skew_pos = uniform + 30
    var skew_neg = uniform - 30
    var skew_slightly_pos = uniform + 1.842
    var skew_slightly_neg = uniform - 1.842
    var big_range = uniform * 1000
    var unitary = SIMD[.float32, group_size](1.0)

    def run_fake_quant(input_vec: SIMD[.float32, group_size]) -> Bool:
        var packed_result = Q4sym[group_size, .float32](input_vec)
        var decoded_result = packed_result.decode_fully()
        print("input_vec        :", input_vec)
        print("fakeq_result     :", decoded_result)
        print("abs-err          :", input_vec - decoded_result)
        print(
            "max abs-err      :", abs(input_vec - decoded_result).reduce_max()
        )

        var l2_norm_err = (
            (input_vec - decoded_result) * (input_vec - decoded_result)
        ).reduce_add()
        l2_norm_err = l2_norm_err**0.5
        var l2_norm_input = (input_vec * input_vec).reduce_add() ** 0.5
        var rel_l2_norm = l2_norm_err / l2_norm_input
        print("rel-l2-norm      :", rel_l2_norm)

        return rel_l2_norm < tolerance

    var allPass: Bool = True
    allPass = allPass and run_fake_quant(uniform)
    allPass = allPass and run_fake_quant(skew_pos)
    allPass = allPass and run_fake_quant(skew_neg)
    allPass = allPass and run_fake_quant(skew_slightly_pos)
    allPass = allPass and run_fake_quant(skew_slightly_neg)
    allPass = allPass and run_fake_quant(big_range)
    allPass = allPass and run_fake_quant(unitary)
    return allPass


def test_fake_quant_error[l2_tolerance: Float32]() raises:
    # Tests round-trippability of encoding/decoding groups of numbers
    print("------------test_fake_quant_error------------")
    print("********** GROUP SIZE 08 **********")
    var g8_result = _run_test_quant[8, l2_tolerance]()
    print("G08 PASS" if g8_result else "G08 FAIL")
    print()
    assert_true(g8_result)

    print("********** GROUP SIZE 16 **********")
    var g16_result = _run_test_quant[16, l2_tolerance]()
    print("G16 PASS" if g16_result else "G16 FAIL")
    print()
    assert_true(g16_result)

    print("********** GROUP SIZE 32 **********")
    var g32_result = _run_test_quant[32, l2_tolerance]()
    print("------------end test_fake_quant_error------------")
    print("G32 PASS" if g32_result else "G32 FAIL")
    print()
    assert_true(g32_result)


def test_alignment_and_size():
    # Tests the total size and alignment of structs is as expected
    print("-------test_alignment_and_size-------")
    print("StructType, size_of, Alignment")
    print(
        "Q5sym[32, DType.float32]",
        size_of[Q4sym[32, DType.float32]](),
        align_of[Q4sym[32, DType.float32]](),
    )
    print(
        "Q5sym[16, DType.float32]",
        size_of[Q4sym[16, DType.float32]](),
        align_of[Q4sym[16, DType.float32]](),
    )
    print(
        "Q5sym[8, DType.float32]",
        size_of[Q4sym[8, DType.float32]](),
        align_of[Q4sym[8, DType.float32]](),
    )
    # Calculation for group size 8:
    # - 2 bytes for fp16 scale
    # - 8 // 2 = 4 bytes for the low bits
    # 2 + 4 = 6
    # Bits per weight: (8 * 6) / 8 = 6bpw
    comptime assert size_of[Q4sym[8]]() == 6

    # Calculation for group size 16:
    # - 2 bytes for fp16 scale
    # - 16 // 2 = 8 bytes for the low bits
    # 2 + 8 = 10
    # Bits per weight: (8 * 10) / 16 = 5bpw
    comptime assert size_of[Q4sym[16]]() == 10

    # Calculation for group size 32:
    # - 2 bytes for fp16 scale
    # - 32 // 2 = 16 bytes for the low bits
    # 2 + 16 = 18
    # Bits per weight: (8 * 18) / 32 = 4.5bpw
    comptime assert size_of[Q4sym[32]]() == 18
    print("-------end test_alignment_and_size-------")
    print()


def _read_write_to_tensors[
    group_size: Int,
    rtol: Float32,
    atol: Float32,
    num_elements: Int = 64,
    rank: Int = 1,
]() -> Bool:
    # Write a quantized tensor, and then immediately decode, making sure results
    # are close.

    # Allocate and populate tensor to encode
    # Buffer with the original data
    var data_matrix_backing = Array[Float32, num_elements](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var data_matrix_ptr: MutPointer[
        Float32, origin_of(data_matrix_backing)
    ] = data_matrix_backing.unsafe_ptr()
    var data_matrix = TileTensor(
        ptr=data_matrix_ptr, layout=row_major[num_elements]()
    )

    # Tensor to store the packed data
    comptime assert num_elements % group_size == 0
    comptime num_blocks = ceildiv(num_elements, group_size)
    comptime block_size = size_of[Q4sym[group_size]]()
    var packed_blob_backing = Array[UInt8, num_blocks * block_size](fill={})
    var packed_blob = TileTensor(
        packed_blob_backing, row_major[num_blocks * block_size]()
    )

    # Tensor to store the dequantized data
    var out_data_matrix_backing = Array[Float32, num_elements](fill=Float32(0))
    var out_data_matrix = TileTensor(
        out_data_matrix_backing, row_major[num_elements]()
    )

    Q4sym[group_size, DType.float32].quantize_and_write_to_tensor(
        data_matrix.make_dynamic[.int64]().to_layout_tensor(),
        packed_blob.make_dynamic[.int64]().to_layout_tensor(),
        IndexList[
            type_of(data_matrix.make_dynamic[.int64]().to_layout_tensor()).rank
        ](num_elements),
    )

    Q4sym[group_size, DType.float32].dequantize_and_write_to_tensor(
        packed_blob.make_dynamic[.int64]().to_layout_tensor(),
        out_data_matrix.make_dynamic[.int64]().to_layout_tensor(),
        IndexList[
            type_of(
                out_data_matrix.make_dynamic[.int64]().to_layout_tensor()
            ).rank
        ](num_elements),
    )

    var allClose: Bool = True
    # See if it prints the correct results!
    for i in range(num_elements):
        var localRDiff = abs(
            (data_matrix[i] - out_data_matrix[i]) / (data_matrix[i] + 1e-10)
        )
        var acceptableErr = abs(data_matrix[i] * rtol) + atol
        print(
            "fake-quantized:",
            out_data_matrix[i],
            " vs original:",
            data_matrix[i],
            " -- rel-diff: ",
            localRDiff,
        )
        allClose = allClose and (localRDiff <= acceptableErr)
    return allClose


def test_read_write_to_tensors[rtol: FloatLiteral, atol: FloatLiteral]() raises:
    print("------------test_read_write_to_tensors------------")

    print("********** GROUP SIZE 08 **********")
    var g8_result = _read_write_to_tensors[8, rtol, atol]()
    print("G08 PASS" if g8_result else "G08 FAIL")
    print()
    assert_true(g8_result)

    print("********** GROUP SIZE 16 **********")
    var g16_result = _read_write_to_tensors[16, rtol, atol]()
    print("G16 PASS" if g16_result else "G16 FAIL")
    print()
    assert_true(g16_result)

    print("********** GROUP SIZE 32 **********")
    var g32_result = _read_write_to_tensors[32, rtol, atol]()
    print("G32 PASS" if g32_result else "G32 FAIL")
    print()
    assert_true(g32_result)

    print("------------end test_read_write_to_tensors------------")
    print()


def main() raises:
    comptime l2_tolerance = 0.1

    test_fake_quant_error[l2_tolerance]()

    test_read_write_to_tensors[rtol=0.1, atol=1.0]()

    # Tests via compile-time constraints on size_of(Q4Sym)
    test_alignment_and_size()
