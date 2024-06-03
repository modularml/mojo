# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from pathlib import Path

from memory import AddressSpace, DTypePointer, bitcast

alias _kStartTensorMarker = "["
alias _kEndTensorMarker = "]"
alias _kTensorFiller = "..., "
alias _kCompactMaxElemsToPrint = 7
alias _kCompactElemPerSide = _kCompactMaxElemsToPrint // 2


fn _serialize_elements_compact[
    serialize_fn: fn[T: Stringable] (elem: T) capturing -> None,
](ptr: DTypePointer, len: Int):
    serialize_fn(_kStartTensorMarker)
    if len < _kCompactMaxElemsToPrint:
        _serialize_elements_complete[serialize_fn=serialize_fn](ptr, len)
        serialize_fn(_kEndTensorMarker)
        return

    _serialize_elements_complete[serialize_fn=serialize_fn](
        ptr, _kCompactElemPerSide
    )
    serialize_fn(", ")
    serialize_fn(_kTensorFiller)
    _serialize_elements_complete[serialize_fn=serialize_fn](
        ptr + len - _kCompactElemPerSide, _kCompactElemPerSide
    )
    serialize_fn(_kEndTensorMarker)


fn _serialize_elements_complete[
    serialize_fn: fn[T: Stringable] (elem: T) capturing -> None,
](ptr: DTypePointer, len: Int):
    if len == 0:
        return
    serialize_fn(Scalar.load(ptr))
    for i in range(1, len):
        serialize_fn(", ")
        serialize_fn(Scalar.load(ptr, i))


fn _serialize_elements[
    serialize_fn: fn[T: Stringable] (elem: T) capturing -> None,
    compact: Bool = False,
](ptr: DTypePointer, len: Int):
    @parameter
    if compact:
        _serialize_elements_compact[serialize_fn=serialize_fn](ptr, len)
    else:
        _serialize_elements_complete[serialize_fn=serialize_fn](ptr, len)


fn _serialize[
    serialize_fn: fn[T: Stringable] (elem: T) capturing -> None,
    serialize_dtype: Bool = True,
    serialize_shape: Bool = True,
    serialize_end_line: Bool = True,
](ptr: DTypePointer, shape: List[Int]):
    var rank = len(shape)
    if rank == 0:
        if serialize_end_line:
            serialize_fn("\n")
        return

    # What we are doing is printing a series of 2D arrays if the dimension
    # is greater than 2. We limit print depth to 7 in height, width and depth
    # Assume the dimension is [1, 9, 100, 100]. We would print pick first 3
    # [1, 100, 100] matrices, and last 3 [1, 100, 100] matrices and print
    # them. In each of those we only print first, and last 3 rows and
    # first and last 3 columns. The intermediaries are filled with '...'
    # to indicate something is here but we are not displaying it.

    var column_elem_count = 1 if rank < 1 else shape[-1]
    # If the tensor is a rank-1 vector, then the number of rows is 1.
    var row_elem_count = 1 if rank < 2 else shape[-2]

    var matrix_elem_count = column_elem_count * row_elem_count

    # Open parens for every other dimension other than row_elem_count &
    # column_elem_count
    for i in range(2, rank):
        serialize_fn(_kStartTensorMarker)

    # We are basically printing a bunch of 2D tensors in succession
    # So if a tensor is of num_matrices * row_elem_count * column_elem_count
    # dimension, we print num_matrices tensors of
    # row_elem_count * column_elem_count dimension. num_matrices equals the
    # product of all dims other than last two.

    var num_matrices = 1
    for i in range(max(rank - 2, 0)):
        num_matrices *= shape[i]

    var matrix_idx = 0
    while matrix_idx < num_matrices:
        if matrix_idx > 0:
            serialize_fn(",\n")
        serialize_fn(_kStartTensorMarker)

        # Print row.
        var row_idx = 0
        while row_idx < row_elem_count:
            if row_idx > 0:
                serialize_fn("\n")

            _serialize_elements[serialize_fn=serialize_fn, compact=True](
                ptr
                + matrix_idx * matrix_elem_count
                + row_idx * column_elem_count,
                column_elem_count,
            )

            row_idx += 1

            # We are skipping printing comma after the last bracket.
            if row_idx != row_elem_count:
                serialize_fn(",")

            # Intermediate rows are filled with "..." and rowIdx is advanced to third
            # from last.
            if (
                row_elem_count >= _kCompactMaxElemsToPrint
                and row_idx == _kCompactElemPerSide
            ):
                serialize_fn("\n")
                serialize_fn(_kTensorFiller)
                row_idx = row_elem_count - _kCompactElemPerSide

        serialize_fn(_kEndTensorMarker)
        matrix_idx += 1

        # Again intermediate matrices are skipped for compactness.
        if (
            num_matrices >= _kCompactMaxElemsToPrint
            and matrix_idx == _kCompactElemPerSide
        ):
            serialize_fn("\n")
            serialize_fn(_kTensorFiller)
            matrix_idx = num_matrices - _kCompactElemPerSide

    # Now every element is printed. We just have to close all open parans.
    for i in range(2, rank):
        serialize_fn(_kEndTensorMarker)

    if serialize_dtype:
        serialize_fn(", dtype=")
        serialize_fn(ptr.type)

    if serialize_shape:
        serialize_fn(", shape=")
        var shape_str: String = ""
        for i in range(len(shape)):
            if i:
                shape_str += "x"
            shape_str += str(shape[i])
        serialize_fn(shape_str)

    if serialize_end_line:
        serialize_fn("\n")
