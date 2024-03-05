# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max
from pathlib import Path

from memory.unsafe import AddressSpace, DTypePointer, bitcast
from tensor import Tensor, TensorShape, TensorSpec

alias _kStartTensorMarker = "["
alias _kEndTensorMarker = "]"
alias _kTensorFiller = "..., "
alias _kCompactMaxElemsToPrint = 7
alias _kCompactElemPerSide = _kCompactMaxElemsToPrint // 2

# Serialization constants
alias _SERIALIZATION_MAJOR_FORMAT: UInt32 = 0
alias _SERIALIZATION_MINOR_FORMAT: UInt32 = 1
# 0x93 🔥 0x93
alias _SERIALIZATION_HEADER = StaticTuple[Int8, 6](
    0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x93
)


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
    serialize_fn(ptr.load())
    for i in range(1, len):
        serialize_fn(", ")
        serialize_fn(ptr.load(i))


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
](ptr: DTypePointer, shape: TensorShape):
    var rank = shape.rank()
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
        serialize_fn(shape.__str__())
    if serialize_end_line:
        serialize_fn("\n")


fn _serialize_as_tensor[
    type: AnyRegType
](inout object: type) -> Tensor[DType.int8]:
    """Serialize the given object into a Tensor of bytes.

    Args:
      object: Object to serialize.

    Returns:
      Tensor containing the bytes of object.
    """
    var self_ptr = bitcast[Int8](Pointer.address_of(object))
    alias size = sizeof[type]()
    var bytes = Tensor[DType.int8](size)
    memcpy(bytes.data(), DTypePointer[DType.int8](self_ptr.address), size)
    return bytes ^


fn _serialize_to_file[type: DType](tensor: Tensor[type], path: Path) raises:
    """Serialize given tensor to file. This method preserves
       shape and datatype information.

    Args:
      tensor: Tensor to serialize.
      path: Path of file.
    """
    var header_size = len(_SERIALIZATION_HEADER)
    var header_bytes = Tensor[DType.int8](header_size)

    for i in range(header_size):
        header_bytes.store(i, _SERIALIZATION_HEADER[i])

    var major_format: UInt32 = _SERIALIZATION_MAJOR_FORMAT
    var major_format_bytes = _serialize_as_tensor(major_format)
    var minor_format: UInt32 = _SERIALIZATION_MINOR_FORMAT
    var minor_format_bytes = _serialize_as_tensor(minor_format)
    var spec_size: UInt32 = sizeof[TensorSpec]()
    var spec_size_bytes = _serialize_as_tensor(spec_size)
    var spec = tensor.spec()
    var spec_bytes = _serialize_as_tensor[TensorSpec](spec)

    var bytes = Tensor[DType.int8](
        header_bytes.num_elements()
        + major_format_bytes.num_elements()
        + minor_format_bytes.num_elements()
        + spec_size_bytes.num_elements()
        + spec_bytes.num_elements()
        + tensor.num_elements() * type.sizeof()
    )
    var copied: Int = 0

    @always_inline("nodebug")
    fn _copy_bytes(
        inout dest: Tensor[DType.int8], offset: Int, src: Tensor[DType.int8]
    ) -> Int:
        var size = src.num_elements()
        memcpy(
            dest.data() + offset,
            src.data(),
            size,
        )
        return offset + size

    copied = _copy_bytes(bytes, copied, header_bytes)
    copied = _copy_bytes(bytes, copied, major_format_bytes)
    copied = _copy_bytes(bytes, copied, minor_format_bytes)
    copied = _copy_bytes(bytes, copied, spec_size_bytes)
    # TODO: Numpy aligns this to 64 byte boundary.
    copied = _copy_bytes(bytes, copied, spec_bytes)

    # TODO: Avoid this copy.
    memcpy(
        bytes.data() + copied,
        bitcast[DType.int8](tensor.data()),
        tensor.num_elements() * type.sizeof(),
    )
    copied += tensor.num_elements() * type.sizeof()

    debug_assert(bytes.num_elements() == copied, "expected these to be same.")

    bytes.tofile(path)
