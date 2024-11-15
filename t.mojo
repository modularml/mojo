from memory import memcpy,UnsafePointer
from sys import sizeof

alias Bytes = List[Byte]

fn pointer_bitcast[
    To: AnyType
](ptr: Pointer) -> Pointer[To, ptr.origin, ptr.address_space, *_, **_] as out:
    return __type_of(out)(
        _mlir_value=__mlir_op.`lit.ref.from_pointer`[
            _type = __type_of(out)._mlir_type
        ](
            UnsafePointer(__mlir_op.`lit.ref.to_pointer`(ptr._value))
            .bitcast[To]()
            .address
        )
    )

@value
# @register_passable("trivial")
struct MyInt:
    var value: Int

    fn __del__(owned self):
        print('deleting')


fn as_bytes(value: Int) raises -> List[Byte]:
    """Convert the integer to a byte array.

    Returns:
        The byte array.
    """
    var ptr = Pointer.address_of(value.value)
    var byte_ptr = pointer_bitcast[Byte](ptr)
    var len = sizeof[__mlir_type.index]()
    # var res = List(ptr=UnsafePointer.address_of(byte_ptr[]), length=len, capacity=len)
    var res = List[Byte](capacity=len)
    for i in range(len):
        item = UnsafePointer.address_of(byte_ptr[]).load(i)
        print('i={}, b={}'.format(i, int(item)))
        res.append(UnsafePointer.address_of(byte_ptr[]).load(i))
    return res


fn as_bytes_unsafe(value: Int) raises -> List[Byte]:
    """Convert the integer to a byte array.

    Returns:
        The byte array.
    """

    # @parameter
    # if is_big_endian() and not big_endian:
    #     value = byte_swap(value)
    # elif not is_big_endian() and big_endian:
    #     value = byte_swap(value)

    var ptr = UnsafePointer.address_of(value.value)
    var len = sizeof[MyInt]()
    var byte_ptr = ptr.bitcast[Byte]()
    var dest_ptr = UnsafePointer[Byte].alloc(len)
    memcpy(dest=dest_ptr, src=byte_ptr, count=len)
    res = List(ptr=dest_ptr, length=len, capacity=len)
    for i in range(len):
        print('i={}, b={}'.format(i, int(res[i])))
    return res

fn main() raises:
    var x = Int(1_000_000_000)
    var bytes = as_bytes(x)
    print('bytes={}'.format(String(bytes)))
    print('UNSAFE')
    var bytes_unsafe = as_bytes_unsafe(x)
    print('bytes_unsafe={}'.format(String(bytes_unsafe)))
