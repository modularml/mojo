# crash_lib.mojo — reproducer for the Mojo 1.0.0b2 JIT instability.
from std.memory import stack_allocation


@extern("ms_stack_alloc")
def _ms_stack_alloc(
    bytes: Int,
    out_base: UnsafePointer[UnsafePointer[Byte, MutAnyOrigin], MutUntrackedOrigin],
    out_top: UnsafePointer[UnsafePointer[Byte, MutAnyOrigin], MutUntrackedOrigin],
) abi("C") -> Int32:
    ...


struct S(ImplicitlyCopyable, ImplicitlyDeletable):
    var a: Int
    var arr: InlineArray[Int, 21]

    def __init__(out self, a: Int):
        self.a = a
        self.arr = InlineArray[Int, 21](fill=0)


def make(n: Int) raises -> S:
    var slots = stack_allocation[2, UnsafePointer[Byte, MutAnyOrigin]]()
    var rc = _ms_stack_alloc(n, slots, slots + 1)
    if rc != 0:
        raise Error("alloc failed rc=" + String(rc))
    var s = S(Int(slots[]))
    return s^
