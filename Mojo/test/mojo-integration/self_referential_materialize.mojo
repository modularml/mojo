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

# RUN: %mojo %s

# Materializing a comptime value that holds a pointer into its own storage.
# That pointer in the materialized object should point at the storage of
# the materialized object. This is a regression test for MOCO-813.
#
# The shapes here vary the nesting around the pointer, where in the object it
# points, and what else the value carries. A pointer that survives a move is
# aimed at its final home with `repoint`, since moving a value does not fix up
# pointers into the moved-from storage.

from std.testing import assert_equal, assert_true

comptime TAG: StaticString = "modular"
comptime MaybePtr = Optional[Pointer[Int8, MutUntrackedOrigin]]


struct SmallStr(Copyable, Movable):
    var buf: Array[Int8, 8]
    var data: Pointer[Int8, MutUntrackedOrigin]

    def __init__(out self):
        self.buf = Array[Int8, 8](uninitialized=True)
        self.data = self.buf.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

    def repoint(mut self):
        self.data = self.buf.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

    def skew(self) -> Int:
        """Distance from the self-pointer to the buffer it should target."""
        return Int(self.data) - Int(self.buf.unsafe_ptr())


struct Record(Copyable, Movable):
    var id: Int
    var name: SmallStr

    def __init__(out self):
        self.id = 7
        self.name = SmallStr()


struct Middle(Copyable, Movable):
    var tag: Int
    var s: SmallStr

    def __init__(out self):
        self.tag = 1
        self.s = SmallStr()


struct Deep(Copyable, Movable):
    var header: Int
    var mid: Middle

    def __init__(out self):
        self.header = 2
        self.mid = Middle()


struct Nest(Copyable, Movable):
    """Self-pointing at two nesting depths at once."""

    var buf: Array[Int8, 8]
    var data: Pointer[Int8, MutUntrackedOrigin]
    var inner: SmallStr

    def __init__(out self):
        self.buf = Array[Int8, 8](uninitialized=True)
        self.inner = SmallStr()
        self.data = self.buf.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.inner.repoint()

    def skew(self) -> Int:
        return Int(self.data) - Int(self.buf.unsafe_ptr())


struct Pair(Copyable, Movable):
    var a: SmallStr
    var b: SmallStr

    def __init__(out self):
        self.a = SmallStr()
        self.b = SmallStr()


struct Arr3(Copyable, Movable):
    var items: Array[SmallStr, 3]

    def __init__(out self):
        # `fill` copies one element into every slot, so each copy arrives
        # pointing at the original's buffer. Re-point each at its own.
        self.items = Array[SmallStr, 3](fill=SmallStr())
        for i in range(3):
            self.items[i].repoint()


struct Grid(Copyable, Movable):
    """Offsets come from two levels of array stride arithmetic."""

    var rows: Array[Array[SmallStr, 2], 2]

    def __init__(out self):
        var row = Array[SmallStr, 2](fill=SmallStr())
        self.rows = Array[Array[SmallStr, 2], 2](fill=row)
        for i in range(2):
            for j in range(2):
                self.rows[i][j].repoint()


struct HeapElem(Copyable, Movable):
    """The self-pointer and its target both live in the list's heap storage."""

    var items: List[SmallStr]

    def __init__(out self):
        self.items = List[SmallStr]()
        self.items.append(SmallStr())
        self.items[0].repoint()


struct HeapFirst(Copyable, Movable):
    var nums: List[Int]
    var s: SmallStr

    def __init__(out self):
        var tmp: List[Int] = [7, 8, 9]
        self.nums = tmp^
        self.s = SmallStr()


struct HeapLast(Copyable, Movable):
    var s: SmallStr
    var nums: List[Int]

    def __init__(out self):
        self.s = SmallStr()
        var tmp: List[Int] = [7, 8, 9]
        self.nums = tmp^


struct Interior(Copyable, Movable):
    """Two pointers aimed at the same unaligned interior byte."""

    var buf: Array[Int8, 8]
    var mid_a: Pointer[Int8, MutUntrackedOrigin]
    var mid_b: Pointer[Int8, MutUntrackedOrigin]

    def __init__(out self):
        self.buf = Array[Int8, 8](uninitialized=True)
        var base = self.buf.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.mid_a = base.unsafe_offset(3)
        self.mid_b = base.unsafe_offset(3)


struct BothWays(Copyable, Movable):
    """One pointer targets a field before it, the other a field after it."""

    var lo: Array[Int8, 8]
    var back: Pointer[Int8, MutUntrackedOrigin]
    var fwd: Pointer[Int8, MutUntrackedOrigin]
    var hi: Array[Int8, 8]

    def __init__(out self):
        self.lo = Array[Int8, 8](uninitialized=True)
        self.hi = Array[Int8, 8](uninitialized=True)
        self.back = self.lo.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.fwd = self.hi.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


struct Mixed(Copyable, Movable):
    """A self-pointer beside an external pointer and an absent one."""

    var buf: Array[Int8, 8]
    var here: Pointer[Int8, MutUntrackedOrigin]
    var there: Pointer[Byte, ImmUntrackedOrigin]
    var absent: MaybePtr

    def __init__(out self):
        self.buf = Array[Int8, 8](uninitialized=True)
        self.here = self.buf.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.there = TAG.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
        self.absent = MaybePtr(None)


struct Inner(Copyable, Movable):
    var to_header: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self, p: Pointer[Int, MutUntrackedOrigin]):
        self.to_header = p


struct Enclosing(Copyable, Movable):
    """The pointer targets a field outside the struct that holds it."""

    var header: Int
    var inner: Inner

    def __init__(out self):
        self.header = 1234
        self.inner = Inner(
            Pointer(to=self.header).unsafe_origin_cast[MutUntrackedOrigin]()
        )


@no_inline
def skew_by_ref(ref s: SmallStr) -> Int:
    # Taking a reference keeps the object in memory as a whole, so the
    # self-pointer is materialized rather than scalarized away.
    return Int(s.data) - Int(s.buf.unsafe_ptr())


def main() raises:
    comptime base = SmallStr()
    var m0 = materialize[base]()
    assert_equal(skew_by_ref(m0), 0, "standalone value")

    comptime rec = Record()
    var m1 = materialize[rec]()
    assert_equal(m1.name.skew(), 0, "value nested as a field")
    assert_equal(m1.id, 7, "field beside the nested value")

    comptime deep = Deep()
    var m2 = materialize[deep]()
    assert_equal(m2.mid.s.skew(), 0, "value nested three levels down")

    comptime nest = Nest()
    var m3 = materialize[nest]()
    assert_equal(m3.skew(), 0, "outer of two nested self-pointers")
    assert_equal(m3.inner.skew(), 0, "inner of two nested self-pointers")

    comptime pair = Pair()
    var m4 = materialize[pair]()
    assert_equal(m4.a.skew(), 0, "first of two self-pointers")
    assert_equal(m4.b.skew(), 0, "second of two self-pointers")
    assert_true(
        Int(m4.a.data) != Int(m4.b.data),
        "two self-pointers must target different buffers",
    )

    comptime arr = Arr3()
    var m5 = materialize[arr]()
    for i in range(3):
        assert_equal(m5.items[i].skew(), 0, "array element self-pointer")
    assert_true(
        Int(m5.items[0].data) != Int(m5.items[1].data),
        "array elements must target their own buffers",
    )

    comptime grid = Grid()
    var m6 = materialize[grid]()
    for i in range(2):
        for j in range(2):
            assert_equal(m6.rows[i][j].skew(), 0, "2-D array element")

    comptime heap = HeapElem()
    var m7 = materialize[heap]()
    assert_equal(m7.items[0].skew(), 0, "element inside heap storage")

    comptime hf = HeapFirst()
    var m8 = materialize[hf]()
    assert_equal(m8.s.skew(), 0, "self-pointer after a heap field")
    assert_equal(m8.nums[0] + m8.nums[1] + m8.nums[2], 24, "heap intact")

    comptime hl = HeapLast()
    var m9 = materialize[hl]()
    assert_equal(m9.s.skew(), 0, "self-pointer before a heap field")
    assert_equal(m9.nums[0] + m9.nums[1] + m9.nums[2], 24, "heap intact")

    comptime interior = Interior()
    var m10 = materialize[interior]()
    var ibase = Int(m10.buf.unsafe_ptr())
    assert_equal(Int(m10.mid_a) - ibase, 3, "unaligned interior target")
    assert_equal(Int(m10.mid_b) - ibase, 3, "second pointer, same target")

    comptime bw = BothWays()
    var m11 = materialize[bw]()
    assert_equal(
        Int(m11.back) - Int(m11.lo.unsafe_ptr()), 0, "target before the pointer"
    )
    assert_equal(
        Int(m11.fwd) - Int(m11.hi.unsafe_ptr()), 0, "target after the pointer"
    )

    comptime mixed = Mixed()
    var m12 = materialize[mixed]()
    assert_equal(
        Int(m12.here) - Int(m12.buf.unsafe_ptr()),
        0,
        "self-pointer beside other references",
    )
    assert_equal(Int(m12.there[]), 109, "external pointer left alone")
    assert_true(not Bool(m12.absent), "absent pointer left alone")

    comptime enc = Enclosing()
    var m13 = materialize[enc]()
    assert_equal(
        Int(m13.inner.to_header),
        Int(Pointer(to=m13.header).unsafe_origin_cast[MutUntrackedOrigin]()),
        "pointer to a field of the enclosing object",
    )
    m13.header = 999
    assert_equal(
        m13.inner.to_header[], 999, "write reaches the enclosing object's field"
    )

    # Two materializations of one comptime value are two independent objects.
    var c1 = materialize[base]()
    var c2 = materialize[base]()
    assert_equal(skew_by_ref(c1), 0, "first of two materializations")
    assert_equal(skew_by_ref(c2), 0, "second of two materializations")
    assert_true(
        Int(c1.data) != Int(c2.data),
        "separate materializations must not share a buffer",
    )

    _ = m0^
    _ = m1^
    _ = m2^
    _ = m3^
    _ = m4^
    _ = m5^
    _ = m6^
    _ = m7^
    _ = m8^
    _ = m9^
    _ = m10^
    _ = m11^
    _ = m12^
    _ = m13^
    _ = c1^
    _ = c2^
