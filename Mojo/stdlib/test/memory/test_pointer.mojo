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

from std.memory import MaybeUninit
from test_utils import check_write_to
from std.testing import TestSuite
from std.testing import (
    assert_equal,
    assert_not_equal,
    assert_true,
    assert_false,
)


def test_copy_reference_explicitly() raises:
    var a = [1, 2, 3]

    var b = Pointer(to=a)
    var c = b.copy()

    c[][0] = 4
    assert_equal(a[0], 4)
    assert_equal(b[][0], 4)
    assert_equal(c[][0], 4)


def test_equality() raises:
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    assert_true(Pointer(to=a) == Pointer(to=a))
    assert_true(Pointer(to=b) == Pointer(to=b))
    assert_true(Pointer(to=a) != Pointer(to=b))


def test_str() raises:
    var a = Int(42)
    var a_ref = Pointer(to=a)
    assert_true(String(a_ref).startswith("0x"))


def test_write_to() raises:
    var a = Int(42)
    check_write_to(Pointer(to=a), contains="0x", is_repr=False)
    var s = String("hello")
    check_write_to(Pointer(to=s), contains="0x", is_repr=False)


def test_write_repr_to() raises:
    var n = Int(42)
    check_write_to(
        Pointer(to=n),
        contains=(
            "Pointer[mut=True, SIMD[DType.int, 1],"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        Pointer(to=n).as_imm(),
        contains=(
            "Pointer[mut=False, SIMD[DType.int, 1],"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )

    var s = String("hello")
    check_write_to(
        Pointer(to=s),
        contains=(
            "Pointer[mut=True, String, address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )


def test_pointer_to() raises:
    var local = 1
    assert_not_equal(0, Pointer(to=local)[])


# Test pointer merging with ternary operation.
def test_merge() raises:
    var a: List = [1, 2, 3]
    var b: List = [4, 5, 6]

    def inner(cond: Bool, x: Int, mut a: List[Int], mut b: List[Int]):
        var either = Pointer(to=a) if cond else Pointer(to=b)
        either[].append(x)

    inner(True, 7, a, b)
    inner(False, 8, a, b)

    assert_equal(a, [1, 2, 3, 7])
    assert_equal(b, [4, 5, 6, 8])


def test_nicheable() raises:
    var x = 42
    comptime PointerType = Pointer[Int, ImmOrigin(origin_of(x))]

    assert_equal(PointerType.niche_count(), 1)

    var memory = MaybeUninit[PointerType]()

    PointerType.write_niche(Pointer(to=memory))
    assert_true(PointerType.isa_niche(Pointer(to=memory)))

    memory.unsafe_write(Pointer(to=x))
    assert_false(PointerType.isa_niche(Pointer(to=memory)))


# We don't actually need to run this,
# but Mojo's exclusivity check shouldn't complain
def _test_get_imm() raises -> Int:
    def foo(x: ImmPointer[Int, ...], y: ImmPointer[Int, ...]) -> Int:
        return x[]

    var x = Int(0)
    return foo(Pointer(to=x), Pointer(to=x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
