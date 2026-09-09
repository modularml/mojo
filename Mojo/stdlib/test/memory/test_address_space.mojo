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

from std.memory import AddressSpace
from test_utils import check_write_to
from std.testing import assert_equal, TestSuite


comptime ADDRESS_SPACE_STRINGS = [
    (AddressSpace.GENERIC, "AddressSpace.GENERIC"),
    (AddressSpace.GLOBAL, "AddressSpace.GLOBAL"),
    (AddressSpace.SHARED, "AddressSpace.SHARED"),
    (AddressSpace.CONSTANT, "AddressSpace.CONSTANT"),
    (AddressSpace.LOCAL, "AddressSpace.LOCAL"),
    (AddressSpace.SHARED_CLUSTER, "AddressSpace.SHARED_CLUSTER"),
    (AddressSpace(42), "AddressSpace(42)"),
]


def test_address_space_write_to() raises:
    for address_space, expected in materialize[ADDRESS_SPACE_STRINGS]():
        check_write_to(address_space, expected=expected, is_repr=False)


def test_address_space_write_repr_to() raises:
    for address_space, expected in materialize[ADDRESS_SPACE_STRINGS]():
        check_write_to(address_space, expected=expected, is_repr=True)


def test_address_space_named_values() raises:
    # The built-in GPU address spaces resolve to their fixed ABI values and are
    # not shadowed by the target-extensible `__getattr_param__` lookup that
    # backs target-specific names.
    assert_equal(Int(AddressSpace.GENERIC), 0)
    assert_equal(Int(AddressSpace.GLOBAL), 1)
    assert_equal(Int(AddressSpace.SHARED), 3)
    assert_equal(Int(AddressSpace.CONSTANT), 4)
    assert_equal(Int(AddressSpace.LOCAL), 5)
    assert_equal(Int(AddressSpace.SHARED_CLUSTER), 7)
    assert_equal(Int(AddressSpace.BUFFER_RESOURCE), 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
