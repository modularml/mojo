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
# RUN: %mojo %s

from testing import assert_equal


def main():
    test_str()
    test_repr()
    test_format_to()
    test_type_from_none()


def test_str():
    assert_equal(NoneType().__str__(), "None")


def test_repr():
    assert_equal(NoneType().__repr__(), "None")


def test_format_to():
    assert_equal(String.write(NoneType()), "None")


struct FromNone:
    var value: Int

    @implicit
    fn __init__(out self, none: NoneType):
        self.value = -1

    # FIXME: None literal should be of NoneType not !kgen.none.
    @always_inline
    @implicit
    fn __init__(out self, none: __mlir_type.`!kgen.none`):
        self = NoneType()

    @implicit
    fn __init__(out self, value: Int):
        self.value = value


def test_type_from_none():
    obj = FromNone(5)

    obj = FromNone(None)

    # -------------------------------------
    # Test implicit conversion from `None`
    # -------------------------------------

    fn foo(arg: FromNone):
        pass

    # FIXME:
    #   This currently fails, because it requires 2 "hops" of conversion:
    #       1. !kgen.none => NoneType
    #       2. NoneType => FromNone
    # foo(None)
    #
    #   But, interestingly, this does not fail?
    var obj2: FromNone = None
