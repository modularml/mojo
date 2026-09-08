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


# The invalid dtype, built directly from the underlying MLIR attribute so this
# debugger test does not depend on any public `DType` alias. This exercises the
# type system's rendering of `scalar<invalid>` (see PrimitiveTypesTest.cpp).
comptime _invalid_dtype = DType(
    mlir_value=__mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
)


@fieldwise_init
struct A(ImplicitlyCopyable):
    var x: Pointer[Scalar[_invalid_dtype], MutUntrackedOrigin]

    def __init__(out self):
        var y = alloc[Int8]({count = 1}).unsafe_leak()
        self.x = y.unsafe_bitcast[Scalar[_invalid_dtype]]()


def test_key_element() raises:
    var a = A()
    print("bp")  # breakpoint
    _ = a


def main() raises:
    test_key_element()
