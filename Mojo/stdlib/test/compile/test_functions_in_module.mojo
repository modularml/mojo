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

from std.reflection import get_linkage_name
from std.testing import assert_equal


def foo():
    pass


def bar(x: Int) -> Int:
    return 1


# NOTE: this is intentionally in the middle here, to ensure that the intrinsic
# correctly resolves signatures that are declared after the call.
comptime funcs = __functions_in_module()


def main() raises:
    var expected_names = [
        "test_functions_in_module::foo()",
        "test_functions_in_module::bar(::SIMD[DType.int, 1])",
        (
            "test_functions_in_module::bar(::SIMD[DType.int,"
            " 1],::SIMD[DType.int, 1])"
        ),
        "test_functions_in_module::foobar(z:::SIMD[DType.float64, 1])",
    ]

    comptime for i in range(len(funcs)):
        comptime name = get_linkage_name[funcs[i]]()
        assert_equal(name, expected_names[i])


def bar(y: Int, z: Int):
    pass


def foobar(*, z: Float64 = 1.6) raises:
    pass
