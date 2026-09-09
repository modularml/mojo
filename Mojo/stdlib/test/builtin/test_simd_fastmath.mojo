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

from std.builtin.simd import FastMathFlag
from std.compile import compile_info
from std.testing import TestSuite, assert_false, assert_true


def test_simd_fma_fastmath() raises:
    def my_fma(a: Float32, b: Float32, c: Float32) -> Float32:
        return a.fma[FastMathFlag.FAST](c, b)

    var asm = compile_info[my_fma, emission_kind="llvm"]()

    assert_true(" call fast float @llvm.fma.f32" in asm)


def main() raises:
    var suite = TestSuite()

    suite.test[test_simd_fma_fastmath]()

    suite^.run()
