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

# Test that struct members typed as SIMD[T, 1] (i.e. Scalar[T]) display
# correctly in the debugger.  This exercises the !kgen.simd<1, ...> fallback
# in unwrapToScalarOrPointer.


@fieldwise_init
struct ScalarMembers(TrivialRegisterPassable):
    var int_scalar: Int
    var bool_scalar: SIMD[.bool, 1]
    var uint8_scalar: UInt8

    def __init__(out self):
        self.int_scalar = Int(42)
        self.bool_scalar = SIMD[.bool, 1](True)
        self.uint8_scalar = UInt8(255)


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    var s = ScalarMembers()

    print("breakpoint")  # breakpoint

    keep_alive(s)
