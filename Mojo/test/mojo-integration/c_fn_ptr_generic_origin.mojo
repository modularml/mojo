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

# A pointer parameter with an inferred origin leaves the callee generic over a
# parametric-mutability origin, so the `lit.bind_params` that specializes it
# binds an origin value that prints in its sugared `(mutcast ...)` form.

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

from std.ffi import c_char


comptime MyFnType = def(ImmPointer[c_char, _], Int) thin abi("C") -> Int32


# CHECK-LABEL: lit.fn @"call_it
# CHECK: lit.bind_params
# CHECK-SAME: :origin<false> (mutcast mut=
# CHECK-SAME: to !lit.generator<
def call_it(c_fn: MyFnType, s: StringSlice) -> Int32:
    return c_fn(
        s.as_bytes().unsafe_ptr().unsafe_bitcast[c_char](),
        s.byte_length(),
    )
