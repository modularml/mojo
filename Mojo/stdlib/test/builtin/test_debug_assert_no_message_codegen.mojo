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

# A `debug_assert` with no message must not allocate a message buffer, which on
# GPU targets can exceed the stack frame limit (MSTDL-2240). Checking the
# reported message cannot catch that, so check the generated code instead.
#
# `--emit llvm` is unoptimized, so the failure path is still present. The assert
# mode is pinned rather than inherited, because `%mojo-build` only enables
# assertions when `MOJO_ENABLE_ASSERTIONS_IN_TESTS` is set, and without them the
# assert bodies are elided entirely.
#
# The conditions must be opaque to the compiler. A condition it can prove true
# folds the failure path away before it reaches the IR, leaving nothing to check.
#
# The check names the buffer type rather than an allocation size, so it keeps
# its meaning if the size changes. `_FixedWriteBuffer` reaches the IR through the
# mangled names of the writer specializations the message path instantiates.
# `--implicit-check-not` scans the whole file; a `CHECK-NOT` directive would only
# cover the region before the first positive match.

# RUN: mkdir -p %t
# RUN: %mojo-build-no-debug-no-assert %s -D ASSERT=all -o %t/test_debug_assert_no_message_codegen.ll --emit llvm
# RUN: FileCheck %s --input-file=%t/test_debug_assert_no_message_codegen.ll --implicit-check-not='_FixedWriteBuffer'

# The static message is emitted only on the no-message path, and the trailing
# nul is what the reported length counts on.
# CHECK: c"assertion failed\00"


from std.sys.arg import argv


def bool_overload(x: Int):
    assert x == 0


def closure_overload(x: Int):
    def cond() {x} -> Bool:
        return x == 0

    debug_assert(cond)


def main():
    var opaque = len(argv())
    bool_overload(opaque)
    closure_overload(opaque)
