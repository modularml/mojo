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

# Math functions such as `hypot` and `expm1` are calls into libm, which the C
# compiler driver used for linking does not pull in on its own, so `mojo build`
# has to ask for it. Apple platforms keep those entry points in libSystem.
# UNSUPPORTED: system-darwin

# Stand in for the linker driver with a command that prints its arguments, to
# check that the flag reaches the link. The configured system libraries are
# dropped because they carry their own `-lm` on some toolchains, which would
# hide a missing one here.
# RUN: env MODULAR_MOJO_MAX_SYSTEM_LIBS= \
# RUN:   MODULAR_MOJO_MAX_LINKER_DRIVER=/bin/echo %mojo-build %s -o %t.args \
# RUN:   2>&1 | FileCheck %s --check-prefix=LINK-ARGS
# LINK-ARGS: -lm

# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

from std.math import expm1, hypot, tanh
from std.sys.arg import argv
from std.testing import assert_almost_equal


def main() raises:
    # `argv` keeps the inputs opaque to constant folding, so the libm calls
    # survive into the linked executable.
    var one = Float64(len(argv()))

    assert_almost_equal(hypot(3.0 * one, 4.0 * one), 5.0)
    assert_almost_equal(tanh(one), 0.7615941559557649)
    assert_almost_equal(expm1(one), 1.718281828459045)

    # CHECK: libm math linked
    print("libm math linked")
