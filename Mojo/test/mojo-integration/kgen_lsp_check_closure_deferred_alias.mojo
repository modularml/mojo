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

# ===----------------------------------------------------------------------=== #
#
# Exercises `kgen -lsp` (see kgen_lsp_check.mojo) on a nested closure whose
# body declares a `comptime` alias that is never itself referenced again and
# whose initializer names a `comptime` binding from the enclosing function.
# `inner` captures `y` by `mut`, giving it a real runtime capture, so it gets
# promoted to module scope rather than taking the no-capture "stateless
# closure" fast path.
#
# A single-target `comptime` alias like `unused` is left unresolved until
# something references it by name; `kgen -lsp`'s standalone check forces
# every declaration in the file to resolve, including ones no caller ever
# actually references, unlike a normal compile. Rather than force such a
# still-unresolved alias to resolve after promotion -- against a scope chain
# that promotion has already moved to a new parent, and a captured-parameter
# list that was already finalized before this dead code was ever parsed --
# the compiler leaves it unresolved, the same as any other dead decl a normal
# compile never reaches.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -lsp %s | FileCheck %s


# CHECK: lit.fn{{.*}}outer
def outer[N: Int](val: Int):
    comptime doubled = N * 2

    var y = val

    def inner(z: Int) {mut y}:
        comptime unused = doubled
        y += z

    inner(1)


def main():
    outer[3](1)
