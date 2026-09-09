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

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

# A `comptime` type alias makes a variable's def/use sites flow through a
# `kgen.rebind`. CheckLifetimes must look through `kgen.rebind` symmetrically
# for both lifetime end markers (already handled) and lifetime start markers,
# otherwise a def that only goes through a rebind gets an unbalanced
# `lit.var.lifetime.end` with no matching `lit.var.lifetime.start`.


struct Inner(Copyable, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


comptime Aliased = Inner


def use(a: Inner):
    pass


# CHECK-LABEL: lit.fn @"main()"
def main() raises:
    # CHECK: %v = lit.var.decl "v"

    # The initial def of `v` is rebound from the alias to the concrete type
    # before being passed as the `__init__` call's byref_result argument.
    # CHECK: kgen.rebind %v
    # CHECK-NEXT: lit.var.lifetime.start %v
    var v: Aliased = Inner(0)
    use(v)

    # CHECK: lit.var.lifetime.end %v

    # The reassignment's def goes through a `kgen.rebind` too. This is the
    # start marker that used to be dropped.
    # CHECK: kgen.rebind %v
    # CHECK-NEXT: lit.var.lifetime.start %v
    v = Inner(2)
    use(v)

    # CHECK: lit.var.lifetime.end %v
