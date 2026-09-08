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
# RUN: mkdir -p %t.closure-dir
# RUN: mojo precompile %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: mojo -I %t.closure-dir %s 4 | FileCheck %s
# RUN: kgen-opt %t.closure-dir/closure.mojoc | FileCheck %s -check-prefix=CHECK-PACK

# CHECK-PACK: lit.trait.decl @"def(x: Int) -> Int"
# CHECK-PACK: definesClosure

from closure import printIt, defineIt
from std.sys import argv


def aThing(y: Int):
    def myclosure(x: Int) {var} -> Int:
        return y + x

    printIt[type_of(myclosure)](myclosure, y)
    defineIt(y)


def main() raises:
    # CHECK: 8
    # CHECK: 8
    var x = atol(argv()[1])
    aThing(x)
