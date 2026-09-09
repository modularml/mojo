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

# Checks the diagnostic notes attached to conflicting imported declarations
# when source location information isn't available: the note should print a
# synthesized signature of each conflicting decl rather than nothing. Like
# diagnostics_in_missing_locs.mojo, this precompiles the packages and deletes
# the original source.

# RUN: mkdir -p %t
# RUN: cp -r %S/inputs/clash_a %t
# RUN: cp -r %S/inputs/clash_b %t
# RUN: mojo precompile %t/clash_a -o %t/clash_a.mojoc
# RUN: mojo precompile %t/clash_b -o %t/clash_b.mojoc
# RUN: rm -r %t/clash_a %t/clash_b
# RUN: not %mojo-build -I %t %s 2>&1 | FileCheck %s

from clash_a import Gadget
from clash_b import Gadget

from clash_a import helper
from clash_b import helper


def main():
    # CHECK: error: import of 'Gadget' is ambiguous
    # CHECK: note: 'Gadget' declared here
    # CHECK-NEXT: struct Gadget[size: Int]    # note - synthetic signature
    # CHECK: note: 'Gadget' also declared here
    # CHECK-NEXT: struct Gadget    # note - synthetic signature
    # The first import remains usable after the ambiguity error.
    var g = Gadget[4]()

    # CHECK: error: import of 'helper' is ambiguous
    # CHECK: note: 'helper' declared here
    # CHECK-NEXT: def helper(x: Int) -> Int    # note - synthetic signature
    # CHECK: note: 'helper' also declared here
    # CHECK-NEXT: struct helper    # note - synthetic signature
    _ = helper(1)
