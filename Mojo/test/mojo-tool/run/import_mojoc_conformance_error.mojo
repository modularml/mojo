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

# A conformance failure against a struct imported from a `.mojoc` must produce
# the ordinary diagnostic. The package sources are deleted after precompiling
# so the note falls back to a synthesized signature, which renders the
# defaulted parameter `H: Helper = default_helper` whose underlying struct
# declaration is never materialized from the bytecode; printing it must not
# crash (MOCO-4508).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/deps %t.dir/src
# RUN: cp -r %S/inputs/precompiled_conformance/pkg %t.dir/src/pkg
# RUN: mojo precompile %t.dir/src/pkg -o %t.dir/deps/pkg.mojoc
# RUN: rm -rf %t.dir/src
# RUN: not mojo build -I %t.dir/deps %s -o %t.dir/main 2>&1 | FileCheck %s

# CHECK: error: 'Container' parameter 'T' has 'Marked' type, but value has type 'AnyStruct[NotMarked]'
# CHECK: note: 'Container' declared here
# CHECK: struct Container[T: Marked, H: Helper = DefaultHelper[Int(4)]]
# CHECK-SAME: # note - synthetic signature

from pkg import Container


@fieldwise_init
struct NotMarked(Copyable, Movable):
    var value: Int


def main():
    _ = Container[NotMarked]()
