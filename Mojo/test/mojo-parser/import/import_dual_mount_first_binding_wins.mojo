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

# Which of two names for one package names its contents: the first one bound,
# not the shorter, the longer, or the one derived from the layout. Same package
# and same roots as import_dual_mount_source_package.mojo with the two imports
# swapped, so the winning name swaps with them - in the diagnostic and in the
# symbol path the type is uniqued under.

# RUN: %parse-mojo-isolated -verify-diagnostics \
# RUN:   -I=%S/inputs/dual_mount_pkg/lib \
# RUN:   -I=%S/inputs/dual_mount_pkg/lib/mylib %s \
# RUN: | FileCheck %s

# CHECK: !lit.struct<@utils::@a::@Thing>

# expected-note @below {{'utils' is the name used in error messages and debug info}}
from utils.a import Thing, make
# expected-warning @below {{'mylib.utils' and 'utils' name the same package; remove the duplicate import root or file that reaches it twice}}
from mylib.utils.a import Thing as SameThing


def unwrap(value: SameThing) -> Int:
    return value.x


def use() -> Int:
    var value: Thing = make()
    return unwrap(value)
