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

# Regression test: a name re-exported along two import paths produces two
# resolved `lit.import` gates in one scope (keep-and-gate). Once that package is
# precompiled, the resolved gates are serialized; reloading the bytecode must
# collapse them instead of reporting "invalid redefinition of 'foo'". This was
# originally only caught transitively by precompiling //max:layout (std re-exports
# names such as `isnan`, `nan`, `alloc`, `is_apple_gpu` along multiple paths).
#
# NOTE: this file must NOT be named after the package it imports, or resolution
# hits a recursive self-reference (MOCO-1946).

# RUN: mkdir -p %t.reexport-gate-dup
# RUN: mojo precompile %S/inputs/reexport_gate_dup -o %t.reexport-gate-dup/reexport_gate_dup.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.reexport-gate-dup %s | FileCheck %s

from reexport_gate_dup import use_foo


# CHECK-LABEL: lit.fn @"main
def main():
    # CHECK: lit.call {{.*}}@reexport_gate_dup::@{{.*}}@"{{(use_foo|foo)}}
    _ = use_foo()
