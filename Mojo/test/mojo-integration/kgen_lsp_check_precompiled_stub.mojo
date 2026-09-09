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
# Regression test for cloneDeclModuleForCompilation giving imported-but-unused
# stdlib functions a valid stub body. `std` is precompiled for this
# directory's mojo_deps (see BUILD.bazel), so checking any file here clones
# the whole precompiled std module, including functions this file never
# calls, like `__MLIRType.copy` -- those are left as signature-only `external` stubs.
# FileCheck the resulting stub directly: it must have a real block argument
# for its `ptr` parameter and a single block terminated by `lit.end_fn
# unresolved`, rather than crashing (empty argument list) or failing
# verification (0-block region).
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -lsp %s | FileCheck %s


def main():
    pass


# CHECK: lit.fn @"copy(::__MLIRType{{.*}}(%self: {{.*}}%__result__: {{.*}}attributes {defaultFnRef = @std::@traits::@copyable::@Copyable::@"copy($0)"
# CHECK-SAME: external
# CHECK-NEXT: lit.end_fn unresolved
# CHECK-NEXT: }
