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
# This pins down the core resolution-depth divergence between the two parse
# paths for a decl imported from another file:
#
#   * `-lsp` (parseFileForLSP): referenced external decls are resolved to
#     SIGNATURE depth only. The body is never materialized, so the imported
#     function is emitted as a stub whose region is just `lit.end_fn unresolved`.
#   * regular (importMojoFile): the referenced import is fully BODY-resolved
#     (and kept, since it is referenced), so its real body ops are emitted.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -lsp -I %S/inputs %s | FileCheck %s
# RUN: %parse-mojo-isolated -I %S/inputs %s | FileCheck %s --check-prefix=REGULAR

from helper import compute_secret_value


def main():
    var x = compute_secret_value()


# Under -lsp the imported function's body is left unresolved: the line right
# after the function header is the unresolved terminator, with no body ops.
# CHECK:      lit.fn @"compute_secret_value()
# CHECK-NEXT: lit.end_fn unresolved

# Under the regular compiler path the same function is fully body-resolved: its
# real body (the literal `7` from `var a`) is materialized, and it is never left
# as an unresolved stub.
# REGULAR:      lit.fn @"compute_secret_value()
# REGULAR-NEXT: lit.var.decl "a"
# REGULAR-NOT:  lit.end_fn unresolved
