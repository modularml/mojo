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
# This exercises the parser through the language-server parse path
# (`parseFileForLSP`) via `kgen-translate -import-mojo -lsp`, rather than the
# regular compiler `importMojoFile` path.
#
# Besides checking that the user's own decl is emitted, this asserts on output
# that is unique to LSP mode: `lit.unresolved_import` ops. These are the lazy
# named imports that `resolveSignaturesForLSP` deliberately leaves unparsed, and
# that the LSP path preserves because it skips DCE (`eraseUnreachableDecls`).
# The regular compiler path resolves and then strips all imports, so it never
# emits a `lit.unresolved_import`. A second RUN confirms that absence, so the
# test fails if `-lsp` silently falls back to the compiler path.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -lsp %s | FileCheck %s
# RUN: %parse-mojo-isolated %s | FileCheck %s --check-prefix=REGULAR

# CHECK: lit.struct.decl @BoxedInt
# REGULAR: lit.struct.decl @BoxedInt
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int


def main():
    pass

# LSP mode keeps imports lazy as unresolved-import ops; the compiler path
# resolves and strips them via DCE. The implicit prelude import is the clearest
# case: every module gets a `from std.prelude import *`, which LSP preserves as
# an unresolved wildcard (its symbols are never eagerly pulled in) and the
# compiler path strips. `REGULAR-NOT: lit.unresolved` covers both the wildcard
# and any named unresolved imports.
# CHECK: lit.unresolved_wildcard_import from <0, ["std", "builtin", "stubs"]>
# REGULAR-NOT: lit.unresolved
