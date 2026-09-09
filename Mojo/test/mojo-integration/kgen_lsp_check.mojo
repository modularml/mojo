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
# Exercises the `kgen -lsp` command, which reproduces how the language server
# processes an open document (`MojoDocument::checkModuleSemantics`): an
# error-tolerant parse (`parseFileForLSP`) followed by the module-level check
# pipeline (`runCheckLITPipeline`) on the per-decl clone. The resulting checked
# module IR is printed to stdout (diagnostics, if any, go to stderr).
#
# This asserts that the command emits the checked IR and that the user's own
# declarations from this file survive into it, mirroring how `kgen-translate
# -lsp` is tested under mojo-parser. Patterns are matched loosely so they hold
# whether the module prints in generic or pretty form.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -lsp %s | FileCheck %s


# CHECK: lit.struct.decl{{.*}}BoxedInt
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int


# CHECK: lit.fn{{.*}}get_boxed
def get_boxed() -> BoxedInt:
    return BoxedInt(42)
