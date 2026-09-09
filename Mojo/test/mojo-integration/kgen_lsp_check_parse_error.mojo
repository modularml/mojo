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
# Regression test for `kgen -lsp` reporting crashes the language server cannot
# have. The server refuses to run semantic checking once the parse reported an
# error (`MojoDocument::checkModuleSemantics` returns early on
# `hasParserErrors`) because the check passes -- LowerSemanticCF,
# VerifyParameters, CheckLifetimes -- are not written to tolerate the partial
# IR a failed parse leaves behind. `kgen -lsp` used to run the check pipeline
# unconditionally and segfault in those passes on inputs the server never
# checks, making the tool close to pure noise as an LSP oracle on broken code.
#
# A file mid-edit almost always has an error, so this is the common case for a
# language server. The command must now report the diagnostic and stop, like
# the server does.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -lsp %s 2>&1 | FileCheck %s


def broken():
    var x = this_name_does_not_exist


# The diagnostic is reported, and nothing downstream of it runs: no module IR
# on stdout, and no crash in the check passes the server would have skipped.
# CHECK: error: use of unknown declaration 'this_name_does_not_exist'
# CHECK-NOT: lit.fn
# CHECK-NOT: PLEASE submit a bug report
