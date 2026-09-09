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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Verify that \u and \U unicode escape sequences in docstrings are resolved
# correctly. Docstrings use the same Lexer::getStringLiteralValue path as
# regular strings, so no special handling is needed — this test confirms that.


# CHECK: #lit.doc.string<"ASCII: h, Latin: \C3\A9, CJK: \E4\B8\AD, Emoji: \F0\9F\98\80"
def unicode_escapes():
    """ASCII: \u0068, Latin: \u00E9, CJK: \u4E2D, Emoji: \U0001F600"""
    pass
