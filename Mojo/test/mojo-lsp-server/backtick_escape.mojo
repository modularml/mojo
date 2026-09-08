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

# RUN: mojo-lsp-simple-client --fail-on-diagnostics %s

# Regression test: docstrings with escape sequences in prose must not produce
# backtick identifier errors. Covers three classes of escape sequences:
#   1. `\\t`, `\\xNN` etc: source shrinks (2+ bytes → 1), shifts byte offset.
#   2. `\n`: produces a real newline in processed string (2 source bytes → 1),
#      breaking any approach that counts newlines in the processed string.
#   3. `\<newline>` line continuation: source newline vanishes in processed
#      (2 source bytes → 0), also breaking newline-count approaches.


struct ByteShrink:
    """Escape sequences that shrink the processed string.

    Characters: "\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e".

    ```mojo
    var x = 1 + 1
    ```
    """

    def __init__(out self):
        pass


struct EmbeddedNewline:
    """Escape sequence that inserts a real newline into processed string.

    A \n escape in prose produces a newline (0x0A) in the processed string,
    which would cause a line-counting approach to map the code block one source
    line too late.

    ```mojo
    var x = 1 + 1
    ```
    """

    def __init__(out self):
        pass


struct LineContinuation:
    """Escape sequence that removes a newline from the processed string.

    A backslash-newline line continuation consumes a source newline but \
produces no output, so the processed string has one fewer line than the source.

    ```mojo
    var x = 1 + 1
    ```
    """

    def __init__(out self):
        pass
