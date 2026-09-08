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

# RUN: mojo-lsp-simple-client --fail-on-diagnostics %s > %t 2>&1; true
# RUN: FileCheck %s < %t

# Regression test: REPL buffer identifiers must not be re-imported as modules.
# When a docstring has two consecutive ```mojo``` blocks and the first defines a
# variable, the LSP REPL carries it into the second block as a persistent
# variable.  Its type is stored as a SymbolRef rooted at the REPL buffer name,
# which is the source file path plus a wrapper suffix (e.g.
# "/abs/path/file.mojo wrapper_at(N)").  Before the fix, importModuleState used
# name.contains('.') to detect dotted module paths; REPL buffer names also
# contain '.' (from ".mojo"), so they triggered importRelativeModuleState, which
# split on '.' and tried to locate a phantom module "/abs/path/file".
#
# CHECK-NOT: unable to locate module


def example():
    """Two consecutive mojo code blocks; second block references first's variable.

    ```mojo
    struct Foo:
        var x: Int
        def __init__(out self, x: Int):
            self.x = x
    var f = Foo(1)
    ```

    ```mojo
    print(f.x)
    ```
    """
    pass
