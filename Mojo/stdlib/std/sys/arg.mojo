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
"""Implements functions and variables for interacting with execution and system
environment.
"""


from std.ffi import external_call


# TODO: When we have global variables, this should be a global list.
def argv() -> Span[StaticString, ImmStaticOrigin]:
    """Gets the list of command line arguments given to the `mojo` CLI.

    For example:

    ```mojo title="app.mojo"
    from std.sys import argv

    def main() raises:
        args = argv()
        for arg in args:
            print(arg)
    ```

    ```sh
    mojo app.mojo "Hello world"
    ```

    ```output
    app.mojo
    Hello world
    ```

    Returns:
        The list of command line arguments provided when mojo was invoked.
    """
    # SAFETY:
    #   It is valid to use `StringSlice` (and thus `StaticString`) here because
    #   `StringSlice` is guaranteed to be ABI compatible with `llvm::StringRef`.
    var result = Span[StaticString, ImmStaticOrigin]()
    external_call["KGEN_CompilerRT_GetArgV", NoneType](Pointer(to=result))
    return result
