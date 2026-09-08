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

# Regression test for MOCO-3517: a decorator preceding a struct must not be
# mis-attributed to the plain statement that follows the struct body.
# The specific bug was: "@fieldwise_init struct Foo { ... } var f = ..."
# produced "'var' statement does not support decorators; remove the decorator".
#
# CHECK-NOT: does not support decorators


def example_decorator_then_var():
    """Decorated struct followed by a var statement.

    ```mojo
    @fieldwise_init
    struct Point:
        var x: Int
        var y: Int
    var p = Point(1, 2)
    ```
    """
    pass


def example_decorator_then_try():
    """Decorated struct followed by a try statement.

    ```mojo
    @fieldwise_init
    struct MaybePoint:
        var x: Int
        var y: Int
    try:
        var p = MaybePoint(1, 2)
    except:
        pass
    ```
    """
    pass


def example_stacked_decorators():
    """Multiple stacked decorators followed by a var statement.

    ```mojo
    @fieldwise_init
    @fieldwise_init
    struct Pair:
        var a: Int
        var b: Int
    var p = Pair(3, 4)
    ```
    """
    pass


def example_decorator_last_line():
    """Decorator with no following declaration (last line of code block).

    Should not crash or produce spurious diagnostics.

    ```mojo
    @fieldwise_init
    ```
    """
    pass


def example_decorator_after_statement():
    """Decorator as last line after a preceding statement.

    Should not crash or produce spurious diagnostics.

    ```mojo
    var x = 1
    @fieldwise_init
    ```
    """
    pass
