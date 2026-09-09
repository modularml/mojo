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

# TODO: Currently this file doesn't test doc output—KGEN tests
# should do that—it only ensures that the code examples in the
# doc are syntactically correct.


@doc_hidden
def standalone_function():
    pass


struct Calculator:
    """A simple calculator struct demonstrating @doc_hidden."""

    var value: Int

    def __init__(out self, initial_value: Int = 0):
        """Creates a new Calculator with an initial value.

        Args:
            initial_value: The starting value for the calculator. Defaults to 0.
        """
        self.value = initial_value

    @doc_hidden
    def __init__(out self):
        """Internal initializer that should not appear in public documentation.

        This constructor exists for implementation purposes but users should
        prefer the constructor that takes an initial value.
        """
        self.value = 0

    def add(mut self, amount: Int):
        """Adds a value to the calculator.

        Args:
            amount: The value to add.
        """
        self.value += amount


struct Point:
    @doc_hidden
    def __init__(out self):
        pass


@doc_hidden
struct InternalHelper:
    pass


@doc_hidden
comptime INTERNAL_CONSTANT = 42


struct PublicStruct:
    @doc_hidden
    var implementation_detail: Int


def main():
    # Create a calculator using the public constructor
    var calc = Calculator(10)

    # Use the public method
    calc.add(5)
    print("Calculator value:", calc.value)
