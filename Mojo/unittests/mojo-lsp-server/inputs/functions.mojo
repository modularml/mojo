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


struct SomeStruct[size: Int, other_param: Bool]:
    """Docstring for SomeStruct.

    More docstring for SomeStruct.

    Constraints:
        The constraints of SomeStruct.

    Parameters:
        size: The size of SomeStruct.
        other_param: Another param.
    """

    def __init__(
        out self,
        imm borrowed_input: Int,
        init_arg: Int,
        var owned_input: Int,
        *init_kargs: Int,
    ):
        """Init documentation.

        Args:
            borrowed_input: An imm argument.
            init_arg: An Int argument.
            owned_input: An owned argument.
            init_kargs: Multiple arguments.
        """
        _ = init_arg
        _ = init_kargs
        pass

    @staticmethod
    def static_method() -> Int:
        return 420

    def bar(mut self):
        def non_capturing_nested_function():
            pass

    async def async_function(mut self):
        @__parameter
        def parameter_nested_function():
            pass

        def another_nested_function():
            pass

    def function_that_raises(
        mut self, arg_in_function_that_raises: Int
    ) raises -> String:
        """A function that raises.

        Args:
            arg_in_function_that_raises: An arg in a function with by-ref result.
        """
        return "foo"

    def function_with_param[Param1: Int, Param2: Int](mut self):
        """A function with param.

        Parameters:
          Param1: An Int param.
          Param2: Another Int param.
        """
        pass


def exported_function():
    "This is an exported function."

    def a_closure():
        pass

    a_closure()


def def_function() raises -> Int:
    return 120


def main():
    print("foo")
