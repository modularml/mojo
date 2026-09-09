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
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/simple_cases/function_trailing_comma.py
#
# ===----------------------------------------------------------------------=== #


def f(
    a,
):
    d = {
        "key": "value",
    }
    tup = (1,)


def f2(
    a,
    b,
):
    d = {
        "key": "value",
        "key2": "value2",
    }
    tup = (
        1,
        2,
    )


def f(
    a: int = 1,
):
    call(
        arg={
            "explode": "this",
        }
    )
    call2(
        arg=[1, 2, 3],
    )
    x = {
        "a": 1,
        "b": 2,
    }["a"]
    if (
        a
        == {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
        }["a"]
    ):
        pass


def xxxxxxxxxxxxxxxxxxxxxxxxxxxx() -> Set[
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]:
    json = {
        "k": {
            "k2": {
                "k3": [
                    1,
                ]
            }
        }
    }


# The type annotation shouldn't get a trailing comma since that would change its type.
# Relevant bug report: https://github.com/psf/black/issues/2381.
def some_function_with_a_really_long_name() -> (
    returning_a_deeply_nested_import_of_a_type_i_suppose
):
    pass


def some_method_with_a_really_long_name(
    very_long_parameter_so_yeah: str, another_long_parameter: int
) -> (
    another_case_of_returning_a_deeply_nested_import_of_a_type_i_suppose_cause_why_not
):
    pass


def func() -> (
    also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(
        this_shouldn_t_get_a_trailing_comma_too
    )
):
    pass


def func() -> (
    (
        also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(
            this_shouldn_t_get_a_trailing_comma_too
        )
    )
):
    pass


# Make sure inner one-element tuple won't explode
some_module.some_function(
    argument1, (one_element_tuple,), argument4, argument5, argument6
)

# Inner trailing comma causes outer to explode
some_module.some_function(
    argument1,
    (
        one,
        two,
    ),
    argument4,
    argument5,
    argument6,
)

# output


def f(
    a,
):
    d = {
        "key": "value",
    }
    tup = (1,)


def f2(
    a,
    b,
):
    d = {
        "key": "value",
        "key2": "value2",
    }
    tup = (
        1,
        2,
    )


def f(
    a: int = 1,
):
    call(
        arg={
            "explode": "this",
        }
    )
    call2(
        arg=[1, 2, 3],
    )
    x = {
        "a": 1,
        "b": 2,
    }["a"]
    if (
        a
        == {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
        }["a"]
    ):
        pass


def xxxxxxxxxxxxxxxxxxxxxxxxxxxx() -> Set[
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]:
    json = {
        "k": {
            "k2": {
                "k3": [
                    1,
                ]
            }
        }
    }


# The type annotation shouldn't get a trailing comma since that would change its type.
# Relevant bug report: https://github.com/psf/black/issues/2381.
def some_function_with_a_really_long_name() -> (
    returning_a_deeply_nested_import_of_a_type_i_suppose
):
    pass


def some_method_with_a_really_long_name(
    very_long_parameter_so_yeah: str, another_long_parameter: int
) -> (
    another_case_of_returning_a_deeply_nested_import_of_a_type_i_suppose_cause_why_not
):
    pass


def func() -> (
    also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(
        this_shouldn_t_get_a_trailing_comma_too
    )
):
    pass


def func() -> (
    (
        also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(
            this_shouldn_t_get_a_trailing_comma_too
        )
    )
):
    pass


# Make sure inner one-element tuple won't explode
some_module.some_function(
    argument1, (one_element_tuple,), argument4, argument5, argument6
)

# Inner trailing comma causes outer to explode
some_module.some_function(
    argument1,
    (
        one,
        two,
    ),
    argument4,
    argument5,
    argument6,
)
