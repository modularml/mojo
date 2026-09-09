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

# Imports from 'mojo_module.so'
import mojo_module as feature_overview  # type: ignore[import-not-found]
import pytest


def test_case_return_arg_tuple() -> None:
    result = feature_overview.case_return_arg_tuple(
        1, 2, "three", ["four", "four.B"]
    )

    assert result == (1, 2, "three", ["four", "four.B"])


def test_case_raise_empty_error() -> None:
    with pytest.raises(ValueError) as cm:
        feature_overview.case_raise_empty_error()

    assert cm.value.args == ()


def test_case_raise_string_error() -> None:
    with pytest.raises(ValueError) as cm:
        feature_overview.case_raise_string_error()

    assert cm.value.args == ("sample value error",)


def test_case_mojo_raise() -> None:
    with pytest.raises(Exception) as cm:
        feature_overview.case_mojo_raise()

    assert cm.value.args == ("Mojo error",)


def test_case_mojo_mutate() -> None:
    list_obj = [1, 3, 5]
    feature_overview.case_mojo_mutate(list_obj)
    assert list_obj[0] == 2


def test_case_downcast_unbound_type() -> None:
    with pytest.raises(Exception) as err:
        feature_overview.case_downcast_unbound_type(5)

    assert err.value.args == (
        "No Python type object registered for Mojo type with name: "
        "mojo_module.NonBoundType",
    )


def test_case_create_mojo_type_instance() -> None:
    person = feature_overview.Person()

    assert type(person).__name__ == "Person"

    assert person.name() == "John Smith"

    assert repr(person) == "Person(name='John Smith', age=Int(123))"

    with pytest.raises(Exception) as cm:
        person.change_name("John Modular")

    assert cm.value.args == ("cannot make name longer than current name",)

    person.change_name("John Doe")
    assert person.name() == "John Doe"

    # Test that an error is raised if passing any arguments to the initializer
    with pytest.raises(ValueError) as cm:
        person = feature_overview.Person("John")

    assert cm.value.args == (
        (
            "unexpected arguments passed to default initializer"
            " function of wrapped Mojo type"
        ),
    )


def test_failed_mojo_object_creation_does_not_del() -> None:
    """Test that if a Mojo object was not fully initialized due to an
    exception raised during construction, Python will not call its
    __del__ method."""

    # Test that an error is raised if passing any arguments to the initializer
    with pytest.raises(ValueError) as cm:
        feature_overview.FailToInitialize("illegal argument")

    assert cm.value.args == (
        (
            "unexpected arguments passed to default initializer"
            " function of wrapped Mojo type"
        ),
    )

    # If we reach this point, we know `FailToInitialize.__del__()` was not
    # called, because it aborts.


def test_case_create_mojo_object_in_mojo() -> None:
    # Returns a new Mojo 'String' object, not derived from
    # any of the arguments. This requires creating a PythonObject from
    # within Mojo code.
    string = feature_overview.create_string()

    assert repr(string) == "'Hello'"


def test_case_mutate_wrapped_object() -> None:
    mojo_int = feature_overview.Int()
    assert repr(mojo_int) == "Int(0)"

    feature_overview.incr_int(mojo_int)
    assert repr(mojo_int) == "Int(1)"

    feature_overview.incr_int(mojo_int)
    assert repr(mojo_int) == "Int(2)"

    # --------------------------------
    # Test passing the wrong arguments
    # --------------------------------

    #
    # Too few arguments
    #

    with pytest.raises(Exception) as cm:
        feature_overview.incr_int()

    assert cm.value.args == (
        "TypeError: incr_int() missing 1 required positional argument",
    )

    #
    # Too many arguments
    #

    with pytest.raises(Exception) as cm:
        feature_overview.incr_int(1, 2, 3)

    assert cm.value.args == (
        ("TypeError: incr_int() takes 1 positional argument but 3 were given"),
    )

    #
    # Wrong type of argument
    #

    with pytest.raises(Exception) as cm:
        feature_overview.incr_int("string")

    assert cm.value.args == (
        (
            "TypeError: incr_int() expected Mojo 'SIMD[DType.int, 1]' type argument, got 'str'"
        ),
    )


def test_case_mojo_value_convert_from_python() -> None:
    mojo_int = feature_overview.Int()
    assert repr(mojo_int) == "Int(0)"

    feature_overview.add_to_int(mojo_int, 5)
    assert repr(mojo_int) == "Int(5)"

    feature_overview.add_to_int(mojo_int, 3)
    assert repr(mojo_int) == "Int(8)"

    #
    # Wrong type of argument
    #

    with pytest.raises(Exception) as cm:
        feature_overview.add_to_int(mojo_int, "foo")

    assert cm.value.args == (
        (
            "TypeError: add_to_int() expected argument at position 1 to be "
            "instance of (or convertible to) Mojo 'SIMD[DType.int, 1]'; got 'str'. "
            "(Note: attempted conversion failed due to: invalid literal for "
            "int() with base 10: 'foo')"
        ),
    )
