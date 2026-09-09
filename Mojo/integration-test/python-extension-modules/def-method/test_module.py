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

import sys

# Imports from 'mojo_module.so'
import mojo_module as def_method  # type: ignore[import-not-found]
import pytest


def test_get_name() -> None:
    person = def_method.Person()
    assert person.get_name() == "John Smith"

    setattr(sys.modules[__name__], "deny_name", True)  # noqa: B010
    try:
        with pytest.raises(Exception) as cm:
            person.get_name()
        assert cm.value.args == ("name cannot be accessed",)
    finally:
        delattr(sys.modules[__name__], "deny_name")


def test_split_name() -> None:
    person = def_method.Person()
    assert person.split_name(" ") == ["John", "Smith"]


def test_with() -> None:
    person = def_method.Person()
    same_person = person._with("Jane Doe", 25)
    assert person.get_name() == "Jane Doe"
    assert person.get_age() == 25
    assert same_person.get_name() == person.get_name()
    assert same_person.get_age() == person.get_age()


def test_get_age() -> None:
    assert def_method.Person().get_age() == 123


def test_get_birth_year() -> None:
    assert def_method.Person()._get_birth_year(2025) == 1902


def test_with_first_last_name() -> None:
    person = def_method.Person()
    assert person._with_first_last_name("Jane", "Doe") == person
    assert person.get_name() == "Jane Doe"


def test_erase_name() -> None:
    person = def_method.Person()

    person.erase_name()
    assert person.get_name() == ""

    with pytest.raises(Exception) as cm:
        person.erase_name()
    assert cm.value.args == ("cannot erase name if it's already empty",)


def test_set_age() -> None:
    person = def_method.Person()
    person.set_age(25)
    assert person.get_age() == 25

    with pytest.raises(Exception) as cm:
        person.set_age("10.5")
    assert cm.value.args == ("cannot set age to 10.5",)


def test_set_name_and_age() -> None:
    person = def_method.Person()
    person.set_name_and_age("John Doe", 25)
    assert person.get_name() == "John Doe"
    assert person.get_age() == 25

    with pytest.raises(Exception) as cm:
        person.set_name_and_age("John Modular", "10.5")
    assert cm.value.args == ("cannot set age to 10.5",)


def test_reset() -> None:
    person = def_method.Person()
    person.set_name_and_age("John Doe", 25)

    person.reset()
    assert person.get_name() == "John Smith"
    assert person.get_age() == 123


def test_set_name() -> None:
    person = def_method.Person()
    person.set_name("John Doe")
    assert person.get_name() == "John Doe"


def test_set_age_from_dates() -> None:
    person = def_method.Person()
    person._set_age_from_dates(1991, 2025)
    assert person.get_age() == 34


def test_set_name_auto() -> None:
    person = def_method.Person()
    person.set_name_auto("Alice Auto")
    assert person.get_name_auto() == "Alice Auto"


def test_get_name_auto() -> None:
    person = def_method.Person()
    assert person.get_name_auto() == "John Smith"


def test_increment_age_auto() -> None:
    person = def_method.Person()
    new_age = person.increment_age_auto(7)
    assert new_age == 130
    assert person.get_age() == 130


def test_reset_auto() -> None:
    person = def_method.Person()
    person.set_name_auto("Modified Name")
    person.increment_age_auto(50)

    person.reset_auto()
    assert person.get_name_auto() == "Auto Reset Person"
    assert person.get_age() == 999


def test_sum_kwargs_ints_py() -> None:
    person = def_method.Person()
    initial_age = person.get_age()

    result = person.sum_kwargs_ints_py()
    assert result == initial_age

    result = person.sum_kwargs_ints_py(a=5, b=10, c=15)
    expected_age = initial_age + 30
    assert result == expected_age
    assert person.get_age() == expected_age


def test_sum_kwargs_ints() -> None:
    person = def_method.Person()
    initial_age = person.get_age()

    result = person.sum_kwargs_ints()
    assert result == initial_age

    result = person.sum_kwargs_ints(a=5, b=10, c=15)
    expected_age = initial_age + 30
    assert result == expected_age
    assert person.get_age() == expected_age


def test_add_kwargs_to_age_auto() -> None:
    person = def_method.Person()
    initial_age = person.get_age()

    result = person.add_kwargs_to_age_auto()
    assert result == initial_age

    result = person.add_kwargs_to_age_auto(bonus=10, extra=5)
    expected_age = initial_age + 15
    assert result == expected_age
    assert person.get_age() == expected_age


def test_set_age_from_kwargs_auto() -> None:
    person = def_method.Person()
    assert person.set_age_from_kwargs_auto(20, bonus=10, extra=5) is None
    assert person.get_age() == 35
