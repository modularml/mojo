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
from collections.abc import Callable

# Imports from 'mojo_module.so'
import mojo_module as def_staticmethod  # type: ignore[import-not-found]
import pytest


def _test_takes_zero_returns(fut: Callable[[], str]) -> None:
    setattr(sys.modules[__name__], "s", "just a python string")  # noqa: B010
    result = fut()
    assert result == "just another python string"


def _test_takes_one_returns(fut: Callable[[str], str]) -> None:
    result = fut("foo")
    assert result == "foo"


def _test_takes_two_returns(fut: Callable[[str, str], str]) -> None:
    result = fut("foo", "bar")
    assert result == "foobar"


def _test_takes_three_returns(fut: Callable[[str, str, str], str]) -> None:
    result = fut("foo", "bar", "baz")
    assert result == "foobarbaz"


def test_takes_zero_returns() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_zero_returns(dummy.takes_zero_returns)


def test_takes_one_returns() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_one_returns(dummy.takes_one_returns)


def test_takes_two_returns() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_two_returns(dummy.takes_two_returns)


def test_takes_three_returns() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_three_returns(dummy.takes_three_returns)


def test_takes_zero_raises_returns() -> None:
    dummy = def_staticmethod.Dummy
    setattr(sys.modules[__name__], "s", "a special python string")  # noqa: B010
    with pytest.raises(Exception) as cm:
        dummy.takes_zero_raises_returns()
    assert cm.value.args == ("`s` must be 'just a python string'",)

    _test_takes_zero_returns(dummy.takes_zero_raises_returns)


def test_takes_one_raises_returns() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_one_raises_returns("quux")
    assert cm.value.args == ("input must be 'foo'",)

    _test_takes_one_returns(dummy.takes_one_raises_returns)


def test_takes_two_raises_returns() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_two_raises_returns("quux", "bar")
    assert cm.value.args == ("first input must be 'foo'",)

    _test_takes_two_returns(dummy.takes_two_raises_returns)


def test_takes_three_raises_returns() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_three_raises_returns("quux", "bar", "baz")
    assert cm.value.args == ("first input must be 'foo'",)

    _test_takes_three_returns(dummy.takes_three_raises_returns)


def _test_takes_zero(fut: Callable[[], None]) -> None:
    setattr(sys.modules[__name__], "s", "just a python string")  # noqa: B010
    fut()
    assert (
        getattr(sys.modules[__name__], "s")  # noqa: B009
        == "Hark! A mojo function calling into Python, called from Python!"
    )


def _test_takes_one(fut: Callable[[list], None]) -> None:  # type: ignore[type-arg]
    list_obj = [1, 2, 5]
    fut(list_obj)
    assert list_obj[0] == "baz"


def _test_takes_two(fut: Callable[[list, str], None]) -> None:  # type: ignore[type-arg]
    list_obj = [1, 2, 5]
    fut(list_obj, "foo")
    assert list_obj[0] == "foo"


def _test_takes_three(fut: Callable[[list, str, str], None]) -> None:  # type: ignore[type-arg]
    list_obj = [1, 2, 5]
    fut(list_obj, "foo", "bar")
    assert list_obj[0] == "foobar"


def test_takes_zero() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_zero(dummy.takes_zero)


def test_takes_one() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_one(dummy.takes_one)


def test_takes_two() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_two(dummy.takes_two)


def test_takes_three() -> None:
    dummy = def_staticmethod.Dummy
    _test_takes_three(dummy.takes_three)


def test_takes_zero_raises() -> None:
    dummy = def_staticmethod.Dummy
    setattr(sys.modules[__name__], "s", "a special python string")  # noqa: B010
    with pytest.raises(Exception) as cm:
        dummy.takes_zero_raises()
    assert cm.value.args == ("`s` must be 'just a python string'",)

    _test_takes_zero(dummy.takes_zero_raises)


def test_takes_one_raises() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_one_raises([1, 2])
    assert cm.value.args == ("list_obj must have length 3",)

    _test_takes_one(dummy.takes_one_raises)


def test_takes_two_raises() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_two_raises([1, 2], "foo")
    assert cm.value.args == ("list_obj must have length 3",)

    _test_takes_two(dummy.takes_two_raises)


def test_takes_three_raises() -> None:
    dummy = def_staticmethod.Dummy
    with pytest.raises(Exception) as cm:
        dummy.takes_three_raises([1, 2], "foo", "bar")
    assert cm.value.args == ("list_obj must have length 3",)

    _test_takes_three(dummy.takes_three_raises)


def test_kwargs_staticmethods() -> None:
    dummy = def_staticmethod.Dummy
    assert dummy.sum_pos_arg_and_kwargs(20, bonus=10, extra=5) == 35

    values: list[int] = []
    assert dummy.append_kwarg(values, value=42) is None
    assert values == [42]
    assert dummy.ignore_kwargs(a=1, b=2) is None
