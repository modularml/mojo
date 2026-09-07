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

from abc import ABC, abstractmethod


class Person(ABC):  # noqa: B024
    pass


class Foo:
    def __init__(self, bar) -> None:  # noqa: ANN001
        self.bar = bar


class AbstractPerson(ABC):
    @abstractmethod
    def method(self): ...  # noqa: ANN201


def my_function(name) -> str:  # noqa: ANN001
    return f"Formatting the string from Lit with Python: {name}"


def eat_it_all(veggie, *args, fruit, **kwargs) -> str:  # noqa: ANN001
    return f"{veggie} {args} fruit={fruit} {kwargs}"
