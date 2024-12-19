# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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


class Person(ABC):
    pass


class Foo:
    def __init__(self, bar):
        self.bar = bar


class AbstractPerson(ABC):
    @abstractmethod
    def method(self):
        ...


def my_function(name):
    return f"Formatting the string from Lit with Python: {name}"


def eat_it_all(veggie, *args, fruit, **kwargs):
    return f"{veggie} {args} fruit={fruit} {kwargs}"
