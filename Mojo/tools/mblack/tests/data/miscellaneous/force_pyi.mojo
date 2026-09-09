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
#   Path:   tests/data/miscellaneous/force_pyi.py
#
# ===----------------------------------------------------------------------=== #

from typing import Union


@bird
def zoo():
    ...


class A:
    ...


@bar
class B:
    def BMethod(self) -> None:
        ...

    @overload
    def BMethod(self, arg: List[str]) -> None:
        ...


class C:
    ...


@hmm
class D:
    ...


class E:
    ...


@baz
def foo() -> None:
    ...


class F(A, C):
    ...


def spam() -> None:
    ...


@overload
def spam(arg: str) -> str:
    ...


val: int = 1


def eggs() -> Union[str, int]:
    ...


# output

from typing import Union

@bird
def zoo(): ...

class A: ...

@bar
class B:
    def BMethod(self) -> None: ...
    @overload
    def BMethod(self, arg: List[str]) -> None: ...

class C: ...

@hmm
class D: ...

class E: ...

@baz
def foo() -> None: ...

class F(A, C): ...

def spam() -> None: ...
@overload
def spam(arg: str) -> str: ...

val: int = 1

def eggs() -> Union[str, int]: ...
