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

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-warning @+1 {{@export requires an explicit 'abi()' effect on the function}}
@export("noabi")
def no_abi():
    ...


# expected-error @+2 {{'abi("C")' function may not be marked 'raises'; remove 'raises' or use 'abi("Mojo")'}}
@export("c_raises")
def c_raises() abi("C") raises:
    ...


# expected-error @+1 {{@export requires a string specifying the name of the exported symbol}}
@export(1)
def export_me() abi("C"):
    ...


# expected-note @+1 {{previous export here}}
@export("my_foo")
def foo() abi("C"):
    ...


# expected-error @+1 {{invalid re-export of my_foo}}
@export("my_foo")
def bar() abi("C"):
    ...


# expected-warning @+2 {{ABI="C" is deprecated; use abi("C") instead}}
# expected-error @+1 {{my+foo is not a valid C identifier}}
@export("my+foo", ABI="C")
def bad_name():
    ...

# expected-warning @+2 {{ABI="C" is deprecated; use abi("C") instead}}
# expected-error @+1 {{'abi("C")' function may not be marked 'raises'; remove 'raises' or use 'abi("Mojo")'}}
@export("c_raises", ABI="C")
def c_raises_old_abi() raises:
    ...

# expected-note @+1 {{previous export here}}
@export
def func_overloaded(x: Int) abi("C"):
    ...


# expected-error @+1 {{invalid re-export of func_overloaded}}
@export
def func_overloaded(x: Bool) abi("C"):
    ...
