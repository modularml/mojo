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

# RUN: kgen %s -elaborate=no-use-parametric-interpreter -verify-diagnostics
# RUN: not kgen -elaborate  -D TEST_RECURSION2=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-RECURSION2
# RUN: not kgen -elaborate  -D TEST_RECURSION3=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-RECURSION3

from std.collections.string.string_span import StaticString, _get_kgen_string
from std.sys import get_defined_bool


# expected-error @+2{{function instantiation failed}}
@export
def entry_method() abi("Mojo"):
    foo()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
def foo():
    bar()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
def bar():
    baz()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
def baz():
    __mlir_op.`kgen.param.assert`[
        cond=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`,
        message="oops".value,
    ]()  # expected-note {{constraint failed}}


# expected-error @+2{{function instantiation failed}}
@export
def test_no_params() abi("Mojo"):
    no_parameters()  # expected-note {{call expansion failed}}


# expected-note @+2{{function instantiation failed}}
@no_inline
def no_parameters():
    parametric[1]()  # expected-note {{call expansion failed}}


@no_inline
def parametric[param: Int]():  # expected-note {{function instantiation failed}}
    comptime assert (  # expected-note {{constraint failed: param must be 2}}
        param == 2
    ), "param must be 2"


# This is copied so the note ends up in this file.
@always_inline("nodebug")
def constrained[cond: Bool, msg: StaticString]():
    comptime msg_literal = _get_kgen_string[msg]()
    __mlir_op.`kgen.param.assert`[
        cond=cond.__mlir_bool__(), message=msg_literal
    ]()


# expected-error @+2{{function instantiation failed}}
@export
def test_comptime_assert() abi("Mojo"):
    parametric_assert[1]()  # expected-note {{call expansion failed}}


# expected-note @+2{{function instantiation failed}}
@no_inline
def parametric_assert[param: Int]():
    # expected-note @below {{constraint failed: param must be 2}}
    comptime assert param == 2, "param must be 2"


# This creates recursive cycles: foo[D] -> bar[D] -> foo[D] and foo[D] -> baz[D] -> foo[D]
def bar[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


def baz[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


def foo[D: Int]() -> Int:
    var x = bar[D]()
    # CHECK-RECURSION2: call expansion failed with parameter value(s): ("D": 2)
    var y = baz[D]()
    _ = x
    _ = y
    return y


def test_recursion2():
    comptime run_test = get_defined_bool["TEST_RECURSION2", False]()

    comptime if run_test:
        # CHECK-RECURSION2: call expansion failed with parameter value(s): ("D": 2)
        _ = foo[2]()

        # CHECK-RECURSION2: function instantiation in parameter domain that recursively requires itself
        # CHECK-RECURSION2: recursively instantiated through here


# This creates a recursive cycle: foo1[D] -> bar1[D] -> foo1[D]
def bar1[D: Int]() -> Int:
    comptime x = foo1[D]()
    return x


def foo1[D: Int]() -> Int:
    # CHECK-RECURSION3: call expansion failed with parameter value(s): ("D": 1)
    var x = bar1[D]()
    return x


def test_recursion3():
    comptime run_test = get_defined_bool["TEST_RECURSION3", False]()

    comptime if run_test:
        # CHECK-RECURSION3: call expansion failed with parameter value(s): ("D": 1)
        _ = foo1[1]()

        # CHECK-RECURSION3: function instantiation in parameter domain that recursively requires itself
        # CHECK-RECURSION3: recursively instantiated through here


def main():
    test_recursion2()
    test_recursion3()
