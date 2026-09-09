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

# Tests for the fixit on the implicit-declaration error. Uses the JSON
# diagnostic format to verify the exact fixit position: 'var ' immediately
# before the undeclared name. Walrus targets no longer implicitly declare at
# all.
# Related to MOCO-3182.

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


def one() -> Int:
    return 1


def truthy() -> Bool:
    return True


def use(x: Int):
    pass


def simple():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'x' is not allowed; add 'var' to declare a new name"
    x = 1
    use(x)


# The insertion goes before the name, not before the type annotation.
def annotated():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'y' is not allowed; add 'var' to declare a new name"
    y: Int = 5
    use(y)


# A walrus target must already be an LValue; it does not implicitly declare.
def walrus_in_call():
    # CHECK: "message":"use of unknown declaration 'v'"
    use(v := one())
    use(v)


# Same for a walrus in a condition.
def walrus():
    # CHECK: "message":"use of unknown declaration 'z'"
    if z := truthy():
        use(1)


# Nested blocks get the same fixit as top-level assignments.
def nested_block(c: Bool):
    # CHECK: "fixIts":[{"end":{"column":9,"line":[[#@LINE+3]]},"start":{"column":9,"line":[[#@LINE+3]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'e' is not allowed; add 'var' to declare a new name"
    if c:
        e = 1
    use(1)


# A short-circuit walrus operand also requires a prior declaration.
def and_operand(c: Bool) -> Bool:
    # CHECK: "message":"use of unknown declaration 'w'"
    return c and (w := truthy())


# Both targets of a chain get their own fixit, and 'var a = var b = 1' compiles.
def chained():
    # CHECK: "fixIts":[{"end":{"column":9,"line":[[#@LINE+4]]},"start":{"column":9,"line":[[#@LINE+4]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'b' is not allowed; add 'var' to declare a new name"
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'a' is not allowed; add 'var' to declare a new name"
    a = b = 1
    use(a)
    use(b)


# Tuple targets get a fixit before the first unresolved name. Emission stops
# after that element.
def tuple_target():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'c' is not allowed; add 'var' to declare a new name"
    c, d = Tuple(1, 2)
    use(c)
    use(d)
