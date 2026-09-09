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

# Tests for the error on an implicitly declared variable, and for the binding
# forms that must stay silent because they already spell out how they bind.
# Related to MOCO-3182.

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null


struct ExampleCM(ImplicitlyCopyable):
    def __enter__(self) -> Int:
        return 1

    def __exit__(self):
        pass


def one() -> Int:
    return 1


def truthy() -> Bool:
    return True


def use(x: Int):
    pass


def use_bool(x: Bool):
    pass


def simple():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x = 1
    use(x)


def annotated():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x: Int = 5
    use(x)


# An annotation with no initializer declares just as much as an assignment does.
def annotated_no_initializer():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x: Int
    x = 5
    use(x)


# Only the first assignment is diagnosed; the name is then registered, so later
# assignments and uses resolve against it.
def reassigned():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x = 1
    x = 2
    use(x)


# A single 'var' covers a whole tuple target. Emission stops after the first
# unresolved element, so only that name is diagnosed and only it is registered;
# later elements stay unknown.
def tuple_target():
    # expected-error @+1 {{implicit declaration of 'a' is not allowed; add 'var' to declare a new name}}
    a, b = Tuple(1, 2)
    use(a)
    # expected-error @+1 {{use of unknown declaration 'b'}}
    use(b)


# Only the fresh element is diagnosed when the target mixes declared and fresh names.
def tuple_mixed():
    var a = 0
    # expected-error @+1 {{implicit declaration of 'b' is not allowed; add 'var' to declare a new name}}
    a, b = Tuple(1, 2)
    use(a)
    use(b)

# With no binder anywhere in the target, emission still stops at the first
# unresolved name.
def nested_tuple_target_no_binder():
    # expected-error @+1 {{implicit declaration of 'a' is not allowed; add 'var' to declare a new name}}
    (a, b), c = Tuple(Tuple(1, 2), 3)
    use(a)
    # expected-error @+1 {{use of unknown declaration 'b'}}
    use(b)
    # expected-error @+1 {{use of unknown declaration 'c'}}
    use(c)


def nested_tuple_target_no_binder_migrated():
    var (a, b), c = Tuple(Tuple(1, 2), 3)
    use(a)
    use(b)
    use(c)


# TODO(MOCO-4520): the migrated form of the case above belongs here, spelling
# the two names as `var a: Int` / `var b: Int`. It is omitted because an inner
# tuple that fully resolves, beside a sibling that does not, strands an owning
# `RCRef<TupleDLValue>` in the parser's persistent arena, and LeakSanitizer
# reports it. The unmigrated form above is unaffected: its names are fresh, so
# the inner tuple never resolves and no `TupleDLValue` is formed.


# TODO(MOCO-4520): the same spelling is missing for the tuple chain below;
# omitted for the same arena leak.


# Each target of a chain declares, and 'var' on each is a valid spelling.
def chained():
    # expected-error @+2 {{implicit declaration of 'a' is not allowed; add 'var' to declare a new name}}
    # expected-error @+1 {{implicit declaration of 'b' is not allowed; add 'var' to declare a new name}}
    a = b = 1
    use(a)
    use(b)


def chained_migrated():
    var a = var b = 1
    use(a)
    use(b)


# A chain of tuple targets diagnoses the first unresolved name in each target.
def chain_tuple():
    # expected-error @+2 {{implicit declaration of 'a' is not allowed; add 'var' to declare a new name}}
    # expected-error @+1 {{implicit declaration of 'c' is not allowed; add 'var' to declare a new name}}
    a, b = c, d = Tuple(1, 2)
    use(a)
    # expected-error @+1 {{use of unknown declaration 'b'}}
    use(b)
    use(c)
    # expected-error @+1 {{use of unknown declaration 'd'}}
    use(d)


def chain_tuple_migrated():
    var a, b = var c, d = Tuple(1, 2)
    use(a)
    use(b)
    use(c)
    use(d)


# A walrus target must already be an LValue, so it never reaches the
# implicit-declaration path at all.
def walrus_condition():
    # expected-error @+1 {{use of unknown declaration 'x'}}
    if x := truthy():
        use_bool(x)


# Hoisting the declaration leaves the walrus assigning it rather than
# introducing one, so nothing is reported.
def walrus_condition_hoisted():
    var x: Bool
    if x := truthy():
        use_bool(x)
    use_bool(x)


def walrus_in_call_hoisted():
    var x: Int
    use(x := 1)


# One 'var' covers a whole tuple target, but a walrus target takes none, so each
# element hoists instead of naming the target.
def walrus_tuple_target_hoisted():
    var a: Int
    var b: Int
    use(((a, b) := Tuple(1, 2))[0])
    use(a)
    use(b)


# Each target of a mixed chain answers for itself, so the plain one keeps the
# name message and its fixit. The other order is not a shape to pin:
# `a := b = 1` does not parse.
def walrus_in_chain():
    var c: Int
    var d: Int
    d = c := 5
    use(c)
    use(d)


# ===----------------------------------------------------------------------=== #
# Sites inside a nested block still reject implicit declarations the same way.
# ===----------------------------------------------------------------------=== #


def nested_if(c: Bool):
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    if c:
        x = 1
    use(1)


def nested_loop(items: List[Int]):
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    for i in items:
        x = i
    use(1)


def nested_with(cm: ExampleCM):
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    with cm as v:
        x = v
    use(1)


# The first arm is diagnosed and registers the name, so the second arm and the
# use after the statement resolve against it.
def nested_both_arms(c: Bool):
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    if c:
        x = 1
    else:
        x = 2
    use(x)


# Explicit declaration before the nested statement.
def nested_hoisted(c: Bool):
    var x: Int
    if c:
        x = 1
    else:
        x = 2
    use(x)


# Nested tuple targets get the same message; emission stops at the first
# unresolved name.
def nested_tuple_target(c: Bool):
    # expected-error @+2 {{implicit declaration of 'a' is not allowed; add 'var' to declare a new name}}
    if c:
        a, b = Tuple(1, 2)
    use(1)


def nested_while(c: Bool):
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    while c:
        x = 1
    use(1)

def nested_comptime_if():
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    comptime if True:
        x = 1
    use(1)


def nested_for_else(items: List[Int]):
    for i in items:
        use(i)
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    else:
        x = 1
    use(1)


# A nested 'def' has its own function body; implicit decls there are rejected
# the same way.
def nested_def_body():
    def inner():
        # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
        x = 1
        use(x)

    inner()


# Registration is on the enclosing function, so a nested 'def' resolves the name
# through parent scope rather than reporting it again.
def nested_def_reads_outer():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x = 1

    def inner():
        use(x)

    inner()


# A capture list resolves through a lookup of its own, so it is pinned too.
def closure_reads_outer():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x = 1

    def cl(z: Int) {imm x} -> Int:
        return x + z

    use(cl(1))


# A lambda's capture list reaches that lookup by a different path.
def lambda_capture_reads_outer():
    # expected-error @+1 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    x = 1

    use((lambda (z: Int) {imm x} -> Int: x + z)(1))


# Registration is per name, so a later 'var' of that name is a redefinition and
# is reported as one.
def redeclared_with_var():
    # expected-error @+2 {{implicit declaration of 'x' is not allowed; add 'var' to declare a new name}}
    # expected-note @+1 {{previous definition here}}
    x = 1
    # expected-error @+1 {{invalid redefinition of 'x'}}
    var x = 2
    use(x)


# ===----------------------------------------------------------------------=== #
# A short-circuit operand, a conditional-expression arm and a comprehension body
# are reachable only by a walrus, and each also gets a block of its own, so both
# rules point the same way there.
# ===----------------------------------------------------------------------=== #

def comprehension_walrus_hoisted(items: List[Int]):
    var x: Int
    var doubled = [(x := i) for i in items]
    use(doubled[0])


# ===----------------------------------------------------------------------=== #
# Forms that already spell out their binding, and must stay silent.
# ===----------------------------------------------------------------------=== #


def explicit_var():
    var x = 1
    use(x)


def for_target(items: List[Int]):
    for i in items:
        use(i)


def for_var_target(items: List[Int]):
    for var i in items:
        use(i)


def with_target(cm: ExampleCM):
    with cm as x:
        use(x)


def except_target() raises:
    try:
        use(one())
    except err:
        pass


def comprehension_target(items: List[Int]):
    var doubled = [i * 2 for i in items]
    use(doubled[0])


def discard():
    _ = one()
