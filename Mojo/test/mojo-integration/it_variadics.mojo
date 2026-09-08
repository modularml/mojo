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

# RUN: %mojo %s | FileCheck %s


# This isn't copyable or movable, but it is talkative!
struct TalkativeMem(Writable):
    var state: Int

    @implicit
    def __init__(out self, state: Int):
        self.state = state
        print("initializing", state)

    def __deinit__(deinit self):
        print("destroying", self.state)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("talkative ", self.state)


# This isn't copyable or movable, but it is talkative!
struct TalkativeReg(RegisterPassable, Writable):
    var state: Int

    @implicit
    def __init__(out self, state: Int):
        self.state = state
        print("initializing", state)

    def __deinit__(deinit self):
        print("destroying", self.state)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("talkative ", self.state)


# This is copyable, movable, and talkative!  It doesn't print on move.
struct TalkativeCopableReg(ImplicitlyCopyable, RegisterPassable, Writable):
    var state: Int

    @implicit
    def __init__(out self, state: Int):
        self.state = state
        print("initializing", state)

    def __init__(out self, *, copy: Self):
        self.state = copy.state
        print("copying", self.state)

    def __deinit__(deinit self):
        print("destroying", self.state)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("talkative ", self.state)


# This is copyable, movable, and talkative!  It prints on move.
struct TalkativeCopableMovableMem(ImplicitlyCopyable, Writable):
    var state: Int

    @implicit
    def __init__(out self, state: Int):
        self.state = state
        print("initializing", state)

    def __init__(out self, *, copy: Self):
        self.state = copy.state
        print("copying", self.state)

    def __init__(out self, *, deinit move: Self):
        self.state = move.state
        print("moving", self.state)

    def __deinit__(deinit self):
        print("destroying", self.state)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("talkative ", self.state)


trait Bumpable:
    """Test helper: in-place increment for mut variadic-pack mutation tests."""

    def bump(mut self):
        ...


@fieldwise_init
struct CountBox(Bumpable, ImplicitlyCopyable):
    var n: Int

    def bump(mut self):
        self.n += 7


# ===----------------------------------------------------------------------=== #
# Parameter varargs
# ===----------------------------------------------------------------------=== #


def takes_int_params[*args: Int]():
    # CHECK-LABEL: -- Testing parameter varargs
    print("-- Testing parameter varargs")

    # can dynamically index the parameter list.
    var total = 0
    for i in range(len(args)):
        total += args[i]
    print("index:", total)
    # CHECK-NEXT: index: 15

    # can iterate the parameter list.
    total = 0
    for i in args:
        total += i
    print("iterate: ", total)
    # CHECK-NEXT: iterate: 15

    # can also get statically-indexed elements as comptime values.
    comptime elt = args[3]
    print("comptime elt3: ", elt)
    # CHECK-NEXT: comptime elt3: 4


def test_param_varargs():
    takes_int_params[1, 2, 3, 4, 5]()


# ===----------------------------------------------------------------------=== #
# mut varargs
# ===----------------------------------------------------------------------=== #


def test_mut_varargs():
    # CHECK-LABEL: -- Testing mut varargs
    print("-- Testing mut varargs")
    var s1 = "hello"
    var s2 = "konnichiwa"
    var s3 = "bonjour"

    def make_worldly(mut *strs: String):
        for i in range(len(strs)):
            strs[i] += " world"

    make_worldly(s1, s2, s3)
    print(s1)  # CHECK-NEXT: hello world
    print(s2)  # CHECK-NEXT: konnichiwa world
    print(s3)  # CHECK-NEXT: bonjour world

    # CHECK: -- Testing mut varargs destructors
    print("-- Testing mut varargs destructors")
    var v1 = TalkativeMem(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeMem(2)  # CHECK-NEXT: initializing 2

    def double(mut *tm: TalkativeMem):
        for i in range(len(tm)):
            tm[i].state *= 2

    double(v1, v2)
    # CHECK-NEXT: destroying 4
    # CHECK-NEXT: destroying 2

    # CHECK-NEXT: test forwarding
    print("test forwarding")

    var s4 = "first"
    var s5 = "second"

    def take_forwarded(mut *strs: String):
        for ref s in strs:
            s += " item"

    def pass_forwarded(mut *strs: String):
        take_forwarded(*strs)

    pass_forwarded(s4, s5)
    print(s4)  # CHECK-NEXT: first item
    print(s5)  # CHECK-NEXT: second item


# ===----------------------------------------------------------------------=== #
# Owned varargs
# ===----------------------------------------------------------------------=== #


def test_owned_varargs():
    # CHECK-LABEL: -- testing owned mem varargs
    print("\n-- testing owned mem varargs")

    var v1 = TalkativeMem(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeMem(2)  # CHECK-NEXT: initializing 2
    var v3 = TalkativeMem(3)  # CHECK-NEXT: initializing 3

    # CHECK-NEXT: first test
    print("first test")

    def handle_owned_mem(var *strs: TalkativeMem):
        # owned arguments are mutable and live as long as they are used.
        for i in range(len(strs)):
            strs[i].state *= 2

        # So they should die here, after the loop, before the print.
        # CHECK-NEXT: destroying 6
        # CHECK-NEXT: destroying 4
        # CHECK-NEXT: destroying 2

        # CHECK-NEXT: after last use
        print("after last use")

    handle_owned_mem(v1^, v2^, v3^)

    # CHECK-NEXT: test forwarding
    print("test forwarding")

    v1 = TalkativeMem(5)  # CHECK-NEXT: initializing 5
    v2 = TalkativeMem(10)  # CHECK-NEXT: initializing 10

    def take_forwarded(var *strs: TalkativeMem):
        for ref s in strs:
            s.state *= 2

        # So they should die here, after the loop, before the print.
        # CHECK-NEXT: destroying 20
        # CHECK-NEXT: destroying 10

        # CHECK-NEXT: after last use
        print("after last use")

    def pass_forwarded(var *strs: TalkativeMem):
        take_forwarded(*strs^)

    pass_forwarded(v1^, v2^)

    # CHECK-NEXT: after call
    print("after call")


def test_owned_reg_varargs():
    # CHECK-LABEL: -- testing owned reg varargs
    print("\n-- testing owned reg varargs")

    var v1 = TalkativeReg(1)  # CHECK-NEXT: initializing 1
    var v2 = TalkativeReg(2)  # CHECK-NEXT: initializing 2
    var v3 = TalkativeReg(3)  # CHECK-NEXT: initializing 3

    def handle_owned_reg(var *strs: TalkativeReg):
        # owned arguments are mutable and live as long as they are used.
        for ref s in strs:
            s.state *= 2

        # So they should die here, after the loop, before the print.
        # CHECK-NEXT: destroying 6
        # CHECK-NEXT: destroying 4
        # CHECK-NEXT: destroying 2

        # CHECK-NEXT: after last use
        print("after last use")

    handle_owned_reg(v1^, v2^, v3^)

    # CHECK-NEXT: after call
    print("after call")


def test_non_trivial_reg_varargs():
    # CHECK-LABEL: -- test_non_trivial_reg_varargs
    print("\n-- test_non_trivial_reg_varargs")

    def callee(*args: TalkativeCopableReg):
        var arg: TalkativeCopableReg

        if len(args) == 1:
            arg = args[0]
        else:
            arg = args[1]
        print("here")
        print(arg)

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: copying 1
    # CHECK-NEXT: here
    # CHECK-NEXT: talkative 1
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: destroying 1
    callee(TalkativeCopableReg(1))

    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: initializing 3
    # CHECK-NEXT: copying 3
    # CHECK-NEXT: here
    # CHECK-NEXT: talkative 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 2
    callee(TalkativeCopableReg(2), TalkativeCopableReg(3))


# ===----------------------------------------------------------------------=== #
# owned variadic packs
# ===----------------------------------------------------------------------=== #


def owned_variadic_pack[*Ts: Writable](var *pack: *Ts):
    print("-- testing owned variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    comptime for i in range(pack.__len__()):
        print("hello", pack[i])

    print("owned_variadic_pack done")


def test_owned_variadic_pack():
    # CHECK-LABEL: -- testing owned variadic pack with 0 elements
    owned_variadic_pack()
    # CHECK-NEXT: owned_variadic_pack done
    # CHECK-NEXT: done zero
    print("done zero")

    # CHECK-NEXT: -- testing owned variadic pack with 2 elements
    # CHECK-NEXT: hello foo
    # CHECK-NEXT: hello 42
    # CHECK-NEXT: owned_variadic_pack done
    owned_variadic_pack("foo", 42)
    # CHECK-NEXT: done two
    print("done two")

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: initializing 3
    # CHECK-NEXT: -- testing owned variadic pack with 3 elements
    # CHECK-NEXT: hello talkative 1
    # CHECK-NEXT: hello talkative 2
    # CHECK-NEXT: hello talkative 3
    # CHECK-NEXT: destroying 3
    # CHECK-NEXT: destroying 2
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: owned_variadic_pack done
    owned_variadic_pack(TalkativeMem(1), TalkativeMem(2), TalkativeMem(3))

    # CHECK-NEXT: done three
    print("done three")


def inout_variadic_pack[*Ts: Writable](mut *pack: *Ts):
    print("-- testing mut variadic pack with", len(pack), "elements")

    # TODO: Test mutation of the value not just reading of the value.
    comptime for i in range(pack.__len__()):
        print("hello", pack[i])


def test_inout_variadic_pack():
    # CHECK-LABEL: -- testing mut variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    var string = "foo"
    var integer = 42
    inout_variadic_pack(string, integer)
    print("")

    # CHECK: initializing 1
    # CHECK: initializing 2
    # CHECK: -- testing mut variadic pack with 2 elements
    # CHECK: hello talkative 1
    # CHECK: hello talkative 2
    # CHECK: destroying 1
    # CHECK: after call
    # CHECK: destroying 2
    var m1 = TalkativeMem(1)
    var m2 = TalkativeMem(2)
    inout_variadic_pack(m1, m2)
    print("after call")
    _ = m2^  # keep alive for the test
    print("")


def bump_each_box[*Ts: Bumpable](mut *pack: *Ts):
    comptime for i in range(pack.__len__()):
        pack[i].bump()


def forward_bump_each_box[*Ts: Bumpable](mut *pack: *Ts):
    bump_each_box(*pack)


def test_forward_mut_pack():
    # Forwarding `mut *pack` must preserve aliases so mutations hit caller storage.
    # CHECK-LABEL: test_forward_mut_pack
    print("test_forward_mut_pack")
    var c0 = CountBox(1)
    var c1 = CountBox(2)
    forward_bump_each_box(c0, c1)
    print(c0.n)  # CHECK-NEXT: 8
    print(c1.n)  # CHECK-NEXT: 9


def borrowed_variadic_pack[*Ts: Writable](*pack: *Ts):
    print("-- testing read-only variadic pack with", len(pack), "elements")

    comptime for i in range(pack.__len__()):
        print("hello", pack[i])


def test_borrowed_variadic_pack():
    # CHECK-LABEL: -- testing read-only variadic pack with 2 elements
    # CHECK: hello foo
    # CHECK: hello 42
    borrowed_variadic_pack("foo", 42)
    print("")

    # CHECK: initializing 2
    # CHECK: initializing 1
    # CHECK: -- testing read-only variadic pack with 2 elements
    # CHECK: hello talkative 1
    # CHECK: hello talkative 2
    # CHECK: destroying 1
    # CHECK: after call
    # CHECK: destroying 2
    var m2 = TalkativeMem(2)
    borrowed_variadic_pack(TalkativeMem(1), m2)
    print("after call")
    _ = m2^  # Keep alive to for the test
    print("")


# Sum all the specified values together.
def sum_intable[*Ts: Intable](*pack: *Ts) -> Int:
    var result = 0

    comptime for i in range(pack.__len__()):
        result += Int(pack[i])

    return result


def forward_sum_intable[*Ts: Intable](*pack: *Ts) -> Int:
    return sum_intable(*pack)


def sum_intable_with_bias[*Ts: Intable](*pack: *Ts, bias: Int) -> Int:
    return sum_intable(*pack) + bias


def forward_sum_intable_with_bias[*Ts: Intable](*pack: *Ts, bias: Int) -> Int:
    return sum_intable_with_bias(*pack, bias=bias)


# Concatenate all pack elements as `Writable` (same forwarding shape as the
# `Intable` helpers above).
def concat_writable[*Ts: Writable](*pack: *Ts) -> String:
    return String(*pack)


def forward_concat_writable[*Ts: Writable](*pack: *Ts) -> String:
    return concat_writable(*pack)


def concat_writable_with_suffix[
    *Ts: Writable
](*pack: *Ts, suffix: String) -> String:
    return concat_writable(*pack) + suffix


def forward_concat_writable_with_suffix[
    *Ts: Writable
](*pack: *Ts, suffix: String) -> String:
    return concat_writable_with_suffix(*pack, suffix=suffix)


# Check to see if we can do packs at comptime.
def test_comptime_pack():
    # CHECK-LABEL: test_comptime_pack
    print("test_comptime_pack")

    var str1 = sum_intable(4, 5.0, 7)
    print(str1)  # CHECK: 16

    comptime str2 = sum_intable(4, 5.0, 7)
    print(str2)
    # CHECK: 16


def test_forward_comptime_pack():
    # CHECK-LABEL: test_forward_comptime_pack
    print("test_forward_comptime_pack")

    print("forwarded empty", forward_sum_intable())
    # CHECK: forwarded empty 0

    print("forwarded", forward_sum_intable(4, 5.0, 7))
    # CHECK: forwarded 16

    print(
        "forwarded with bias", forward_sum_intable_with_bias(4, 5.0, 7, bias=2)
    )
    # CHECK: forwarded with bias 18


def test_forward_comptime_pack_writable():
    # CHECK-LABEL: test_forward_comptime_pack_writable
    print("test_forward_comptime_pack_writable")

    print("forwarded empty", "[" + forward_concat_writable() + "]")
    # CHECK: forwarded empty []

    var ab: String = "ab"
    print(
        "forwarded",
        forward_concat_writable(ab, "cd", 42),
    )
    # CHECK: forwarded abcd42

    print(
        "forwarded with suffix",
        forward_concat_writable_with_suffix(ab, "cd", 42, suffix=String("!")),
    )
    # CHECK: forwarded with suffix abcd42!


def use_value[T: AnyType](value: T):
    pass


def test_tuple():
    # CHECK-LABEL: -- test_tuple
    print("-- test_tuple")

    # TODO: The initializer for tuple is copying+destroying the elements
    # unnecessarily.

    # CHECK-NEXT: initializing 1
    # CHECK-NEXT: initializing 2
    # CHECK-NEXT: moving 1
    # CHECK-NEXT: moving 2
    var t1 = TalkativeCopableMovableMem(1), TalkativeCopableMovableMem(2)

    # CHECK-NEXT: p1: talkative 2 before copy
    print("p1:", t1[1], "before copy")

    # CHECK-NEXT: copying 1
    # CHECK-NEXT: copying 2
    var t2 = t1

    # CHECK-NEXT: p2: talkative 1 before transfer
    print("p2:", t2[0], "before transfer")

    # CHECK-NEXT: moving 1
    # CHECK-NEXT: moving 2
    var t3 = t1^

    # CHECK-NEXT: before use t2
    print("before use t2")

    use_value(t2)
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: destroying 2

    # CHECK-NEXT: before use t3
    print("before use t3")
    use_value(t3)
    # CHECK-NEXT: destroying 1
    # CHECK-NEXT: destroying 2

    # CHECK-NEXT: test_tuple done!
    print("test_tuple done!")


def takes_variadic_params[
    T: Copyable, //, *values: T
]() -> Span[T, ImmStaticOrigin]:
    return values.get_span()


def test_comptime_variadics():
    # CHECK-LABEL: test_comptime_variadics
    print("test_comptime_variadics")

    for elt in takes_variadic_params[1, 2, 3]():
        print(elt)
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3

    for elt in takes_variadic_params["foo" + String(4), "bar", "baz"]():
        print(elt)
    # CHECK-NEXT: foo4
    # CHECK-NEXT: bar
    # CHECK-NEXT: baz


def print_kwargs(var **kwargs: Int):
    for item in kwargs.items():
        print(item.key, item.value)


def test_kwargs():
    print("test_kwargs")
    # CHECK-LABEL: test_kwargs

    print_kwargs(x=1, y=2, z=3)
    # CHECK-NEXT: x 1
    # CHECK-NEXT: y 2
    # CHECK-NEXT: z 3

    def forward_kwargs(var **kwargs: Int):
        print_kwargs(**kwargs^)

    forward_kwargs(aa=4, y=5, z=6)
    # CHECK-NEXT: aa 4
    # CHECK-NEXT: y 5
    # CHECK-NEXT: z 6


def main():
    test_param_varargs()
    test_mut_varargs()
    test_owned_varargs()
    test_owned_reg_varargs()
    test_non_trivial_reg_varargs()
    test_owned_variadic_pack()
    test_inout_variadic_pack()
    test_forward_mut_pack()
    test_borrowed_variadic_pack()
    test_comptime_pack()
    test_forward_comptime_pack()
    test_forward_comptime_pack_writable()
    test_tuple()
    test_comptime_variadics()
    test_kwargs()
