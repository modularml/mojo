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

# End-to-end tests for type refinement via where-clause constraints.
#
# RUN: %mojo -debug-level full %s | FileCheck %s

from std.builtin.rebind import downcast
import std.memory


trait Greetable:
    def greet(self) -> String:
        ...


trait Describable:
    def describe(self) -> String:
        ...


struct Dog(Copyable, Deinitable, Describable, Greetable, Movable):
    var name: String

    def __init__(out self, name: String):
        self.name = name

    def greet(self) -> String:
        return "Woof! I'm " + self.name

    def describe(self) -> String:
        return "Dog(" + self.name + ")"


struct Cat(Copyable, Deinitable, Greetable, Movable):
    var name: String

    def __init__(out self, name: String):
        self.name = name

    def greet(self) -> String:
        return "Meow! I'm " + self.name


trait StaticLabel:
    @staticmethod
    def static_label() -> String:
        ...


trait StaticOriginalLabel:
    @staticmethod
    def original_static_label() -> String:
        ...


trait StaticRefinedLabel:
    @staticmethod
    def refined_static_label() -> String:
        ...


trait StaticOriginalOverload:
    @staticmethod
    def overloaded_static() -> String:
        ...


trait StaticRefinedOverload:
    @staticmethod
    def overloaded_static(value: Int) -> String:
        ...


trait ArgConstructible(Deinitable):
    def __init__(out self, value: Int):
        ...


trait RefinedDefaultConstructible:
    def __init__(out self):
        ...


trait RefinedStringConstructible:
    @implicit
    def __init__(out self, value: String):
        ...


struct StaticDog(StaticLabel):
    def __init__(out self):
        pass

    @staticmethod
    def static_label() -> String:
        return "static dog"


struct StaticOriginalAndRefined(StaticOriginalLabel, StaticRefinedLabel):
    def __init__(out self):
        pass

    @staticmethod
    def original_static_label() -> String:
        return "original static"

    @staticmethod
    def refined_static_label() -> String:
        return "refined static"


struct StaticOriginalAndRefinedOverload(
    StaticOriginalOverload,
    StaticRefinedOverload,
):
    def __init__(out self):
        pass

    @staticmethod
    def overloaded_static() -> String:
        return "original overload"

    @staticmethod
    def overloaded_static(value: Int) -> String:
        return "refined overload"


struct RefinedConstructorWitness(
    ArgConstructible,
    RefinedDefaultConstructible,
    RefinedStringConstructible,
):
    def __init__(out self):
        pass

    def __init__(out self, value: Int):
        pass

    @implicit
    def __init__(out self, value: String):
        pass


struct TypeToken[T: AnyType]:
    def __init__(out self):
        pass


def require_type_token[T: AnyType](token: TypeToken[T]) -> String:
    return "token-ok"


def static_label_comptime_if[T: AnyType]() -> String:
    comptime if conforms_to(T, StaticLabel):
        return T.static_label()
    else:
        return "fallback"


def static_label_alias[T: AnyType]() -> String:
    comptime Alias = T
    comptime if conforms_to(T, StaticLabel):
        return Alias.static_label()
    return "fallback"


def static_label_type_of_value[
    T: AnyType
](x: T) -> String where conforms_to(T, StaticLabel):
    return type_of(x).static_label()


def static_label_preserves_original_bound[T: StaticOriginalLabel]() -> String:
    comptime if conforms_to(T, StaticRefinedLabel):
        return T.refined_static_label() + " | " + T.original_static_label()
    return "fallback"


def static_label_alias_preserves_original_bound[
    T: StaticOriginalLabel
]() -> String:
    comptime Alias = T
    comptime if conforms_to(T, StaticRefinedLabel):
        return (
            Alias.refined_static_label() + " | " + Alias.original_static_label()
        )
    return "fallback"


def static_label_type_of_preserves_original_bound[
    T: StaticOriginalLabel
](x: T) -> String:
    comptime if conforms_to(T, StaticRefinedLabel):
        return (
            type_of(x).refined_static_label()
            + " | "
            + type_of(x).original_static_label()
        )
    return "fallback"


def static_label_same_name_overload[T: StaticOriginalOverload]() -> String:
    comptime if conforms_to(T, StaticRefinedOverload):
        return T.overloaded_static(1)
    return "fallback"


def static_label_does_not_rewrite_type_uses[
    T: AnyType
](token: TypeToken[T]) -> String where conforms_to(T, StaticLabel):
    var label = T.static_label()
    return label + " | " + require_type_token[T](token)


def default_construct_refined[
    T: AnyType
]() where conforms_to(T, Defaultable & Deinitable):
    _ = T()


def default_construct_with_original_candidate[
    T: ArgConstructible
]() -> String where conforms_to(T, RefinedDefaultConstructible & Deinitable):
    _ = T()
    return "refined-constructor"


def original_construct_with_refined_candidates[
    T: ArgConstructible
]() -> String where conforms_to(T, RefinedDefaultConstructible & Deinitable):
    _ = T(1)
    return "original-constructor"


def accept_refined_implicit_conversion[
    T: ArgConstructible
](value: T) -> String where conforms_to(T, Deinitable):
    return "refined-implicit-conversion"


def implicit_construct_with_original_candidate[
    T: ArgConstructible
]() -> String where conforms_to(T, RefinedStringConstructible & Deinitable):
    return accept_refined_implicit_conversion[T](String("converted"))


def test_static_type_refinement():
    # CHECK: static-if: static dog
    print("static-if:", static_label_comptime_if[StaticDog]())
    # CHECK: static-if-fallback: fallback
    print("static-if-fallback:", static_label_comptime_if[Int]())
    # CHECK: static-alias: static dog
    print("static-alias:", static_label_alias[StaticDog]())
    # CHECK: static-type-of: static dog
    print("static-type-of:", static_label_type_of_value(StaticDog()))
    # CHECK: static-original-bound: refined static | original static
    print(
        "static-original-bound:",
        static_label_preserves_original_bound[StaticOriginalAndRefined](),
    )
    # CHECK: static-alias-original-bound: refined static | original static
    print(
        "static-alias-original-bound:",
        static_label_alias_preserves_original_bound[StaticOriginalAndRefined](),
    )
    # CHECK: static-type-of-original-bound: refined static | original static
    print(
        "static-type-of-original-bound:",
        static_label_type_of_preserves_original_bound(
            StaticOriginalAndRefined()
        ),
    )
    # CHECK: static-same-name-overload: refined overload
    print(
        "static-same-name-overload:",
        static_label_same_name_overload[StaticOriginalAndRefinedOverload](),
    )
    # CHECK: static-no-leak: static dog | token-ok
    print(
        "static-no-leak:",
        static_label_does_not_rewrite_type_uses[StaticDog](
            TypeToken[StaticDog]()
        ),
    )
    default_construct_refined[Int]()
    # CHECK: refined-constructor: refined-constructor
    print(
        "refined-constructor:",
        default_construct_with_original_candidate[RefinedConstructorWitness](),
    )
    # CHECK: original-constructor: original-constructor
    print(
        "original-constructor:",
        original_construct_with_refined_candidates[RefinedConstructorWitness](),
    )
    # CHECK: refined-implicit-conversion: refined-implicit-conversion
    print(
        "refined-implicit-conversion:",
        implicit_construct_with_original_candidate[RefinedConstructorWitness](),
    )


trait TypeArgMarker:
    pass


trait OriginalTypeArg:
    pass


trait RefinedTypeArg:
    pass


struct TypeArgWitness(TypeArgMarker):
    pass


struct OriginalAndRefinedTypeArg(OriginalTypeArg, RefinedTypeArg):
    pass


def accepts_type_arg[T: TypeArgMarker]() -> String:
    return "function"


def accepts_original_type_arg[T: OriginalTypeArg]() -> String:
    return "original"


def accepts_refined_type_arg[T: RefinedTypeArg]() -> String:
    return "refined"


@fieldwise_init
struct TypeArgBox[T: TypeArgMarker]:
    pass


def type_arg_function_binding[T: AnyType]() -> String:
    comptime if conforms_to(T, TypeArgMarker):
        return accepts_type_arg[T]()
    return "fallback"


def type_arg_struct_binding[T: AnyType]() -> String:
    comptime if conforms_to(T, TypeArgMarker):
        _ = TypeArgBox[T]()
        return "struct"
    return "fallback"


def type_arg_alias_binding[T: AnyType]() -> String:
    comptime Alias = T
    comptime if conforms_to(T, TypeArgMarker):
        _ = TypeArgBox[Alias]()
        return "alias"
    return "fallback"


struct TypeArgPack[*Ts: AnyType]:
    def __init__(out self):
        pass

    def bind_element[i: Int](self) -> String:
        comptime if conforms_to(Self.Ts[i], TypeArgMarker):
            _ = TypeArgBox[Self.Ts[i]]()
            return "variadic"
        return "fallback"


trait HasTypeArgElement:
    comptime Element: AnyType


struct TypeArgContainer(HasTypeArgElement):
    comptime Element = TypeArgWitness


def type_arg_associated_binding[C: AnyType]() -> String:
    comptime if conforms_to(C, HasTypeArgElement):
        comptime assert conforms_to(C.Element, TypeArgMarker)
        _ = TypeArgBox[C.Element]()
        return "associated"
    return "fallback"


def type_arg_preserves_original_bound[T: OriginalTypeArg]() -> String:
    comptime if conforms_to(T, RefinedTypeArg):
        return (
            accepts_refined_type_arg[T]()
            + " | "
            + accepts_original_type_arg[T]()
        )
    return "fallback"


def type_arg_alias_preserves_original_bound[T: OriginalTypeArg]() -> String:
    comptime Alias = T
    comptime if conforms_to(T, RefinedTypeArg):
        return (
            accepts_refined_type_arg[Alias]()
            + " | "
            + accepts_original_type_arg[Alias]()
        )
    return "fallback"


struct OriginalTypeArgPack[*Ts: OriginalTypeArg]:
    def __init__(out self):
        pass

    def bind_element[i: Int](self) -> String:
        comptime if conforms_to(Self.Ts[i], RefinedTypeArg):
            return (
                accepts_refined_type_arg[Self.Ts[i]]()
                + " | "
                + accepts_original_type_arg[Self.Ts[i]]()
            )
        return "fallback"


trait HasOriginalTypeArgElement:
    comptime Element: OriginalTypeArg


struct OriginalTypeArgContainer(HasOriginalTypeArgElement):
    comptime Element = OriginalAndRefinedTypeArg


def type_arg_associated_preserves_original_bound[C: AnyType]() -> String:
    comptime if conforms_to(C, HasOriginalTypeArgElement):
        comptime assert conforms_to(C.Element, RefinedTypeArg)
        return (
            accepts_refined_type_arg[C.Element]()
            + " | "
            + accepts_original_type_arg[C.Element]()
        )
    return "fallback"


def test_type_value_parameter_refinement():
    # CHECK: type-arg-function: function
    print("type-arg-function:", type_arg_function_binding[TypeArgWitness]())
    # CHECK: type-arg-struct: struct
    print("type-arg-struct:", type_arg_struct_binding[TypeArgWitness]())
    # CHECK: type-arg-alias: alias
    print("type-arg-alias:", type_arg_alias_binding[TypeArgWitness]())
    # CHECK: type-arg-variadic: variadic
    print(
        "type-arg-variadic:",
        TypeArgPack[TypeArgWitness]().bind_element[0](),
    )
    # CHECK: type-arg-associated: associated
    print(
        "type-arg-associated:",
        type_arg_associated_binding[TypeArgContainer](),
    )
    # CHECK: type-arg-original-bound: refined | original
    print(
        "type-arg-original-bound:",
        type_arg_preserves_original_bound[OriginalAndRefinedTypeArg](),
    )
    # CHECK: type-arg-alias-original-bound: refined | original
    print(
        "type-arg-alias-original-bound:",
        type_arg_alias_preserves_original_bound[OriginalAndRefinedTypeArg](),
    )
    # CHECK: type-arg-variadic-original-bound: refined | original
    print(
        "type-arg-variadic-original-bound:",
        OriginalTypeArgPack[OriginalAndRefinedTypeArg]().bind_element[0](),
    )
    # CHECK: type-arg-associated-original-bound: refined | original
    print(
        "type-arg-associated-original-bound:",
        type_arg_associated_preserves_original_bound[
            OriginalTypeArgContainer
        ](),
    )


def call_greet_read[
    T: AnyType
](imm x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def call_greet_mut[
    T: AnyType
](mut x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def call_greet_ref[
    T: AnyType
](ref x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def call_greet_var[
    T: Deinitable
](var x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def test_conventions():
    var d = Dog("Rex")
    # CHECK: read: Woof! I'm Rex
    print("read:", call_greet_read(d))
    # CHECK: mut: Woof! I'm Rex
    print("mut:", call_greet_mut(d))
    # CHECK: ref: Woof! I'm Rex
    print("ref:", call_greet_ref(d))
    # CHECK: var: Woof! I'm Rex
    print("var:", call_greet_var(Dog("Rex")))


def copyable_and_greetable[
    T: Copyable & Deinitable
](imm x: T) -> String where conforms_to(T, Greetable):
    var y = x.copy()
    return y.greet()


def test_bound_preservation():
    var d = Dog("Rex")
    # CHECK: Woof! I'm Rex
    print(copyable_and_greetable(d))


def needs_greetable[T: Greetable](x: T) -> String:
    return x.greet()


def needs_describable[T: Describable](x: T) -> String:
    return x.describe()


def greet_and_describe[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable) and conforms_to(
    T, Describable
):
    return x.greet() + " | " + x.describe()


def call_both_from_conjunction[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable) and conforms_to(
    T, Describable
):
    return needs_greetable(x) + " | " + needs_describable(x)


def test_conjunction():
    var d = Dog("Spot")
    # CHECK: Woof! I'm Spot | Dog(Spot)
    print(greet_and_describe(d))
    # CHECK: Woof! I'm Spot | Dog(Spot)
    print(call_both_from_conjunction(d))


trait Countable:
    def count(self) -> Int:
        ...


struct Widget(
    Copyable,
    Countable,
    Deinitable,
    Describable,
    Greetable,
    Movable,
):
    var name: String
    var n: Int

    def __init__(out self, name: String, n: Int):
        self.name = name
        self.n = n

    def greet(self) -> String:
        return "Hi from " + self.name

    def describe(self) -> String:
        return "Widget(" + self.name + ")"

    def count(self) -> Int:
        return self.n


def use_all_three[
    T: Movable
](
    x: T,
) -> String where (
    conforms_to(T, Greetable)
    and conforms_to(T, Describable)
    and conforms_to(T, Countable)
):
    return x.greet() + " | " + x.describe() + " | " + String(x.count())


def and_alongside_or[
    T: Movable, U: Movable
](x: T, y: U) -> String where conforms_to(T, Greetable) and conforms_to(
    T, Describable
):
    return x.greet() + " | " + x.describe()


def test_deep_conjunction():
    var w = Widget("Bolt", 42)
    # CHECK: Hi from Bolt | Widget(Bolt) | 42
    print(use_all_three(w))

    # AND still works when unrelated param U exists
    var dummy = 0
    # CHECK: Hi from Bolt | Widget(Bolt)
    print(and_alongside_or(w, dummy))


def inner_greet[T: Greetable](x: T) -> String:
    return x.greet()


def middle_greet[T: Movable](x: T) -> String where conforms_to(T, Greetable):
    return inner_greet(x)


def outer_greet[T: Movable](x: T) -> String where conforms_to(T, Greetable):
    return middle_greet(x)


def test_passthrough():
    var d = Dog("Buddy")
    # CHECK: Woof! I'm Buddy
    print(needs_greetable(d))
    # CHECK: Woof! I'm Buddy
    print(outer_greet(d))


def selective[
    A: Movable, B: Movable
](a: A, b: B) -> String where conforms_to(A, Greetable):
    return a.greet()


def test_selective_refinement():
    # CHECK: Woof! I'm Scout
    print(selective(Dog("Scout"), Cat("Shadow")))


trait HasFoo:
    def foo(self) -> Int:
        ...


trait HasBar:
    def bar(self) -> Int:
        ...


struct FooBar(Deinitable, HasBar, HasFoo):
    def __init__(out self):
        pass

    def foo(self) -> Int:
        return 10

    def bar(self) -> Int:
        return 20


def use_both[T: HasFoo](x: T) -> Int where conforms_to(T, HasBar):
    return x.foo() + x.bar()


def test_per_candidate_self_binding():
    # CHECK: 30
    print(use_both(FooBar()))


struct Box[T: Deinitable & Movable]:
    var item: Self.T

    def __init__(out self, var item: Self.T):
        self.item = item^

    def greet_item(self) -> String where conforms_to(Self.T, Greetable):
        return self.item.greet()

    def describe_item(self) -> String where conforms_to(Self.T, Describable):
        return self.item.describe()

    def greet_and_describe_item(
        self,
    ) -> String where conforms_to(Self.T, Greetable) and conforms_to(
        Self.T, Describable
    ):
        return self.item.greet() + " | " + self.item.describe()


struct GreetableBox[T: Deinitable & Movable](
    Deinitable,
    Greetable where conforms_to(T, Greetable),
    Movable,
):
    var item: Self.T

    def __init__(out self, var item: Self.T):
        self.item = item^

    def greet(self) -> String where conforms_to(Self.T, Greetable):
        return "Box says: " + self.item.greet()


def use_greetable[T: Greetable](x: T):
    print(x.greet())


def test_struct_methods():
    var dog_box = Box(Dog("Rocky"))
    # CHECK: Woof! I'm Rocky
    print(dog_box.greet_item())
    # CHECK: Dog(Rocky)
    print(dog_box.describe_item())
    # CHECK: Woof! I'm Rocky | Dog(Rocky)
    print(dog_box.greet_and_describe_item())

    var gb = GreetableBox(Dog("Bolt"))
    # CHECK: Box says: Woof! I'm Bolt
    use_greetable(gb)


def describe_or_fallback[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable):
    comptime if conforms_to(T, Describable):
        return x.describe()
    else:
        return x.greet()


def assert_then_greet[T: Movable](x: T) -> String:
    comptime assert conforms_to(T, Greetable), "T must be Greetable"
    return x.greet()


def assert_multi_then_use[T: Movable](x: T) -> String:
    comptime assert conforms_to(T, Greetable) and conforms_to(T, Describable)
    return x.greet() + " | " + x.describe()


def stacked_refinement[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable):
    comptime assert conforms_to(T, Describable), "T must be Describable"
    return needs_greetable(x) + " and " + needs_describable(x)


def test_comptime_scopes():
    var d = Dog("Fido")
    # CHECK: Dog(Fido)
    print(describe_or_fallback(d))
    # CHECK: Meow! I'm Luna
    print(describe_or_fallback(Cat("Luna")))
    # CHECK: Woof! I'm Fido
    print(assert_then_greet(d))
    # CHECK: Woof! I'm Fido | Dog(Fido)
    print(assert_multi_then_use(d))
    # CHECK: Woof! I'm Fido and Dog(Fido)
    print(stacked_refinement(d))


struct RegGreeter(Greetable, TrivialRegisterPassable):
    var x: Int

    def __init__(out self):
        self.x = 0

    def greet(self) -> String:
        return "Hello from RegGreeter"


struct RegDescGreeter(Describable, Greetable, TrivialRegisterPassable):
    var x: Int

    def __init__(out self):
        self.x = 0

    def greet(self) -> String:
        return "Hi from RegDescGreeter"

    def describe(self) -> String:
        return "RegDescGreeter described"


def reg_read[
    T: TrivialRegisterPassable
](imm x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def reg_conjunction[
    T: TrivialRegisterPassable
](imm x: T) -> String where conforms_to(T, Greetable) and conforms_to(
    T, Describable
):
    return x.greet() + " | " + x.describe()


def takes_trp_greet[
    T: TrivialRegisterPassable
](x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


def takes_rp_greet[
    T: RegisterPassable
](x: T) -> String where conforms_to(T, Greetable):
    return x.greet()


# MOCO-4230: refining a generic from a non-register-passable bound up to
# `TrivialRegisterPassable` via a where-clause, then passing it by value to a
# `TrivialRegisterPassable`-bounded parameter (which loads it into an SSA
# register). The refined bound must be honored at the SSA-load gate.
def refine_trp_where_then_pass[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable) and conforms_to(
    T, TrivialRegisterPassable
):
    return takes_trp_greet(x)


# Same shape, but the refinement comes from a `comptime assert` rather than a
# where-clause.
def refine_trp_assert_then_pass[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable):
    comptime assert conforms_to(T, TrivialRegisterPassable)
    return takes_trp_greet(x)


# The non-trivial `RegisterPassable` refinement in the same shape.
def refine_rp_assert_then_pass[
    T: Movable
](x: T) -> String where conforms_to(T, Greetable):
    comptime assert conforms_to(T, RegisterPassable)
    return takes_rp_greet(x)


def test_register_passable():
    # CHECK: Hello from RegGreeter
    print(reg_read(RegGreeter()))
    # CHECK: Hi from RegDescGreeter | RegDescGreeter described
    print(reg_conjunction(RegDescGreeter()))
    # CHECK: refine-trp-where: Hello from RegGreeter
    print("refine-trp-where:", refine_trp_where_then_pass(RegGreeter()))
    # CHECK: refine-trp-assert: Hello from RegGreeter
    print("refine-trp-assert:", refine_trp_assert_then_pass(RegGreeter()))
    # CHECK: refine-rp-assert: Hello from RegGreeter
    print("refine-rp-assert:", refine_rp_assert_then_pass(RegGreeter()))


struct Wrapper[T: Deinitable & Movable]:
    comptime Element = Self.T
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def get(ref self) -> ref[self.value] Self.Element:
        return self.value


struct DoubleAliasWrapper[T: Deinitable & Movable]:
    comptime Inner = Self.T
    comptime Element = Self.Inner
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def get(ref self) -> ref[self.value] Self.Element:
        return self.value


struct Pair[A: Deinitable & Movable, B: Deinitable & Movable]:
    comptime First = Self.A
    comptime Second = Self.B
    var first: Self.A
    var second: Self.B

    def __init__(out self, var first: Self.A, var second: Self.B):
        self.first = first^
        self.second = second^

    def get_first(ref self) -> ref[self.first] Self.First:
        return self.first

    def get_second(ref self) -> ref[self.second] Self.Second:
        return self.second


def greet_wrapper[
    T: Deinitable & Movable
](w: Wrapper[T]) -> String where conforms_to(T, Greetable):
    return w.get().greet()


def greet_double_alias[
    T: Deinitable & Movable
](w: DoubleAliasWrapper[T]) -> String where conforms_to(T, Greetable):
    return w.get().greet()


def greet_first[
    A: Deinitable & Movable,
    B: Deinitable & Movable,
](p: Pair[A, B]) -> String where conforms_to(A, Greetable):
    return p.get_first().greet()


def test_comptime_aliases():
    # CHECK: Woof! I'm Daisy
    print(greet_wrapper(Wrapper(Dog("Daisy"))))
    # CHECK: Meow! I'm Nala
    print(greet_double_alias(DoubleAliasWrapper(Cat("Nala"))))
    # CHECK: Meow! I'm Socks
    print(greet_first(Pair(Cat("Socks"), Dog("Bear"))))


trait SimpleIterator(Deinitable, Movable):
    comptime Element: Movable

    def next(mut self) -> Self.Element:
        ...


trait SimpleIterable:
    comptime IterType: SimpleIterator

    def iter(self) -> Self.IterType:
        ...


struct DogIter(Deinitable, Movable, SimpleIterator):
    comptime Element = Dog
    var dog: Dog

    def __init__(out self, var dog: Dog):
        self.dog = dog^

    def next(mut self) -> Dog:
        return self.dog.copy()


struct DogContainer(SimpleIterable):
    comptime IterType = DogIter
    var dog: Dog

    def __init__(out self, var dog: Dog):
        self.dog = dog^

    def iter(self) -> DogIter:
        return DogIter(self.dog.copy())


def any_greetable[
    C: SimpleIterable
](container: C) -> String where conforms_to(
    C.IterType.Element, Greetable & Deinitable & Movable
):
    var it = container.iter()
    var elem = it.next()
    return elem.greet()


def greet_and_describe_nested[
    C: SimpleIterable
](container: C) -> String where conforms_to(
    C.IterType.Element, Greetable & Deinitable & Movable
) and conforms_to(C.IterType.Element, Describable):
    var it = container.iter()
    var elem = it.next()
    return elem.greet() + " | " + elem.describe()


def discard_associated_iterator_element[
    C: SimpleIterable
](container: C) -> String where conforms_to(
    C.IterType.Element, Deinitable & Movable
):
    var it = container.iter()
    _ = it.next()
    return "discarded"


def test_associated_type_chains():
    # CHECK: Woof! I'm Ziggy
    print(any_greetable(DogContainer(Dog("Ziggy"))))
    # CHECK: Woof! I'm Nova | Dog(Nova)
    print(greet_and_describe_nested(DogContainer(Dog("Nova"))))
    # CHECK: discarded
    print(discard_associated_iterator_element(DogContainer(Dog("Buddy"))))


def call_greet_ptr[
    T: Copyable & Deinitable, O: Origin
](p: Pointer[T, O]) -> String where conforms_to(T, Greetable):
    return p[].greet()


def ptr_eq[
    T: Copyable & Deinitable, O: Origin
](p: Pointer[T, O], value: T) -> Bool where conforms_to(T, Equatable):
    return p[] == value


def accepts_original_ref[T: AnyType](ref x: T) -> Bool:
    return True


def refined_ref_to_original_ref[
    T: AnyType
](ref x: T) -> Bool where conforms_to(T, Equatable):
    return accepts_original_ref[T](x)


def explicit_downcast_ref_to_original_ref[
    T: AnyType
](ref x: T) -> Bool where conforms_to(T, Equatable):
    ref y = rebind[downcast[T, Equatable]](x)
    return accepts_original_ref[T](y)


@fieldwise_init
struct RefinedFieldHolder[T: Copyable & Deinitable]:
    var value: Self.T

    def use_value(self) -> Bool where conforms_to(Self.T, Equatable):
        return accepts_original_ref[Self.T](self.value)


@fieldwise_init
struct ReflectedEquatable(Equatable):
    var value: Int


def reflected_refined_arg_field_eq[
    T: AnyType
](a: T, b: T) -> Bool where conforms_to(T, Equatable):
    comptime field_types = reflect[T].field_types()

    comptime for idx in range(reflect[T].field_count()):
        comptime if conforms_to(field_types[idx], Equatable):
            ref lhs = reflect[T].field_ref[idx](a)
            ref rhs = reflect[T].field_ref[idx](b)
            if lhs != rhs:
                return False
    return True


def test_pointer_deref():
    var d = Dog("Rex")
    # CHECK: Woof! I'm Rex
    print(call_greet_ptr(Pointer(to=d)))

    var x = 42
    # CHECK: True
    print(ptr_eq(Pointer(to=x), 42))
    # CHECK: False
    print(ptr_eq(Pointer(to=x), 99))
    # CHECK: refined-ref-original: True
    print("refined-ref-original:", refined_ref_to_original_ref(x))
    # CHECK: explicit-downcast-original: True
    print(
        "explicit-downcast-original:", explicit_downcast_ref_to_original_ref(x)
    )
    # CHECK: refined-field-original: True
    print("refined-field-original:", RefinedFieldHolder[Int](2).use_value())
    # CHECK: reflected-ref-field: True
    print(
        "reflected-ref-field:",
        reflected_refined_arg_field_eq(
            ReflectedEquatable(1), ReflectedEquatable(1)
        ),
    )


def local_ref_binding[
    T: AnyType
](imm x: T) -> String where conforms_to(T, Greetable):
    ref y = x
    return y.greet()


def local_var_copy[
    T: Copyable & Deinitable
](imm x: T) -> String where conforms_to(T, Greetable):
    var y: T = x.copy()
    return y.greet()


def local_var_move[
    T: Movable & Deinitable
](var x: T) -> String where conforms_to(T, Greetable):
    var y: T = x^
    return y.greet()


def tuple_from_refined_args[
    T: Copyable & Deinitable
](imm a: T, imm b: T) -> String where conforms_to(T, Greetable):
    var t = (a.copy(), b.copy())
    return t[0].greet() + " and " + t[1].greet()


def nested_comptime_outer_binding_refinement[
    T: Copyable & Deinitable
](imm x: T) where conforms_to(T, Greetable) and conforms_to(T, Describable):
    var val: T = x.copy()
    print("outer:", val.greet())
    comptime if conforms_to(T, Describable):
        # The outer `val` should inherit Describable in this refined branch.
        print("outer-inner-greet:", needs_greetable(val))
        print("outer-inner-describe:", val.describe())


def nested_comptime_inner_shadow_refinement[
    T: Copyable & Deinitable
](imm y: T) where conforms_to(T, Greetable) and conforms_to(T, Describable):
    comptime if conforms_to(T, Describable):
        # Inner scope adds Describable on top of the where-clause Greetable.
        var val: T = y.copy()
        print("inner-greet:", needs_greetable(val))
        print("inner:", val.describe())


def nested_comptime_fallback_refinement[
    T: Copyable & Deinitable
](imm x: T, imm y: T) where conforms_to(T, Greetable):
    var val: T = x.copy()
    print("outer:", val.greet())
    comptime if not conforms_to(T, Describable):
        var val: T = y.copy()
        print("inner-fallback:", needs_greetable(val))


def test_local_vars():
    var d = Dog("Rex")
    # CHECK: ref: Woof! I'm Rex
    print("ref:", local_ref_binding(d))
    # CHECK: copy: Woof! I'm Rex
    print("copy:", local_var_copy(d))
    # CHECK: move: Woof! I'm Rex
    print("move:", local_var_move(Dog("Rex")))
    # CHECK: tuple: Woof! I'm Rex and Woof! I'm Fido
    print("tuple:", tuple_from_refined_args(d, Dog("Fido")))
    # CHECK: outer: Woof! I'm Rex
    # CHECK: outer-inner-greet: Woof! I'm Rex
    # CHECK: outer-inner-describe: Dog(Rex)
    nested_comptime_outer_binding_refinement(d)
    # CHECK: inner-greet: Woof! I'm Fido
    # CHECK: inner: Dog(Fido)
    nested_comptime_inner_shadow_refinement(Dog("Fido"))
    # CHECK: outer: Meow! I'm Luna
    # CHECK: inner-fallback: Meow! I'm Shadow
    nested_comptime_fallback_refinement(Cat("Luna"), Cat("Shadow"))


trait DtorMarker:
    pass


def use_refined_dtor[T: AnyType](ref x: T):
    print("use-dtor")


struct DestructTracer(Copyable, Deinitable, Movable):
    var tag: Int

    def __init__(out self, tag: Int):
        self.tag = tag
        print("init-dtor", self.tag)

    def __deinit__(deinit self, /):
        print("deinit-dtor", self.tag)


struct MarkedDestructTracer(Copyable, Deinitable, DtorMarker, Movable):
    var tag: Int

    def __init__(out self, tag: Int):
        self.tag = tag
        print("init-dtor", self.tag)

    def __deinit__(deinit self, /):
        print("deinit-dtor", self.tag)


def destroy_via_where[T: Movable](var x: T) where conforms_to(T, Deinitable):
    use_refined_dtor(x)


def destroy_via_comptime_assert[T: Movable](var x: T):
    comptime assert conforms_to(T, Deinitable)
    use_refined_dtor(x)


def destroy_via_comptime_if[T: Movable](var x: T):
    comptime if conforms_to(T, Deinitable):
        use_refined_dtor(x)
    else:
        # Keep both branches consuming `x` symmetrically.
        std.memory.forget_deinit(x^)


def destroy_via_chained_refinement[
    T: Movable
](var x: T) where conforms_to(T, DtorMarker):
    comptime assert conforms_to(T, Deinitable)
    use_refined_dtor(x)


def test_refined_implicitly_destructible():
    # CHECK-LABEL: dtor-where
    print("dtor-where")
    # CHECK:      init-dtor 1
    # CHECK-NEXT: use-dtor
    # CHECK-NEXT: deinit-dtor 1
    destroy_via_where(DestructTracer(1))

    # CHECK-LABEL: dtor-comptime-assert
    print("dtor-comptime-assert")
    # CHECK:      init-dtor 2
    # CHECK-NEXT: use-dtor
    # CHECK-NEXT: deinit-dtor 2
    destroy_via_comptime_assert(DestructTracer(2))

    # CHECK-LABEL: dtor-comptime-if
    print("dtor-comptime-if")
    # CHECK:      init-dtor 3
    # CHECK-NEXT: use-dtor
    # CHECK-NEXT: deinit-dtor 3
    destroy_via_comptime_if(DestructTracer(3))

    # CHECK-LABEL: dtor-chained-refinement
    print("dtor-chained-refinement")
    # CHECK:      init-dtor 4
    # CHECK-NEXT: use-dtor
    # CHECK-NEXT: deinit-dtor 4
    destroy_via_chained_refinement(MarkedDestructTracer(4))


# The scope-refinement machinery must not interfere with user-authored
# `trait_downcast` calls. Each case scope-refines `T` with `Describable`,
# consumes that refinement (`needs_describable`), drops a user-written
# `trait_downcast` to an unrelated trait (`Greetable`),
# then re-consumes the original `Describable` refinement. If refinement leaks
# into the downcast's input/output typing — or conversely the downcast's
# result perturbs `x`'s refined state — either `needs_describable(x)` call
# would fail to typecheck, or `trait_downcast` would produce a type
# incompatible with `needs_greetable`.
def downcast_preserves_refinement_read[
    T: AnyType
](imm x: T) -> String where conforms_to(T, Describable):
    var before = needs_describable(x)
    ref y = rebind[downcast[T, Greetable]](x)
    var mid = needs_greetable(y)
    var after = needs_describable(x)
    return before + " | " + mid + " | " + after


def downcast_preserves_refinement_mut[
    T: AnyType
](mut x: T) -> String where conforms_to(T, Describable):
    var before = needs_describable(x)
    ref y = rebind[downcast[T, Greetable]](x)
    var mid = needs_greetable(y)
    var after = needs_describable(x)
    return before + " | " + mid + " | " + after


def downcast_preserves_refinement_ref[
    T: AnyType
](ref x: T) -> String where conforms_to(T, Describable):
    var before = needs_describable(x)
    ref y = rebind[downcast[T, Greetable]](x)
    var mid = needs_greetable(y)
    var after = needs_describable(x)
    return before + " | " + mid + " | " + after


# `rebind_var` consumes its argument, so copy to keep `x` live and
# verify its Describable refinement survives across the owning downcast call.
def downcast_preserves_refinement_var[
    T: Copyable & Deinitable
](var x: T) -> String where conforms_to(T, Describable):
    var before = needs_describable(x)
    var y = rebind_var[downcast[T, Greetable]](x.copy())
    var mid = needs_greetable(y)
    var after = needs_describable(x)
    return before + " | " + mid + " | " + after


# Same contract as the `trait_downcast` cases above, but with a hand-rolled
# `kgen.rebind` + `downcast[T, Greetable]` the way a user would synthesize
# one directly. Scope refinement must leave the user's written `_type`
# attribute alone.
#
# Regression guard for the canonicalization fix in
# `mergeOriginalAndRefinedBounds` (Mojo/lib/MojoParser/ExprNodes.cpp). When
# the `var`'s declared type embeds `DowncastAttr(T, trait<@Greetable>)`, the
# original trait set must be canonicalized (to pick up implicit ancestors
# like `@AnyType`) before checking whether the scope's assumptions add any
# new traits. Without that, `refinedBound` — which arrives already
# canonicalized from `getTraitBoundFromAssumptions` — appears to introduce
# `@AnyType`, and refinement clobbers the user's `downcast[T, Greetable]` by
# broadening the `var`'s type to `T(AnyType & Greetable)`, disagreeing with
# the `_type` the user wrote on `kgen.rebind` below.
def rebind_manual_ref[
    T: AnyType
](ref x: T) -> String where conforms_to(T, Greetable):
    var _lit = __get_mvalue_as_litref(x)
    var _rebound = __mlir_op.`kgen.rebind`[
        _type=Pointer[downcast[T, Greetable], origin_of(x)]._mlir_lit_ref
    ](_lit)
    ref y = __get_litref_as_mvalue(_rebound)
    return y.greet()


def greet_variadic_elements[*Ts: AnyType](*args: *Ts):
    comptime for i in range(args.__len__()):
        comptime element_type = Ts[i]
        comptime assert conforms_to(element_type, Greetable)
        print("variadic-refine:", args[i].greet())


def copy_variadic_elements_from_all_copyable[
    *Ts: Deinitable & Movable
](*args: *Ts) where Ts.all_conforms_to[Copyable]():
    comptime for i in range(args.__len__()):
        # The variadic all_conforms_to() constraint should refine each element.
        _ = args[i].copy()
    print("variadic-allcopyable-refine: ok")


def copy_variadic_elements_from_conforms_to_param_list[
    *Ts: Deinitable & Movable
](*args: *Ts) where conforms_to(Ts.values, Copyable):
    comptime for i in range(args.__len__()):
        _ = args[i].copy()
    print("variadic-conforms-to-param-list-refine: ok")


def repr_variadic_tuple_after_all_writable_assert[
    *Ts: Movable
](t: Tuple[*Ts]) -> String:
    comptime assert Ts.all_conforms_to[Writable]()
    return repr(t)


def repr_variadic_tuple_in_all_writable_if[
    *Ts: Movable
](t: Tuple[*Ts]) -> String:
    comptime if Ts.all_conforms_to[Writable]():
        return repr(t)
    return "fallback"


def copy_variadic_elements_after_all_copyable_assert[
    *Ts: Deinitable & Movable
](*args: *Ts):
    comptime assert Ts.all_conforms_to[Copyable]()
    comptime for i in range(args.__len__()):
        _ = args[i].copy()
    print("variadic-allcopyable-assert-refine: ok")


def copy_variadic_elements_in_all_copyable_if[
    *Ts: Deinitable & Movable
](*args: *Ts):
    comptime if Ts.all_conforms_to[Copyable]():
        comptime for i in range(args.__len__()):
            _ = args[i].copy()
        print("variadic-allcopyable-if-refine: ok")
    else:
        print("variadic-allcopyable-if-refine: fallback")


def test_rebind_refinement():
    var d = Dog("Rex")
    # CHECK: read: Dog(Rex) | Woof! I'm Rex | Dog(Rex)
    print("read:", downcast_preserves_refinement_read(d))
    var d_mut = Dog("Rex")
    # CHECK: mut: Dog(Rex) | Woof! I'm Rex | Dog(Rex)
    print("mut:", downcast_preserves_refinement_mut(d_mut))
    # CHECK: ref: Dog(Rex) | Woof! I'm Rex | Dog(Rex)
    print("ref:", downcast_preserves_refinement_ref(d))
    # CHECK: var: Dog(Rex) | Woof! I'm Rex | Dog(Rex)
    print("var:", downcast_preserves_refinement_var(Dog("Rex")))
    # CHECK: manual-ref: Woof! I'm Rex
    print("manual-ref:", rebind_manual_ref(d))


def test_variadic_element_refinement():
    var t = (1, "hello")
    # CHECK: variadic-refine: Woof! I'm Rex
    # CHECK: variadic-refine: Meow! I'm Luna
    greet_variadic_elements(Dog("Rex"), Cat("Luna"))
    # CHECK: variadic-allcopyable-refine: ok
    copy_variadic_elements_from_all_copyable(Dog("Rex"), Cat("Luna"))
    # CHECK: variadic-conforms-to-param-list-refine: ok
    copy_variadic_elements_from_conforms_to_param_list(Dog("Rex"), Cat("Luna"))
    # CHECK: variadic-allwritable-assert-refine: Tuple[SIMD[DType.int, 1], String](Int(1), 'hello')
    print(
        "variadic-allwritable-assert-refine:",
        repr_variadic_tuple_after_all_writable_assert(t),
    )
    # CHECK: variadic-allwritable-if-refine: Tuple[SIMD[DType.int, 1], String](Int(1), 'hello')
    print(
        "variadic-allwritable-if-refine:",
        repr_variadic_tuple_in_all_writable_if(t),
    )
    # CHECK: variadic-allcopyable-assert-refine: ok
    copy_variadic_elements_after_all_copyable_assert(1, "hello")
    # CHECK: variadic-allcopyable-if-refine: ok
    copy_variadic_elements_in_all_copyable_if(1, "hello")


def main():
    test_static_type_refinement()
    test_type_value_parameter_refinement()
    test_conventions()
    test_bound_preservation()
    test_conjunction()
    test_deep_conjunction()
    test_passthrough()
    test_selective_refinement()
    test_per_candidate_self_binding()
    test_struct_methods()
    test_comptime_scopes()
    test_register_passable()
    test_comptime_aliases()
    test_associated_type_chains()
    test_pointer_deref()
    test_local_vars()
    test_refined_implicitly_destructible()
    test_rebind_refinement()
    test_variadic_element_refinement()
