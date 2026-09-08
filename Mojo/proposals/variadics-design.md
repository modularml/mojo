# Variadics in Mojo

**Status**: Draft

## Introduction

This document describes how variadic features work in Mojo at a level that mixes
user-facing behavior with enough implementation detail to explain lowering and
stdlib idioms. It is meant as a working reference for language and standard
library contributors, and for advanced users who need to reason about what the
compiler builds when variadic syntax is used.

Mojo’s variadic story spans several related ideas. At comptime, parameter
lists can bundle types or values into lists (`TypeList`, `ParameterList`, and
related metaprogramming patterns). At runtime call sites, homogeneous variadic
arguments let a single argument accept an arbitrary number of
same-shaped values, backed by `VariadicList`, an arbitrary number of mixed-type
values (backed by `VariadicPack`), and a set of otherwise unhandled keyword
arguments (backed by `OwnedKwargsDict`).

These are expressed with Python-style syntax: variadic lists (`*args`) and
variadic keyword arguments (`**kwargs`) are familiar ergonomically. The variadic
pack syntax matches up a list of variadic arguments with a list of variadic
types using the `*args: *Ts` syntax.

The sections that follow walk parameter-list variadics first, then runtime
homogeneous variadic arguments (including lowering sketches and splatting),
variadic packs and keyword variadics. Examples in the body complement the tests
under the open-source tree (for example
`stdlib/test/builtin/test_variadic.mojo`), which remain the ground
truth for up-to-date syntax.

## Variadic parameter lists

Variadic **parameter** lists appear in compile-time parameter positions (on
`def`, `struct`, `comptime`, etc). They bundle a statically known
number of elements into one binding. Like Python, Mojo uses a leading `*`
to indicate a list of values or types. These have a consistent metatype, e.g.

```mojo
def takes_types[*Ts: AnyType](): ...
def takes_values[*elts: Int](): ...
```

Mojo distinguishes these two cases, and the standard library surfaces them as
`TypeList` and `ParameterList` in `stdlib/std/builtin/variadics.mojo`. Both are
built on the same KGEN notion of a parameter list: an MLIR `!kgen.param_list`
value whose elements share one compile-time shape (either all types obeying a
trait, or all values of one type). They are builtins; you do not import them for
normal use.

**Type lists** bind a sequence of types. The declaration uses a type trait
instead of a value type, for example `*Ts: AnyType` or `*Ts: Movable`. The
compiler exposes them with `TypeList` operations: given the example above,
`type_of(Ts)` will be a `TypeList`. This gives access to a number of useful
operations, including a comptime `size`, indexing at fixed indices (for example
`Ts[i]` in `comptime` loops), and helpers such as `TypeList.of`, `splat`,
`tabulate`, `map`, `reduce`, `filter_idx`, and `contains`. Typical uses are
trait predicates over several type parameters (see
`stdlib/std/traits/movable.mojo`) and metaprogramming that walks or
transforms type packs.

**Value lists** bind a sequence of compile-time **values** that all share one
type `T`, for example `*args: Int` or `*names: String`. That becomes a
`ParameterList` with element type `T`. You can iterate or index at compile time,
and `ParameterList.get_span()` materializes the elements as a contiguous
constant array behind a `Span` when you need pointer-linear layout. Constructors
such as `ParameterList.of` and `splat` mirror the `TypeList` story for values.

The two are intentionally parallel (and we would love to unify them in the
future as Mojo's metatype story improves): `TypeList.map_to_values` maps each
element type through a compile-time generator and produces a
`ParameterList` of homogeneous values, which is a common pattern when a type
pack drives a value pack.

Unlike argument lists, parameter lists don't support heterogenous list or
keyword argument lists. This may be added in the future if there is sufficient
demand to justify the complexity.

## Homogeneous variadic arguments

The previous section was about **parameters** on declarations. This section is
about **arguments** at runtime call sites: a single parameter position that
accepts any number of arguments, as long as every argument has the same
type `T`. The surface syntax is again a leading `*`, but now it names a
`VariadicList` (not a `ParameterList`).

### Signatures and basic use

A homogeneous variadic parameter looks like `*args: Int` or `*parts: String`.
Inside the callee you can treat `args` much like a small sequential collection:
`len(args)`, `args[i]`, `for x in args`, and (when `T` is `Writable`) helpers
such as `args.write_to(s)` to render the tuple shape.

```mojo
def print_ints(*values: Int):
    for i in range(len(values)):
        print(values[i])

def main():
    print_ints(10, 20, 30)  # three separate arguments, one variadic binding
```

The iterator path is equally natural when you only need values without their
index:

```mojo
def sum_ints(*values: Int) -> Int:
    var total = 0
    for v in values:
        total += v
    return total
```

Please take a look at more examples in `stdlib/test/builtin/test_variadic.mojo`
to see more idioms.

### The callee receives `VariadicList`

The `*args: T` syntax is mapped to the standard library type `VariadicList` in
`stdlib/std/builtin/variadics.mojo`. It carries a `Span` of
**pointers to references** to each actual argument, not a dense array of `T` by
value. That layout lets the callee expose each element as `args[idx]` with the
right borrow or move semantics while keeping the glue object small enough to
pass by value. This also enables values that are not `Movable` to be passed
through variadics.

Rough lowering picture (names only, not exact MLIR):

```text
// Caller:  foo(a, b, c)   with   def foo(*xs: T)

1. The compiler allocates storage for each argument as usual (stack slots,
   registers, or whatever the ABI requires).

2. It builds a temporary array whose i-th entry is a pointer-to-ref for the
   i-th argument (all elements have the same pointer/ref MLIR type).

3. It calls `foo` passing a `VariadicList` whose internal `Span` points at that
   array and whose length is the argument count.

4. `VariadicList.__init__` (the implicit constructor from that array) casts the
   POP array to a `Span` of element pointers; `__getitem__` indexes through that
   span and loads the reference.
```

So the variadic bundle is always a span of pointers, even when `T` is
trivial. The important part for reading performance docs is that variadic
arguments are still separate objects in the caller; the list is an indirection
layer for uniform indexing.

### Borrowed versus owned arguments

By default, `*args: T` borrows each element. When you need to take ownership
(for example `T` is a linear type, or you want to move out with
`consume_elements`), use `var *args: T`. The `VariadicList` type tracks that
with the `is_owned` parameter. When set, `__del__` walks the list in reverse and
destroys each element, matching normal argument teardown order.

```mojo
@__parameter
def destroy_elem(_idx: Int, var arg: ExplicitDelOnly):
    arg^.destroy()

def take_owned_linear(var *args: ExplicitDelOnly):
    args^.consume_elements[destroy_elem]()

# Caller passes temporaries; callee consumes them one by one.
take_owned_linear(ExplicitDelOnly(5), ExplicitDelOnly(10))
```

That example is taken directly from `test_variadic_list_linear_type` in
`test_variadic.mojo`. The `^` on `args` in `consume_elements` call sites selects
the owned view of the variadic bundle.

### `consume_elements` and APIs that want `var` elements

Many APIs that sink a run of values use the same shape as `List`’s list-literal
constructor: `var *values: Self.T` plus `values^.consume_elements[...]` to move
each argument into freshly allocated storage.

### Printing and debugging

When `element_type` conforms to `Writable`, `VariadicList` implements
`write_to` / `write_repr_to` by looping and delegating to each element. That is
why the tests expect strings such as `(1, 2, 3)` or
`VariadicList[Int]((Int(1), Int(2), Int(3)))` for `write_to` versus
`write_repr_to`.

## Variadic packs

A "variadic pack" is the heterogeneous counterpart to `VariadicList`. Instead of
`*args: T` (one static element type), you pair a **type parameter pack** with an
**argument pack** whose types are taken from that pack:

```mojo
def callee[*Ts: Writable](*args: *Ts) raises:
    ...
```

The callee receives a `VariadicPack` (defined in
`stdlib/std/builtin/variadics.mojo`). Like `VariadicList`, it is
`RegisterPassable` and participates in ownership (`is_owned`, `var *args: *Ts`,
`consume_elements`, `__del__`), but its internal representation is a
heterogeneous value like a `Tuple`.

### Why packs need `comptime for`

Each argument slot can be a **different** concrete type with a different size
and ABI. The pack is therefore closer to a tuple than to an array: there is no
single `T` you could use for `args[runtime_idx]`. Indexing is exposed (using
`__getitem_param__[index]`) with a **compile-time** index, so the compiler can
emit the correct load for each position.

That is why idiomatic code walks packs with `comptime for`, not a runtime
`for i in range(len(args))` over `args[i]`:

```mojo
def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
    var total = 0

    comptime for i in range(args.__len__()):
        # Each `args[i]` has a different concrete type from `*ArgTypes`.
        total += Int(args[i])

    return total

def main():
    print(count_many_things(Int8(5), UInt32(11), Int(12)))  # 28
```

This example is adapted from the `VariadicPack` docstring in `variadics.mojo`.
The important point is that the loop variable `i` is a compile-time parameter,
so each `args[i]` is monomorphized separately.

### `Writable` packs and forwarding

When every element type conforms to `Writable`, the pack implements `write_to`
and related operations:

```mojo
def helper[*Ts: Writable](*args: *Ts) raises:
    var s = String()
    args.write_to(s)
    # For (1, "hello", True) -> "(1, hello, True)"

def forwarder[*Ts: Writable](*args: *Ts) raises:
    helper(*args)  # splat forwards the pack unchanged

# Caller:
forwarder(1, "hello", True)
```

The forwarding pattern `callee(*args)` is exercised in
`test_variadic_pack_forwarding` and the multi-hop variant
`test_variadic_pack_forwarding_through_two_levels` in
`stdlib/test/builtin/test_variadic.mojo`. Empty and single-element packs forward
the same way (`forwarder()` and `forwarder(42)` in the corresponding tests).

### `SomeTypeList` sugar

When you only need "any number of types, each conforming to `Trait`," you can
name the argument pack without introducing an explicit `*Ts` binding on the
`def`:

```mojo
def foo(*args: *SomeTypeList[Writable]) raises:
    var s = String()
    args.write_to(s)
```

`SomeTypeList` is a comptime alias (see `stdlib/std/builtin/anytype.mojo`) that
expands the idea of `Some[T: Trait]` to a whole `TypeList` constrained by the
same trait. It is particularly useful at variadic call sites.

### Relationship to `Tuple`

`Tuple` is the canonical struct that owns a heterogeneous sequence. Its
constructor takes a variadic pack aligned with its element-type list:

```mojo
# Conceptually (see `stdlib/std/builtin/tuple.mojo`):
struct Tuple[*element_types: Movable](...):
    def __init__(out self, var *args: *Self.element_types):
        ...
```

## Variadic keyword arguments

Variadic **keyword** arguments are the runtime counterpart to Python's
`**kwargs`. A callee can accept any number of extra keyword arguments whose
**values** all share one type `V`. Keys are always runtime `String`s (the
keyword names written at the call site). Unlike variadic packs, there is no
heterogeneous value-type list: `**kwargs: Int` means every passed value must be
an `Int`, not a mixed tuple of types.

The surface syntax is a leading `**` on the argument name, which must be at the
end of the signature after other arguments:

```mojo
def variadic_kwargs(a: Int, b: Int, *args: Int, c: Int, d: Int, **kwargs: Int):
    pass

def print_nicely(**kwargs: Int):
    for item in kwargs.items():
        print(item.key, "=", item.value)

def main():
    print_nicely(a=7, y=8)
```

Only one `**` parameter is allowed per function, and a type annotation is
required: for example `**kwargs: Int`, not bare `**kwargs`.

### The callee receives `OwnedKwargsDict`

The `**kwargs: V` syntax is passes as `var kwargs: OwnedKwargsDict[V]`. The type
behaves like a `Dict[String, V]` for most user-facing operations: `len`,
`in`, `__getitem__`, `keys()`, `values()`, `items()`, `find`, and `pop`.
Users should not construct it directly; the compiler builds it at call sites
and passes it into the callee.

Keyword variadics are passed as **owned** (`var`) arguments, because the
dictionary is typically built by for one call site. You cannot write an explicit
`read` or `mut` convention on them:

```mojo
# Not supported yet.
def borrowed_kwargs(mut **kwargs: Int): ...
```

Inside the callee, `kwargs` owns the dictionary and its inserted values. That
matches how call lowering transfers each keyword operand into the dict with
`_insert`.

### Call lowering

Rough lowering picture (names only, not exact MLIR):

```text
// Caller:  foo(x=9, stuff=8)   with   def foo(**kwargs: Int)

1. Allocate an empty OwnedKwargsDict[Int] (local temporary).

2. For each keyword operand at the call site:
   - Materialize the key as a compile-time String literal.
   - Evaluate the value expression.
   - Call OwnedKwargsDict::_insert(dict, key, value), transferring ownership
     of the value into the dictionary.

3. Pass the filled dict to foo as an owned **kwargs argument.
```

Overload resolution collects any keyword operands that do not bind to earlier
named arguments and routes them to the `kw_vararg` slot. If the callee has no
`**kwargs` argument, those operands are erroroneous. Positional `*args` and
`**kwargs` can appear in the same signature; each eats the operands of its kind.

### Forwarding with `**kwargs^`

To forward an entire keyword bundle to another variadic-kwargs callee, use the
double-star unpack form and transfer ownership with `^`:

```mojo
def pass_kwargs(**kwargs: Int):
    takes_int_variadic_kwargs_multiline(**kwargs^)
```

Without `^`, forwarding tries to copy the dict and fails because
`OwnedKwargsDict` is not implicitly copyable. With `^`, the caller's dict is
moved into the inner call. This is the keyword analogue of splatting a
variadic pack with `callee(*args)`.

Dictionary unpacking from an ordinary `Dict[String, V]` (for example
`print_nicely(**my_dict)`) is not supported yet; only forwarding from another
`**kwargs` binding works today.

### Generic inference

When a function is generic in the value type, keyword arguments can drive type
inference the same way positional arguments do:

```mojo
def infers_from_kwargs[T: SomeTrait](**kwargs: T):
    pass

# T is inferred as MemOnly from the keyword values.
infers_from_kwargs(y=MemOnly(), z=s)
```

The inferred element type must satisfy the stated trait bounds and must be
compatible with transfer into `OwnedKwargsDict::_insert` (owned, movable
values).

### Current limitations

The following gaps are intentional for the first release; see also the
[variadic keyword arguments](/docs/manual/functions#variadic-keyword-arguments)
section of the Mojo manual:

- Homogeneous values only: all keywords share one value type `V`.
- Keys are always `String`; there is no typed-key variant.
- Always owned: no `read **kwargs` or `mut **kwargs`.
- No unpacking from a plain `Dict` at call sites.
- No compile-time variadic keyword **parameters** (`**kwparams: ...` on
  declarations).
- Values must be `Movable` (and satisfy the `OwnedKwargsDict` constraints);
  non-owned linear values follow the same transfer rules as other owned call
  arguments.

Examples in `stdlib/test/collections/test_dict.mojo` (`test_owned_kwargs_dict`)
exercise the dict API surface that variadic kwargs expose inside callees.
