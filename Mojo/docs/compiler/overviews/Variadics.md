# Overview

Highly recommend reading the
[variadics public docs](https://mojolang.org/docs/manual/functions/#variadic-arguments)
first.

TL;DR:

- `VariadicList`: basically a run-time array of a single register-passable type,
  all elements are next to each other contiguously.
  - Example: `def foo(*args: Int): ...`
- `VariadicList`: a run-time array of pointers. Like `VariadicList`, but the
  pointees can be memory types too.
  - Example: `def foo(*args: Spaceship): ...`
- `VariadicPack`: basically a tuple of pointers, pointing to a heterogeneous
  bunch of objects sharing a common trait.
  - Example: `def foo[*arg_types: Stringable](*args: *arg_types): ...`

One note: it may seem inefficient to do everything by indirection: if I pass an
Int and a Float into a variadic pack as “read” values, then I don’t want them
indirected. This is handled by argument convention lowering after elaboration,
so everything ends up nice and efficient at runtime (unlike C++).

## Difference between VariadicList and VariadicPack

Conceptually:

- VariadicList is an **array** of homogenous elements (either register-passable
values in the case of VariadicList, or pointers in the case of VariadicListMem).
- VariadicPack is a heterogeneous **tuple** of values (which are always
  indirect, by pointer).

There are a few main differences:

1. When we know their lengths.
   - VariadicList’s length is a
     [run-time value](https://www.notion.so/Run-time-Values-Can-Also-Be-Compile-Time-Values-15b1044d37bb80cea0c5c8e3f3f8ae95?pvs=21)
     - unless **the list itself** is a parameter, at which point the length is
       known at comptime, because all the elements are too.
   - VariadicPack’s length is known at compile-time (and also run-time),
     because the elaborator instantiates the function with each of the concrete
     list of types for the caller (because the list of types it itself a
     parameter).
2. Whether they’re homogenous vs heterogeneous:
   - Post-elaboration:
     - VariadicList is homogenous post-elaboration. It’s an array at run-time.
     - VariadicPack is heterogeneous post-elaboration. It’s a tuple of
       pointers at run-time.
   - Pre-elaboration:
     - VariadicList is homogeneous.
     - VariadicPack is *kind of* homogeneous depending on how you think of
       it. All elements share a common trait, so they’re homogeneous in that
       way. But conceptually they’re pointing at all sorts of objects that
       can be different types. It makes sense if you don’t think about it.™
3. How we iterate over them.
   - For VariadicList, it’s easy, just use a regular for-loop.
   - For VariadicPack, since all the types might be different at run-time, one
     must use `comptime for` or `pack.each` or `pack.each_idx`.
4. How we can index into them.
    1. For VariadicList, we can index into them at compile-time or run-time
       easily because it’s basically just a run-time array.
    2. For VariadicPack, we can only index into them at compile-time, because
       it’s basically just a tuple.

## Denizens Involved

Important high-level points:

- `!kgen.pack` is a heterogeneous list of elements, and holds them next to each
  other in memory. It's similar to a std::tuple (in fact, our Tuple is just a
  wrapper around a `!kgen.pack`). The parser doesn't use `!kgen.pack` directly,
  it uses a `!lit.ref.pack` instead.
- `!lit.ref.pack` is a heterogeneous list of pointers, and holds the pointers
  next to each other in memory. It's similar to a `std::tuple` of pointers.
  `VariadicPack` wraps this, it’s useful for varargs.
- There is no `lit.pack`, it would be redundant with `kgen.pack`.
- `kgen.variadic` is a homogeneous list of elements. For example,
  `kgen.variadic<Int32>` is a list of ints.
  - It can also be a homogenous list of a certain trait, like
    `kgen.variadic<AnyType>`, which can, **after elaboration,** become a
    heterogeneous list, but one could still think of `kgen.variadic` as
    homogenous.
- `Tuple` wraps `kgen.pack`.
- `VariadicPack` wraps `lit.ref.pack` (which is a bunch of pointers) of various
  types described by a compile-time `kgen.variadic` of types.
  - To emphasize, `VariadicPack` **does not** contain a run-time `kgen.variadic`
    of elements.
- `VariadicList` wraps `kgen.variadic`, and is only usable with trivial types.
  We should deprecate this in favor of `VariadicListMem`.
- `VariadicListMem` also wraps `kgen.variadic`, but it can be used with memory
  types too (because it tracks origins).

`kgen.variadic` appears in there a lot, in seemingly mysterious ways, so let’s
clarify its role:

- It’s basically an array, so think of it as an array.
- `VariadicList`/`VariadicListMem` contains a `kgen.variadic` (array) of
  **run-time** values.
- `VariadicPack` contains a `lit.ref.pack` (tuple of pointers) of run-time
  values, but uses a `kgen.variadic` (array) **of types at compile-time** to
  describe their types.

In other words, `VariadicList` uses a `kgen.variadic` of values at run-time,
`VariadicPack` uses a `kgen.variadic` of types at compile-time (to describe the
types of a run-time value `lit.ref.pack`).

## Loading from a `VariadicPack` into an SSA Value

There are two ways to read a `kgen.pack`:

- Load the entire thing at the same time, like
  `VariadicPack.get_loaded_kgen_pack`.
- Get a pointer to a particular element and load that pointer, such as
  `Tuple.__getitem__`.

We generally avoid the first.

We almost *always* do the second.

The only time it’s useful to load an entire `kgen.pack` into an SSA value is
when we’re dealing with FFI, such as handing a bunch of elements to `printf` or
something similar. This is because this operation is undefined if the element
types are memory-only types. Memory-only types cannot be loaded into an SSA
register.

On top of that, you can’t even do the first unless all the elements are
register-passable; you can’t “load” a memory type into an SSA register.
Therefore, this gets used in specific hack places like calling C vararg
functions.

## Variadic List Element Type Can’t Be a Trait

…because the type of a normal argument can’t be a trait type. We don’t support
existentials, which rely on dynamic dispatch and behavior.

Though, `VariadicPack`’s element type can be a trait. `VariadicPack` can be
heterogeneous post-elaboration.

## Variadics Are Run-time Values

In other words
[(barring comptime/interpreter/optimizers/etc)](https://www.notion.so/Run-time-Values-Can-Also-Be-Compile-Time-Values-15b1044d37bb80cea0c5c8e3f3f8ae95?pvs=21),
they have an address at run-time.

A VariadicList (which wraps `kgen.variadic`), is conceptually similar to Java’s
variadics, where the arguments are gathered into an array of pointers. Relevant:
see
[VariadicListMem docs](https://mojolang.org/docs/std/builtin/builtin_list/VariadicListMem)
how it accepts its index as a runtime argument… only possible if there’s an
array at run-time.

A VariadicPack (which wraps `lit.ref.pack`) is exactly the same, but it lowers
to a struct at run-time, and its `__get__`’s index
[is a generic parameter](https://mojolang.org/docs/std/builtin/builtin_list/VariadicPack).

VariadicLists are like this because we (intentionally) do not instantiate on all
the values passed in via VariadicList.

## Difference Between kgen.pack and kgen.struct

`kgen.pack` is more general than `kgen.struct`.

`kgen.struct` requires a known list of element types, `kgen.pack` allows that
list to be parametric; `kgen.struct` requires something like [Int, F32], but
pack also allows "elements".

`kgen.pack` is parameterized, it takes a TypedAttr as a parameter for its
contents. In the case of `Tuple`, that contents parameter is a `kgen.variadic`.

## VariadicPack’s Element Type Can Only Be A Trait

Looking at VariadicPack’s definition:

```python
alias _AnyTypeMetaType = type_of(AnyType)

struct VariadicPack[
    elt_is_mutable: Bool, //,
    origin: Origin[elt_is_mutable],
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
](RegisterPassable): ...
```

that `_AnyTypeMetaType = type_of(AnyType)` means that `element_trait` can be
any trait. It can’t be e.g. a struct or an int.

## The Asterisk

```mojo
def bar(*vlist: Int):
    ...

def foo[*arg_types: MyTrait](*vpack: *arg_types):
    ...
```

The asterisk means three different things here.

`*vlist: Int` makes a `VariadicList[Int]`.

`*arg_types: MyTrait` makes a `kgen.variadic` (which is basically an array) of
types, each which conforms to `MyTrait`.

`*args: *arg_types` makes a `VariadicPack` whose element types are described by
the `arg_types` `kgen.variadic`, and contains a `kgen.pack` containing values of
those types.

## Variadic-parameterized, variadic-referring, and variadic-capturing Functions (PPPRPCF)

Note the difference between `foo` and `zork` here:

```mojo
def variadic_parameterized[*arg_types: MyTrait](*args: *arg_types):
    ...

struct VariadicStruct[*arg_types: MyTrait]:
    def variadic_referring(*args: *arg_types):
        ...

def variadic_capturing[
    *arg_types: MyTrait, //,
    func: def (*args: *arg_types) -> Int,
]() -> VariadicStruct[*arg_types]:
    ...
```

`variadic_parameterized` is a **variadic-parameterized** function. It can take
in *any* arguments. For example, it can take in:

- An `Int` and a `Bool`
- Six `Spaceship`s
- Nothing!
- Seventy-three `TurtleDoves`, like in that Christmas song.

`variadic_referring` is a **variadic-referring** function. It can't just take in
anything, it can *only* take in exactly the same things that we gave to
`VariadicStruct`.

`variadic_capturing` is a **variadic-capturing** function. It can take in any
function parameter-value, whose arguments are read into the `arg_types`
parameter decl.

More on variadic-capturing functions in IFAIAV.

"Variadic function" is often ambiguous between these three things. Before diving
into anything with variadic functions, be clear on the difference between these
three.

## Inferring Function Args Into A Variadic (IFAIAV)

A function can read an incoming `fn` parameter-value's argument types and infer
it into a `kgen.variadic` parameter-decl.

Example (from test CAIASV):

```mojo
def device_func(a: Int, b: Bool) -> Int:
    ...

@value
struct DeviceFunction[*arg_types: MyTrait]:
    def call(self, *args: *arg_types) -> Int:
        ...

def infer_variadic[
    *arg_types: MyTrait, //, # paraphrased
    func: def (*args: *arg_types) -> Int,
]() -> DeviceFunction[*arg_types]:
    return DeviceFunction[*arg_types]()

def main():
    var thing = infer_variadic[device_func]()
    var result1 = thing.call(42, True)
```

Note that `infer_variadic` is not a variadic-parameterized function (see
PPPRPCF), it's a variadic-capturing function.

Here, we're doing a few things:

1. `main` is giving a `def(Int,Bool)` as a parameter-value to
   `infer_variadic`'s input-parameter `func` which expects something of type
   `def(*:*arg_types)`.
2. In doing so, the `infer_variadic` callsite is inferring that `arg_types` =
   `variadic(Int,Bool)`.
3. The callsite then knows that it should return a
   `DeviceFunction[variadic(Int,Bool)]`.

As seen on the `thing.call` line, this lets us enforce that the user's arguments
to `call` are the correct types that `device_func` takes in.

This is sometimes referred to as the "function types" feature.
