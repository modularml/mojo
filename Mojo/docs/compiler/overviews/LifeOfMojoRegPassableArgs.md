# Life of Mojo reg-passable arguments

Mojo has the ability to represent something as `@register_passable` (shortened
to “RP” in this doc, the decorator is now deprecated in favor of
`RegisterPassable` trait). The former means that the value is **guaranteed** to
be passed in an MLIR “SSA register” (and thus, typically but not always, a
machine register) when passed to a function “imm” or “owned”, or when
**returned from a function** by value. In contrast, values that are not RP (and
all `ref` and `mut` arguments/results) are passed indirectly in memory.

> Note: Mojo has `@register_passable("trivial")`
(now deprecated in favor of TrivialRegisterPassable trait)
 as well, which isn't really
relevant to this doc. “Trivial” guarantees that the type has a trivial copy and
move constructor, which means that it can be copied or moved with memcpy. We
have long wanted to split the notion of “trivial” from RP because they're
orthogonal concepts, and because lots of algorithms can be specialized for
Trivial types (e.g. reallocation in List.append can use memcpy instead of a
loop). We just haven't gotten around to that. >

An example of an RP-trivial type is `Int` and an example of a memory type is
`String`. Let's look at how this work. When you write:

```jsx
def use_int(a: Int): pass
def use_str(a: String): pass

@export
def example():
  var an_int: Int
  var a_string: String

  use_int(an_int)
  use_str(a_string)
```

You get IR out of the parser like this (slightly simplified for readability):

```mlir
    lit.fn @example()  {
      // Note that the origin of these stack variables are explicitly named
      // *"an_int`" and *"a_string`1".
      %an_int = lit.var.decl : !lit.ref<!Int, mut *"an_int`">
      %a_string = lit.var.decl : !lit.ref<!String, mut *"a_string`1">
      // ...

      // Load the int out of memory into an SSA register
      %0 = lit.ref.load %an_int : <!Int, mut *"an_int`">
      // Pass the int value directly.
      %1 = lit.call @use_int(%0) : (!Int) -> !kgen.none

      // Convert the mutable reference to %a_string to an immutable one.
      %2 = lit.ref.immut %a_string : <!String, mut *"a_string`1">
      // Pass the immutable reference.
      %3 = lit.call @use_str[muttoimm *"a_string`1"](%2) : (!lit.ref<!String, imm *[0,0]> imm_mem) -> !kgen.none
```

You can get the full IR by using `kgen-translate -import-mojo x.mojo | less` .
Tip: search for `lit.fn` with the `/` command in `less` .

If you think in C++, you can imagine this is like this C++ code:

```c++
void use_int(size_t a);        // pass in a register
void use_str(const String &a); // pass by const reference.
```

The Mojo approach is superior to the C++ approach for a wide variety of reasons:

- This doesn't require the API author of “use_int” to think about whether to
pass by copy or reference or rvalue reference etc etc etc. Instead, the author
of `Int` makes a decision for all functions that pass and return that type, and
everything works.

- This is better for encapsulation, because clients aren't exposed to
  implementation of the type: if I change a struct to be smaller and decide it
  is beneficial to pass in a register, I can change one place, instead of
  everything that touches it.

- This makes Mojo easier to teach, because new folks don't encounter all these
  weird sigils all over the place. They can generally ignore RP until (and if)
  they get advanced enough to know about performance. When they do, they can
  have a single decorator that
  [can be googled and is documented](https://mojolang.org/docs/manual/decorators/register-passable/)
  rather than trying to figure out what
  [C++ `&&` means when in an argument list](https://learn.microsoft.com/en-us/cpp/cpp/rvalue-reference-declarator-amp-amp?view=msvc-170).

> Note: Early Mojo only had RP types - memory only types came later, and we have
> a range of legacy algorithms that still over-use `@register_passable`. The
> most egregious of which is `struct StaticTuple` which can be
> **arbitrarily large based on its `size` parameter**. This thing is an
> abomination 🧌 in modern Mojo and needs to be phased out.

Ok, now that we know how these things work and have seen some simple examples,
let me blow your mind a bit.

## Non-trivial Register Passable Types

I mentioned that “trivial” doesn't affect the discussion in this document, but
I **lied to you** to ease you into this! The truth is that trivial types are
all over the place, things like `Int` and `SIMD` , and we want the parser to
generate slim IR for simple things for readability and compile time reasons.

For non-trivial types, the parser needs to check for exclusivity violations
(which benefit form MLIR-level origins) and the CheckLifetimes pass has to
decide whether a value is used after it is transferred from and needs to insert
copy constructor and destructor calls.

To conserve compiler complexity, the Mojo **parser** emits the same IR for
`@register_passable` types memory types - that is, it passes them into
functions **by-address**, and relies on later compiler passes to remove this
abstraction. Let's use a look:

### Example RP Type

The Mojo standard library has RP types (e.g. `ArcPointer`), but we'll define
our own here to be self contained. The Mojo lowering process doesn't look into
functions, so we don't need implementations of any of the methods:

```mojo
struct MyRPType(Movable, Copyable, RegisterPassable):
    def __init__(out self): pass

    # Trivial types can't define a copyinit, but this is just RP.
    def __init__(out self, *, other: MyRPType): pass

    # RP Types cannot declare a moveinit, but they always implicitly have one
    # that is defined to transfer ownership.
    #def __init__(out self, *, deinit move: MyRPType): pass

    def __del__(deinit self): pass

# Functions that use MyRPType with different argument conventions.
def use_rp(a: MyRPType): pass # imm convention
def use_rp_type_mut(mut a: MyRPType): pass
def use_rp_type_owned(var a: MyRPType): pass
```

This is your minimal copyable RP type with stubbed out methods and some example
callers.

### How the parser codegen's non-trivial RP types

Let's look at an example:

```mojo
@export
def example2():
  var a_rp = MyRPType()

  use_rp(a_rp) # Pass by read

  use_rp_type_mut(a_rp) # pass by mut

  use_rp_type_owned(a_rp^) # pass ownership

```

This gets lowered into the following IR (simplified a bit again), with
`kgen-translate -import-mojo x.mojo | less` :

```mlir
// These are the lit-level signatures for the functions.  You can see that these
// are all passed as a !lit.ref (basically a pointer + origin).  The origin of
// the argument is an implicit parameter *"a`", and each of the arguments
// has a different MLIR-level convention: imm_mem, owned_in_mem, and mut.
lit.fn @use_rp[imm *"a`"](%a: !lit.ref<!MyRPType, imm *"a`"> imm_mem) {}
lit.fn @use_rp_type_mut[mut *"a`"](%a: !lit.ref<!MyRPType, mut *"a`"> mut) {}
lit.fn @use_rp_type_owned[mut *"a`"](%a: !lit.ref<!MyRPType, mut *"a`"> owned_in_mem) {}

lit.fn @example2():
  // Declare the variable on the stack, it has an origin *"a_rp`".
  %a_rp = lit.var.decl "a_rp" var : !lit.ref<!MyRPType, mut *"a_rp`">

  // Call the initializer, note that RP-nontrivial types are returned in
  // registers even at the parser level because we don't need exclusivity
  // checks and CheckLifetimes is cool with it, see also the copy ctor below.
  %0 = lit.call @MyRPType::@__init__()
  lit.ref.store %0, %a_rp

  // Pass by immutable reference for "imm".
  %1 = lit.ref.immut %a_rp : <!MyRPType, mut *"a_rp`">
  lit.call @use_rp[muttoimm *"a_rp`"](%1)

  // Passing by "mut" is easy, it obviously needs the address of the LValue
  // so the callee can mutate the result.
  lit.call @use_rp_type_mut[mut *"a_rp`"](%a_rp)

  // This is generated by the transfer operator, not important for this doc.
  lit.ownership.use %a_rp

  // Call the owned one, passing vardecl to be consumed directly.
  lit.call @use_rp_type_owned"[mut *"a_rp`"](%a_rp)
```

Ok, if you're looking at the IR, you must be confused. I told you that Mojo
guarantees that register_passable types are passed by register (except to `mut`
arguments), but that clearly isn't happening! We'll dive into that, but let's
go through the lowering pipeline a bit.

### Example after CheckLifetimes

I'm not talking about how CheckLifetimes checks for use of uninitialized
values, decides to optimize copies to moves and insert destructor calls in this
document, but it does, effectively right after the parser. You can see the
output of CheckLifetimes with: `kgen-translate -import-mojo x.mojo | kgen-opt
-verify-parameters -lower-semantic-cf -check-lifetimes | less`

In this case, it doesn't do much interesting because I've arranged the example
to not need CheckLifetimes to do anything. The one thing it does is insert a
destructor call into `use_rp_type_owned`, so it now looks like this:

```mlir
lit.fn @use_rp_type_owned[mut *"a`"](%a: !lit.ref<!MyRPType, mut *"a`"> owned_in_mem) {}
  lit.call @MyRPType::@__del__[mut *"a`"](%a)
}
```

Simple enough, it sees that the argument is not used, but must be consumed by
the function, so it inserts the destructor call to release it.

> 📖 Homework: Look at the example by removing the caret from
> `use_rp_type_owned(a_rp)` . You'll see a copy constructor inserted by the
> parser (because this form passes an lvalue for `a_pr` to an owned function.
> Then look at the CheckLifetimes output - it crushes the copyinit because it
> realizes it can transform it into a move (which is trivial for RP types).
> CheckLifetimes doesn't clean up the extra var decl, but it doesn't matter
> because the optimizer does later.

Ok, let's keep lowering, next up from LIT to KGEN.

### Example Lowered to KGEN

The LIT and KGEN level of representations are similar but have important
differences - things like methods are turned into a flattened form (and called
“generators” for historical reasons), origins are eliminated (so references
turn into raw pointers), and structs are flattened to their representation
instead of using a named symbolic representation.

Our example looks like this (just add the `-lower-lit` flag to the above
`kgen-opt` invocation:

```mlir
  // !lit.ref lowers to !kgen.pointer and origins are gone.  Our RP type is
  // flattened into its structural representation - in this case an empty
  // kgen struct because it has no members
  kgen.generator @use_rp(%arg0: !kgen.pointer<struct<()>> imm_mem) {
  }
  kgen.generator @use_rp_type_mut(%arg0: !kgen.pointer<struct<()>> mut) {
  }
  kgen.generator @use_rp_type_owned(%arg0: !kgen.pointer<struct<()>> owned_in_mem) {
    kgen.call @MyRPType::__del__(%arg0)
  }

  // Functions turn into kgen.generator.
  kgen.generator @example2() {
    // lit.var.decl turns into a stack allocation with explicit lifetime
    // markers.
    %0 = kgen.call @MyRPType::__init__()
    %1 = pop.stack_allocation 1 x struct<()> marked
    pop.stack_alloc.lifetime.start(%1)
    pop.store %0, %1

    %2 = kgen.call @use_rp(%1)
    %3 = kgen.call @use_rp_type_mut(%1)
    %4 = kgen.call @use_rp_type_owned(%1)
    pop.stack_alloc.lifetime.end(%1)

```

Without all the origins and other cruft in the IR, it is way easier to read!

This is the KGEN representation, but it has the same problem as before - these
are register passable types but they're passed indirectly. We're passing the
address of the stack allocation to the function.

What gives? Well we keep lowering. Next up is elaboration, which does nothing
in this example because the function isn't generic, and after elaboration, we
get the same IR, you can see this with

```text
kgen-opt -verify-parameters -lower-semantic-cf -check-lifetimes -verify-parameters -lower-lit -eliminate-dead-symbols -outline-closures -elaborate-generators`
```

> Note: I'm eliding unrelated optimization passes for clarity here.
>
> Helpful hint: Try passing `--mlir-print-ir-before-all` to `kgen-opt`.

### Lowering argument conventions

The next stop along our lowering journey is `-lower-arg-conventions` . We have
reached the hero of the story: try using: `kgen-translate -import-mojo x.mojo |
kgen-opt -verify-parameters -lower-semantic-cf -check-lifetimes
-verify-parameters -lower-lit -elaborate-generators -lower-arg-conventions |
less` .

With this, we get something interesting, let's look at the noop functions first:

```mlir
  kgen.func @use_rp(%arg0: !kgen.struct<()>) {
    %0 = pop.stack_allocation 1 x struct<()>
    pop.store %arg0, %0 : !kgen.pointer<struct<()>>
  }
  kgen.func @use_rp_type_mut(%arg0: !kgen.pointer<struct<()>> mut) {
  }
```

Suddenly things got a lot more interesting! The `-lower-arg-conventions` pass
did our job for us! It noticed that the argument to `use_rp` and
`use_rp_type_owned` are both RP types, and that the argument convention is
“imm” and “owned” respectively, so it transformed the signature of the
function to use the struct in a register (notice that there is no pointer) but
it left the “mut” function alone (because the body could mutate the value, so
it needs to be passed by address).

Of course, the BODY of these functions can be doing anything with the pointer -
it need not be simple loads and stores. For example, the owned version might
pass the owned pointer as a mutable argument to another function:

```mojo
@export
def other_owned(var a: MyRPType):
  use(a)
  use_rp_type_mut(a) # uses a pointer to mutate.
  use(a)
```

It handles this by materializing a local stack allocation in the function body,
storing the register into it, and then “replace all use with” (an efficient and
trivial rewrite using SSA def-use chains) the local value. In the case of
`use_rp` , the value wasn't used, so we just get a local stack temporary and a
store of the register value into it. This is pointless, but is guaranteed to be
removed by the optimizer, so we don't care.

Let's look at the function with the destructor call:

```mlir
  kgen.func @use_rp_type_owned(%arg0: !kgen.struct<()> owned) {
    // Set of the stack temporary just like use_rp
    %0 = pop.stack_allocation 1 x struct<()>
    pop.store %arg0, %0

    // Pass the VALUE to the destructor.
    %1 = pop.load %0
    kgen.call @MyRPType::__del__(%1)
  }
```

Remember that this function had a call to the destructor. If you scroll
above you'll notice that it is passing the address down to the destructor.
This is because the destructor function also "owns" the value it is passed
(`deinit`), and before this pass, owned non-trivial types are passed by memory.

The code above is different though - it loads the temporary out of the stack
allocation and passes the VALUE to the destructor. Why is this? Well, the
signature for the destructor itself also had the address removed, and needs the
value. Changing the signature of a function requires updating both the caller
and the callee sides to match:

- on the caller side we **introduce a load** to turn the address into a value

- on the callee side we **introduce a store to a temporary** to materialize an
  address

This makes the lowering logic all local.

<aside>
💡 Key observation: the composition of these two simple steps has a
desirous property: it (nearly) guarantees that the stack temporaries involved
are completely local and trivially promotable within registers by SROA. This
allows us to do very strong guaranteed optimizations that C++ doesn't allow -
C++ ends up pinning tons of temporaries to the stack because of extraneous
registers.

I'm careful to say “nearly” here because we have one unimplemented feature: we
don't promote `mut` values to be passed by register. This is completely doable,
but we haven't gotten around to doing it - we'd just pass it in as a result,
and then pass it back out as an extra result. KGEN has full support for
multiple results, so that is no problem, but it would probably make LLDB sad.
No one has raised it as a performance issue so we never explored this, it is
something to look at in the future.
</aside>

This is pretty cool, lets look at what it does to our `example2` function. The
same local transformation is applied to it as well, and it now looks like this:

```mlir
 kgen.func @example2() {
    %0 = kgen.call @MyRPType::__init__()
    %1 = pop.stack_allocation 1 x struct<()> marked
    pop.stack_alloc.lifetime.start(%1)
    pop.store %0, %1
    %2 = pop.load %1
    %3 = kgen.call @use_rp(%2)
    %4 = kgen.call @use_rp_type_mut(%1)
    %5 = pop.load %1
    %6 = kgen.call @use_rp_type_owned(%5)
    pop.stack_alloc.lifetime.end(%1)
  }
```

The nice thing about this is that we're doing direct loads and stores to the
stack allocation: if we didn't have `use_rp_type_mut`, we would be able to
trivially promote this to a register with SROA/Mem2Reg and we're good. This is
a super power of Mojo 🔥!

## Parametric Generics Functions Work the Same Way

All the above is a really trivial example, but the benefits of this approach
stack and generalize nicely, let's look at generic functions. I'll focus on the
“imm” convention just to shorten this doc, but please try changing these
around and looking at the IR to see that `var` and `mut` ”just work” as well
with the correct semantics.

```mojo
# Works with any kind of type: which trait it conforms to is not
# important for the example. We use Copyable here.
def use_generic_type[T: Copyable](a: T): pass

@export
def example3():
  var my_str = String()
  var my_rp = MyRPType()
  var my_int = Int()

  use_generic_type(my_str)

  use_generic_type(my_rp)

  use_generic_type(my_int)
```

 Note that I'm defining a generic function and using it with a memory type
 (`String`), a RP type (`MyRPType`) and a RP-Trivial type (`Int`). The generic
 function is lowered into one representation by the parser, and so the parser
 needs to use the most general form, a memory representation. You can see this
 coming out of the parser or after CheckLifetimes, it has a signature that
 looks like this:

```mlir
 lit.fn @use_generic_type<T: !Copyable>[imm *"a`"]
     (%a: !lit.ref<:!Copyable T, imm *"a`"> imm_mem)
```

Let's break this down:

- This function is simplified a bit because the actual IR dump uses the ugly
  mangled name, not the simple “use_generic_type” name. In an IR dump it looks
  like `lit.fn @"use_generic_type[[::Copyable]($0)]($0)"<...`
  but we ignore that.

- You see the `T` parameter and its type bound `!Copyable`, the trait it
  conforms to.

- You see the ``*"a`"`` origin that the argument is passed with.

- You can see the `%a` argument is passed as
  ``!lit.ref<:!Copyable T, imm *"a`"> imm_mem`` which you can read as “a
  reference to value of type T, with origin ``*”a`”`` and passed `imm_mem`
  argument convention.

We don't dive into the body, but you can try checking the example around to do
stuff and you'll see that no matter what the call sites, generic functions
treat their argument as memory types. You can try your own trait, and you can
try making the trait `@register_passable` itself too. As you'd expect, the body
of the function must pass values by reference just like the RP example shown
above.

### Parametric function example after elaboration

Elaboration is the pass that uses parametric functions and instantiates them to
specialize them. In this case, we have a parametric function `use_generic_type`
and three types:

```bash
kgen-translate -import-mojo x.mojo | kgen-opt -verify-parameters -lower-semantic-cf -check-lifetimes -verify-parameters > x.mlir
kgen-opt -lower-lit -eliminate-dead-symbols -outline-closures -elaborate-generators x.mlir | less
```

We get something like this:

```mlir
  kgen.func @"use_generic_type,T=Int"(%arg0: !kgen.pointer<index> imm_mem){
  }
  kgen.func @"use_generic_type,T=String"(%arg0: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>> imm_mem) -> !kgen.none {
  }
  kgen.func @"use_generic_type,T=MyRPType"(%arg0: !kgen.pointer<struct<()>> imm_mem) -> !kgen.none {
  }
```

Here we can see that each of these functions got specialized (i.e.
instantiated) with a type as we would expect. This substitutes out the `T`
parameter, and we name mangle the type substitution into the function name to
keep the names unique.

This change doesn't affect the body of `example3`, other than we now call the
concrete instantiations of the function instead of calling the generic one,
e.g.:

```mlir
    kgen.call @"use_generic_type, T="(%0) : (!kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>> imm_mem) -> !kgen.none
```

Note in the IR dump that we treat RP-Trivial types and RP types and memory
types all the same here. The elaborator has no special knowledge of them.

### Parametric function example after argument lowering

After argument lowering we finally get to what we expect, I'll just show the
callee side again:

```mlir
  kgen.func @"use_generic_type,T=Int"(%arg0: index) {
    %0 = pop.stack_allocation 1 x index
    pop.store %arg0, %0 : !kgen.pointer<index>
  }
  kgen.func @"use_generic_type,T=String"(%arg0: !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>> imm_mem) {
  }
  kgen.func @"use_generic_type,T=MyRPType"(%arg0: struct<()>) {
    %0 = pop.stack_allocation 1 x struct<()>
    pop.store %arg0, %0 : !kgen.pointer<struct<()>>
  }
```

This is very fancy and good - we had the parser generate a fully generic
representation which keeps intermediate optimization passes and the library
simple, then allowed the elaborator the instantiate things in a context
insensitive way, then have a single pass (LowerArgConventions.cpp, which is
<750 lines of C++ code) use information that is only available after
elaboration to adjust the calling convention of the functions.

## Consequences of this design

While I'm not aware of any other compiler that works this way — which doesn't
mean there aren't any, I'm just not aware of any, this was probably invented 50
years ago in LISP — I find this design to be simple and beautiful. Things I
like about it:

1. We get separation of concerns - a number of relatively simple algorithms
   that are able to focus on what they are doing: the elaborator instantiates
   things mechanically, the parser splats out code (though it does have a
   special case for RP-Trivial types to reduce IR bloat), the arg convention
   lowering pass does its thing.

2. We get optimized code for RP cases even when generics are involved. This is
   a guaranteed transformation, which is 100% reliable!

3. We get a simpler user model without having to micro-optimize all the
   libraries, or use SFINAE to optimize important specific cases.

## Heterogenous Variadic “Packs”

Variadic “packs” store a potentially heterogenous set of values that are passed
to a function. As with generic types, they are always represented in a generic
form when coming out of the parser: `VariadicPack` is represented as an array
of pointers to the elements of the pack. That array is formed on the caller
side and then passed down to the callee. Let's look at an example:

```mojo
def example4[*Ts: AnyType](*args: *Ts):
  ... use args ...

@export
def call_example4():
   var my_str = String()
   var my_rp = MyRPType()
   example4(my_str, my_rp)
```

The compiler interprets this variadic pack syntax as saying that `example4`
takes a `VariadicPack` instance as an argument directly. It is as if you wrote
the callee like this:

```mojo
def example4[*Ts: AnyType](args:
    VariadicPack[elt_is_mutable=False, origin=_,
                 element_trait=AnyType, element_types=Ts]):
```

You can go ahead and look at the declaration of `struct VariadicPack` to see
how this works. Before we go any further, notice that variadic packs work with
all argument conventions, including `var` and `mut` and `ref` , this is just
showing the `imm` convention one for simplicity.

Coming out of the parser, the signature ends up looking like this:

```mlir
    lit.fn @example4
      # Declared parameter Ts
      <Ts: param_list<!AnyType> pos_vararg>
      # Implicit origin: one for the `origin` param of the pack and one for the
      # pack itself.
      [imm *"args`", imm *"args`1"]
      # args itself.
      (%args: !lit.ref<
          @VariadicPack<
             :!Bool {:i1 0},
             :@Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = *"args`"},
             :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> Ts>, imm *"args`1"
          > imm_mem|pack) {
```

Ok, that's a lot to take in, but the comments above break it down: we have a
`Ts` parameter for the type list. We have two implicit origins: one for the
`origin=_` parameter of the pack, which is the union of all the origins passed
in by the callee, and one for the VariadicPack itself. Why is the
`VariadicPack` itself passed by reference? The answer is in the sections above:
it is a non-trivial `@register_passable` type (because the `var` form needs a
destructor to destroy the elements).

Ok, what does that look like on the caller side? Let's see how `call_example4`
ends up lowering:

```mlir
    lit.fn @call_example4() {
      %my_str = lit.var.decl "my_str" var : !lit.ref<!String, mut *"my_str`">
      %my_rp = lit.var.decl "my_rp" var : !lit.ref<!MyRPType, mut *"my_rp`1">
      // ... init %my_str and %my_rp ...

      // Convert the mutable origins for the two variables into immutable origins,
      // because this is a pack of "imm" values.
      %2 = lit.ref.immut %my_str : <!String, mut *"my_str`">
      %3 = lit.ref.immut %my_rp : <!MyRPType, mut *"my_rp`1">
      // Rebind the references to a common (unioned) origin of the two origins.
      %4 = kgen.rebind %2 : !lit.ref<!String, muttoimm *"my_str`">
         to !lit.ref<!String, imm {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}>
      %5 = kgen.rebind %3 : !lit.ref<!MyRPType, muttoimm *"my_rp`1">
         to !lit.ref<!MyRPType, imm {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}>
      // Create a lit.ref.pack with the two !lit.ref references.
      %6 = lit.ref.pack.create(%4, %5) : !lit.ref.pack<:param_list<!AnyType> [#String1, #MyRPType1], imm {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}>
      %7 = kgen.param.constant: !Bool = <{:i1 0}>
      // Create an instance of the VariadicPack with __init__.
      %8 = lit.call @VariadicPack::@__init__<:!Bool {:i1 0}, :Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> [#String1, #MyRPType1]>(%6, %7) : !lit.generator<("value": !lit.ref.pack<:param_list<!AnyType> [#String1, #MyRPType1], imm {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}>, "is_owned": !Bool) -> !lit.struct<#VariadicPack <:!Bool {:i1 0}, :@std::@builtin::@type_aliases::@Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> [#String1, #MyRPType1]>>>
      // We need to pass this by-ref into the callee, so create a stack temp.
      // This will be eliminated when argument lowering turns this into a
      // register passable thing.
      %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<@VariadicPack<...
      lit.ref.store %8, %anonymous2A : <@VariadicPack<:!Bool {:i1 0}, :@Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> [#String1, #MyRPType1]>, mut *"anonymous*`2">

      // Pass the temporary as immutable to the callee.
      %9 = lit.ref.immut %anonymous2A : <@std::@builtin::@list_literal::@VariadicPack<:!Bool {:i1 0}, :@std::@builtin::@type_aliases::@Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")}}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> [#String1, #MyRPType1]>, mut *"anonymous*`2">
      %10 = lit.call @example4[
         # Bind the implicit origins for the callee.
         imm {(mutcast mut *"my_rp`1"), (mutcast mut *"my_str`")},
         muttoimm *"anonymous*`2"]<:param_list<!AnyType> [#String1, #MyRPType1
       ]>(%9)

```

I consider this to be in very strong shape. Coming out of elaboration, we can
see that the function gets specialized with the type list, and all the origins
are gone, which makes it easier to read:

```mlir
  kgen.func export @call_example4() -> !kgen.none {
    %0 = pop.stack_allocation 1 x struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly> marked
    pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>>
    %1 = kgen.call @"std::collections::string::string::String::__init__()"(%0) : (!kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>> byref_result) -> !kgen.none
    %2 = pop.stack_allocation 1 x struct<()> marked
    %3 = kgen.call @"y::MyRPType::__init__()"() : () -> !kgen.struct<()>
    pop.stack_alloc.lifetime.start(%2) : !kgen.pointer<struct<()>>
    pop.store %3, %2 : !kgen.pointer<struct<()>>
    %4 = kgen.pack.create(%0, %2) : !kgen.pack<[#type_value4, #type_value5]>
    %5 = kgen.param.constant: i1 = <0>
    %6 = kgen.call @VariadicPack::__init__(%4, %5) : (!kgen.pack<[#type_value4, #type_value5]>, i1) -> !kgen.struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)>
    %7 = pop.stack_allocation 1 x struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)> marked
    pop.stack_alloc.lifetime.start(%7) : !kgen.pointer<struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)>>
    pop.store %6, %7 : !kgen.pointer<struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)>>
    %8 = kgen.call @example4(%7) : (!kgen.pointer<struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)>> imm_mem) -> !kgen.none
    %9 = kgen.call @String::__del__(%0) : (!kgen.pointer<struct<(struct<(pointer<none>, index, index) memoryOnly>) memoryOnly>> owned_in_mem) -> !kgen.none
    %10 = kgen.call @"y::MyRPType::__del__(y::MyRPType)"(%2) : (!kgen.pointer<struct<()>> owned_in_mem) -> !kgen.none
    %11 = kgen.call @VariadicPack::__del__(%7) : (!kgen.pointer<struct<(!kgen.pack<[#type_value4, #type_value5]>, i1)>> owned_in_mem) -> !kgen.none
    pop.stack_alloc.lifetime.end(%7)
```

Lower Argument Conventions then lowers this to get rid of the pointer around
the VariadicPack:

TODO: Fill in IR when the issues below are fixed.

This is the correct representation for the `VariadicPack` form - the pack
itself should be promoted to an “actually register passed” value by argument
lowering (eliminating the need for `%anonymous2A` on the caller side), and
LIT→KGEN lowering should expose the `!kgen.pack` (the consequence of lowering
the `!lit.ref.pack`), into a number of “pointers to elements”

TOWRITE: The remaining challenge is lowering this to expose the argument
pointers. We need to:

1. Get rid of the ‘owned' field in `VariadicPack`, replacing it with a param.
2. Change ArgumentLowering to be iterative

I did more work on heterogenous variadic packs and they are in a good place -
we only have a single type `VariadicPack` and use it consistently (this was
hard won!). It works with trait bounds, it works with all argument conventions,
it is all quite nice. It builds on `!lit.ref.pack` and lowers to `kgen.pack` as
Steffi mentions. The one missing piece is that LowerArgConventions doesn't
lower it.

I maintain this should be simple to add with the same rewrite we're using for
non-pack stuff.

⇒

```mlir
kgen.generator @ex17(%args: !kgen.pack<[Int*, Float*, Bool*]>) {
  ...
  %0 = kgen.struct.load_indirect %args

  kgen.call @Tuple(%args)

```

## Homogenous Variadics

Ok, this is all very exciting and nice. Yay for simple things that compose,
let's hill climb a bit - how does this work for homogenous variadics? Before
the we answer the question, remember that homogenous variadics are not
instantiated with the argument list: their model is that all the elements are
formed on the stack, and a pointer+length is passed down to the function. This
avoids code duplication, and it means that we're structurally passing an array.

TOWRITE: Explain more, halp plz!

```mojo
def use_generic_hvariadic[T: Copyable](*a: T):  pass

@export
def example4():
  var my_str1 = String()
  var my_str2 = String()
  use_generic_hvariadic(my_str1, my_str2)
```

This turns into `VariadicList` and passes the addresses of my_str1 and
my_str2 in as a list

### Homogenous variadics are half transitioned

Well here we get to some unpaved roads, the paver (me) got tired and wandered
on to different things that were more important and interesting. I'll point out
some problems.

1. Generic `VariadicList` is not transformed into a better form when the
   elements are RP, they're always an array of pointers to the elements. This
   sucks.

2. Because we're missing this, the parser notices variadics of RPTrivial type
   and uses `VariadicList` for them to reduce IR bloat. This sucks, we should
   get rid of this and merge these things someday.

```mojo

def use_generic_hvariadic[T: Copyable](*a: T):  pass
def use_int_hvariadic(*a: Int):  pass

@export
def example4():
  var my_int = Int()
  use_int_hvariadic(my_int, my_int)
  use_generic_hvariadic(my_int)
```
