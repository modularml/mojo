# Mojo `deinit` Argument Convention [v2]

**Status**: Implemented.

This document explores what is hopefully the last argument naming convention
repainting that we need to figure out, finally bringing the whole system into a
consistent and easier to reason about set of semantics.

**TL;DR**: This introduces a new `deinit` argument convention, used by implicit
or named destructors, and removes the `__disable_del` syntax. This is important
to make the language more consistent and provide a path to linear types that
fits in with implicit destructors.

## Before we dive in: An Important Grounding Truth

The most important thing to understand going into this document is that a
“struct value” cannot be though of as just a sum of its parts. Consider this
very simple type:

```mojo
struct FileDescriptor:
    var value: Int

    def __del__(owned self):
        sys.close(self.value) # Call close(2)
```

A `FileDescriptor` is a struct that wraps an `Int`. Mojo as a language and
compiler doesn’t know what an `Int` is, but it does know it is a trivial type
(has no destructor and can be moved trivially). Note that `FileDescriptor` is
not trivial, even though it only has one trivial member. This is what we mean
by saying that the behavior of a struct (like `FileDescriptor`) can be very
different than the behavior of its members.

Let’s generalize this observation: While some familiar types (e.g. “a pair” or
“a dataclass”) should behave the same way as a composition of their elements,
this isn’t true in general. A struct can provide additional semantic behavior
on top of the storage they contain. This isn’t surprising: a `List` is far more
than an unsafe pointer, capacity and length!

## Background: How Argument Conventions work

Through a series of steps, we’ve converged the argument convention design in
Mojo to look like this:

```mojo
# Semantic references to values:
def f1(a: SomeType):     # Read only reference to a value.
def f2(mut a: SomeType): # Mutable reference to a value.
def f3(ref a: SomeType): # Reference with contextually inferred mutability

# Semanticly a mutable owned copy of a value, formerly known as "owned":
def f4(var a: SomeType):

# Syntax sugar for "def f5() -> SomeType:".
def f5(out a: SomeType):
```

This design has been through a lot of discussion, has been rolled out now and
seems to be
[well received](https://discord.com/channels/1087530497313357884/1224434323193594059/1390609085380034591)
so far, though it is early days.

While rolling out [the patch to mass adopt the new
spelling](https://github.com/modularml/modular/pull/64891/files), we had to
migrate the code base and hit upon some odd cases, consider core value methods:

```mojo
struct MyThing:
  ...
  def __init__(out self): ...
  def __copyinit__(out self, existing: Self): ...
  def __moveinit__(out self, owned existing: Self): ...
  def __del__(owned self): ...
```

Per the above decisions, the `owned` arguments to `__moveinit__` and `__del__`
should change to `var`.

I decided not to do this, because `def __del__(var self)` ”looked weird”, and it
reminded me is that Mojo still has two outlier cases which are special cased in
the compiler and therefore magic: the “source”/self of `moveinit` and `del`:
they do not work like normal `owned` arguments.

Let’s dig into the current behavior of these special cases to understand them,
identify some problems, then see of we can dispel some of the magic involved
here.

### Behavior of `var` / `owned` arguments and the `__del__`

The `__del__` special function in Mojo is a familiar friend: it implements the
destruction logic that tears down a value after its last use. Mojo decides when
a value has been “last used” and inserts the calls to destructors automatically,
using static garbage collection.

Let’s take a careful look inside a destructor to see how it works, because there
is an unfortunate bit of magic afoot. With top of tree Mojo, `__del__` and
`normal_method1` behave differently due to this magic.

```mojo
def consume_str(var data: String): ... # Consumes the string

struct MyStringWrapper:
    var str: String
    ...
    def __del__(var self): # same as "owned self"
        consume_str(self.str^) # this is ok.
        return # ok!

    def normal_method1(var self):
       use_normally(self.str)
       # Mojo calls self.__del__() for you implicitly here.
       return
```

In normal methods, a `var` argument is a uniquely owned value, and Mojo will
make sure that value is destroyed for you after its last use: e.g. in
`normal_method1` . Mojo is smart - if you consume the whole value inside the
function then it knows that the value is not alive, and will not run its
implicit destructor:

```mojo
extension MyStringWrapper:
    def normal_method2(var self):
       transfer_whole(self^) # transfer_whole consumed all of 'self'.
       return # ok, self.__del__() is NOT called.
```

However, take a look at what happens when a method consuming one member string
out of the value, but not the value as a whole. Mojo realizes that the *value
as a whole* isn’t destroyed (go back to the `FileDescriptor` example - consuming
a field doesn’t mean the whole value is destroyed) so when it tries to destroy
it, it produces a compile time error saying it can’t.

```mojo
extension MyStringWrapper:
    def normal_method3(var self): # same as "owned self"
        consume_str(self.str^)  # this is ok.
        return # Error, cannot run the dtor on self because "str" is missing.
```

Good job Mojo - the `__del__` on `MyStringWrapper` could contain arbitrary
side effects!

This raises the question though - why does `__del__` compile? It is taking
something that looks syntactically like an `owned` or `var` argument, but in
fact, it doesn’t (recursively) run the destructor on self at the end. The
answer is that `__del__` is special cased to disable the full object destruction
state of `self` before any returns, which makes it actually behave like this:

```mojo
extension MyStringWrapper:
    def normal_method4(var str: MyStringWrapper):
        consume_str(self.str^)  # this is ok.
        __disable_del str # disable the dtor
        return # ok!
```

This is what I mean when I say that `__del__` has a bit of implicit magic in it:
all `__del__` methods include an implicit `__disable_del` inside of them, that
are completely invisible in the code. This magic erodes the mental model and
hurts teachability of Mojo.

### Behavior of `existing` in a `__moveinit__`

Similarly to destructors, `__moveinit__` also has the same special case (and
this special case is limited to exactly these two methods): the `existing` value
also gets its destructor disabled. This is why this logic works for
`__moveinit__` but not other methods:

```mojo
struct StringPair:
    var a: String
    var b: String

    def __moveinit__(out self, owned existing: Self):
        self.a = existing.a^ # transfer out of existing
        self.b = existing.b^ # transfer out of existing
        return # Ok!

    def notmoveinit(owned existing: Self):
        consume(existing.a^)
        consume(existing.b^)
        return # error: cannot destroy "existing": a and b fields are uninit!
```

The issue here is the same: `__del__` and `__moveinit__` both have implicit
calls to `__disable_del` on their `owned`/`var` arguments. These are the only
two methods in mojo with this behavior.

It’s worth noting that this behavior is very natural to Mojo programmers in a
`__del__` method, but empirically people find this behavior surprising in a
`__moveinit__`, probably because this behavior is invisible and different than
C++. It is very common to see code like this, for example:

```mojo
struct MyArray:
  var ptr: UnsafePointer[...]
  var size: Int

  def __del__(owned self):
       free(self.ptr)

  def __moveinit__(out self, owned existing: Self):
      self.ptr = existing.ptr
      self.size = existing.size
      # This has no effect, del isn't called anyway
      existing.ptr = UnsafePointer()
```

The logical reason for this is that many Mojo programmers are coming from C++.
In C++, a `std::move` out of a value is really more of a hint than a comment:
all move operations that “steal” out of a value still have a destructor run on
the original value, so it must be reset into a no-op defensive state.

Mojo has moved the puck forward - if a value is being consumed, the destructor
won’t be run on it, and moveinit’s disable the destructor automatically. This
is a great feature that leads to more efficient and easier to implement logic,
but nonetheless, it is clear that normal Mojo programmers don’t understand this
behavior. Any why would they? This is completely magic behavior that only
applies to destructors and moveinit, an it is mostly invisible - until it is
surprising.

### The problem: Mojo should be orthogonal and explainable

While this behavior is explainable, it is complicated and non-obvious. This
leads to surprising behavior because there is nothing in the code that enables
programmers to understand what is going on. This is difficult to explain (e.g.
in documentation) and hurts scalability. This design has been useful to
bootstrap Mojo and get it off the ground, but if we are going to make a change,
it is better to make it sooner rather than later.

The problem here is that Mojo has multiple different ways to destroy a value:

1. The `__del__` member is obviously the normal way to do this.
2. The `__moveinit__` member typically consumes the existing value (but doesn’t
   have to).
3. Linear types (e.g. `Coroutine`) need to have *named* destructors and do not
   have implicit destructors.

The lack of regularity here leads to `__disable_del` needing to exist with its
magic syntax. This is a place holder that we would like to remove, pulling
together all this complexity with one simple solution.

## Proposal: Formalize `deinit` argument convention

The proposed solution is to introduce a new `deinit` argument convention and
remove `__disable_del`. This can be done in a few steps:

### Step 1: Formalize the destructor convention

In Mojo circa 25.4, one defines a destructor with the `owned` convention like
so, which embeds the `__disable_del` call:

```mojo
struct MyStruct:
    ...
    def __del__(owned self):
         consume(self.field^)
```

We propose making this “whole object is implicitly destroyed by the callee”
argument convention explicit by introducing a new argument convention, e.g.
`deinit` or `destroys` (other better names are welcome):

```mojo
struct MyStruct:
    ...
    def __del__(deinit self):
       consume(self.field^)
       # Ok, the destructor is not implicitly run
```

This argument convention works the same was as `var` on the caller side - it
demands a uniquely owned rvalue, and will attempt to copy the value on the
caller side if not owned. Such a convention can be applied to **any** function
though, not just destructors:

```mojo
struct MySpaceShip:

    def launch(deinit self):
        process(self.cargo)
        # del is not invoked here
```

By naming this convention, we get a much easier to explain behavior: we have a
(soft) keyword that encapsulates the behavior, and errors like this would be
diagnosed by the existing “recursive function call” error (though this would
definitely get a specialized error message):

```mojo
struct MyThing:
    def __del__(var self):
       use(self.state)
       return # error: recursive call to self.__del__() is an infinite loop, change "var" to "deinit"
```

The consequence of this is that the behavior of `__moveinit__` is now
explainable, and the user can even opt-in to whatever behavior they want. The
default recommendation would be:

```cpp
struct StringPair:
    var a: String
    var b: String

    def __moveinit__(out self, deinit existing: Self):
        self.a = existing.a^ # transfer out of existing
        self.b = existing.b^ # transfer out of existing
        return # Ok!
```

… but it would also be valid to define `__moveinit__` with a `var existing` if
you want the destructor to run.

### Step 2: Remove `__disable_del`

The `__disable_del` magic function was introduced into Mojo because we needed a
way for linear types to disable the destructor on `self` in a consuming method.
With Step 1 in place, they can just use the new `deinit` argument convention.
This shrinks the language, and makes the eventual roll-out of linear types
simpler because we can tie it back to the core idea of “you’re implementing a
named destructor”.

### Step 3: Restrict usage of the `deinit` convention

We aspire for Mojo to be a memory safe language, and we want types to control
their behavior through the operations they provide on the data they contain.
As such, we propose that `deinit` only be allowed on the `self` argument of a
method in a struct (note: `existing` in `__moveinit__` is a self argument,
because `out` is syntax sugar for a result).

The consequence of this is that code can’t randomly disable destructors on types
they don’t control, things like this will be invalid:

```cpp
struct Tuple[...]:
   def __init__(
        out self,
        var storage: VariadicPack[_, _, Copyable & Movable, *element_types],
    ):
        # ... move the elements out of the pack ...
        # Do not destroy the elements when 'storage' goes away.
        __disable_del storage
```

instead, we need to add a method to `VariadicPack` that does this (note: this
has been done, it is currently called `consume_elements`). This can be a good
way to up-level the APIs in general to being more safe. Mojo is getting
`struct` extensions soon, so this isn’t even a limitation in power.

## Exotic Examples

Here are a few exotic examples that came up in discussion. These aren’t
expected to be common, but explore some of the edge cases.

### The `self` argument of `__del__` need not be `deinit`

While it will be vastly the most common thing for the `self` argument of
`__del__` to be deinit, there is one case where you might want to define it as
`var`: when you delegate to another destructor:

```mojo
# This shows that there are cases where it is valid for __del__ to be
# var instead of deinit - when delegating destruction to a named destructor.
struct DelNeedNotBeDeinit:

   def __del__(var self):
       self^.custom_del(42, "foo")

   def custom_del(deinit self, x: Int, message: String):
       pass # custom logic is here.
```

This opens questions about how common errors are diagnosed. We believe that the
compiler has the right infrastructure to diagnose mistakes correctly, e.g.:

```mojo
# This is a flow-sensitive error: 'self' not being consumed in del leads
# to a recursive call to del, which is an error.
struct UserError2:
  def __del__(var self):
    use(self.state)
    pass       # CheckLifetimes generates an error "declare self as deinit"
```

We don't think there is any need to allow parametric or conditional deinit. If
you have something that is conditional deinit, declare it as 'var' and delegate
the deinit path to a named destructor:

```mojo
struct NoParametricDeinit:
   def __del__(var self):
      if some_cond():
        self^.custom_del(42, "foo")
      else:
         self^._actual_del()

   def custom_del(deinit self, x: Int, message: String):
       pass # whatever logic.

    def _actual_del(deinit self):
      pass # Stub that just drops all the elements.
```

### We can't accept `var` on `__del__` or `__moveinit__` for a while

While we want to support logic like the above for full generality, we can't do
that immediately. The problem is that all Mojo code currently declares their
`__del__` arguments with `owned` or `var` keywords and implicitly get `deinit`
behavior. It would be *extremely* hostile to 3rd party code to silently break
all destructors by changing behavior.

A better approach to phasing this in is to force the world to move to `__del__`
in the 25.6 release (generating a warning with a fixit hint), make `var` an
error in 25.7, and then start accepting the new behavior in a subsequent
release like 25.8.

### `deinit` is not part of the function type

Because `deinit` just changes the implementation details of a function body, it
is not part of its type, and therefore `deinit` isn't supported in function
types:

```mojo
struct Example:
    def deinit_method(deinit self): ...
    def var_method(var self): ...

def example():
    var fp1 : def(var Example) -> None
    # Both are ok.
    fp1 = Example.deinit_method
    fp1 = Example.var_method

    # error: 'deinit' not allowed in function type, use 'var' instead.
    var fp2 : def(deinit Example) -> None
```

### Other minor behaviors

These just show some minor behavior, but shouldn't be surprising:

```mojo
# This is a parser error: 'self' is required to be 'var' or 'deinit' in an
# implicit destructor.
struct UserError1:
  def __del__(self):   # Parser rejects.

# Deinit is ASAP destruction for all the members.
struct Thing:
  var state1: String
  var state2: String
  def custom_del(deinit self, var other_arg: String):
     # state2.__del__()  # Inserted by the compiler.

     use(other_arg)
     # other_arg.__del__() # inserted by the compiler.

     use(self.state1) # Keeps state1 alive
     # state1.__del__(); # inserted by the compiler.
```

## Alternatives considered

There are several different alternatives to consider in this space:

### Different argument convention names

The names above are somewhat arbitrary, and are soft keywords. We could
consider other names (suggestions welcome!).

### Just remove the magic

We could remove the del magic by requiring the use of `__disable_del` (and some
corresponding operation in initializers) explicitly. The good thing about this
is that it would make the behavior very explicit:

```mojo
struct String:
    ...
    def __del__(owned self):
        """Destroy the string data."""
        self._drop_ref()
        __disable_del self
```

Forgetting to write this would cause the Mojo compiler to complain about
recursive destruction of a value, so there is no lack of safety. That said, it
does seem like a lot of boiler plate for a common operation. This would also
raise the urgency of finding a good syntax for `__disable_del` , because that
spelling is just a placeholder. This also doesn’t help with encapsulation, and
make destructors unnecessarily verbose.

### Embrace the magic

One alternative is to just leave one or the other behavior alone. That has the
disadvantage of leaving a footgun in the language and needing to keep
`__disable_del` around, and for linear types to never be first class.
