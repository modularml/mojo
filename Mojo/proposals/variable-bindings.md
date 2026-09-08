# Variable Bindings in Mojo

Updated June 2025. Status: Implemented in Mojo 25.4

This document provides an overview of how variable bindings work in Mojo,
explaining the design points, how the pieces compose together, and some
alternatives considered. This is considered to be an engineering design
document, not a formal part of the Mojo user documentation.

## Variable bindings in Python

Before we talk about Mojo, let's understand first how variable bindings work
in Python. Within a function, Python has no scopes, all variables are
implicitly declared on first use, and everything is dynamically checked. This is
why you see things like this work:

```python
def example():
  if ...:
     x = 4
  else
     x = 7
  use(x)
  use(y) # raises an error dynamically.

  for i in [1, 2]: pass
  print(i) # prints 2, which is the last value of 2.
```

Python doesn't have static types (static analysis tools like mypy are separate
from Python), so we might say that all values in Python implicitly have type
`PythonObject`.

Python supports a
[small grammar of "targets"](https://docs.python.org/3/reference/simple_stmts.html#grammar-token-python-grammar-target_list)
that are "things you can assign into that looks like this:

```text
target_list ::= target ("," target)* [","]
target      ::= identifier
              | "(" [target_list] ")"
              | "[" [target_list] "]"
              | attributeref    # a.b
              | subscription    # a[i]
              | slicing         # a[i:j]
              | "*" target
              | target ":" expr
```

Here are some examples:

```python
def example(...):
    # Implicit declaration or reassignment
    simple = f()
    # Parens are for grouping.
    (parens) = f()
    # Tuples have commas
    tuple1, tuple2 = f()
    (t1, t2), (t3, t4) = f()
    tuple1, *rest = f()
    # accessing attributes of objects
    obj.field.field2 = f()
    # subscripting and slicing too
    list[n] = f()
    # Type annotation
    value: SomeType = f()
```

The way this works is through recursion: this concept is known as a "pattern"
in other languages. A more generalized form of these expressions is used in the
Python "match" statement (which adds so-called "failable" patterns). The type
annotation is technically not recursive in the target grammar (it is tightly
bound to assignment statements), but we include it in the collection here for
simplicity.

## Mojo `var` and `ref` extensions

We want Mojo to support all these forms, and we need to solve a few extra
considerations because Mojo has systems programming features: types, scopes,
reference bindings, etc. We want Mojo to be compatible with Python wherever
possible, but can add some extensions. Furthermore, while compatibility is
specifically important for `def` declarations and while we do have more
flexibility `fn`s, we don't want them to be completely discordant from each
other.

Sidebar: Mojo and Python being syntactically identical is not required, but we
want to keep the languages close to each other and mechanically transformable
with high confidence.

The two most important extensions that Mojo adds are the notion of a `var`
binding and the notion of a `ref` binding: `var` defines a "lexically scoped"
variable (as opposed to a "function scoped" variable like in Python) and `ref`
captures the address of an LValue expression, rather than dereferencing and
copying the value.

Let's look at an example of these, working on a mutable
list of integers, with an `if` statement. Recall that `List.__getitem__`
is defined to return a `ref` to the element inside the list.

```mojo
def foo(mut list: List[Int]):
  if some_cond():
     # A `var` defines scoped storage that owns a value.  As such, assigning
     # the ref result of the getitem does a copy of the value (and `Int`, which
     # has a trivial copyinit) into the storage.  The scope of `a` is limited
     # to the `if` statement.
     var a = list[0]

     # Mutations of 'a' mutate the copy of the value the variable maintains.
     a += 1
     print(list[0]) # Value in the list is unchanged.

     # Assigning into an unknown name creates a new implicit *function*-scope
     # variable in Mojo, just like in Python.
     b = list[0]

     # Mutations of 'b' mutate the copy of the value the variable maintains.
     b += 1
     print(list[0]) # Value in the list is unchanged.

     # Assigning to a `ref` captures the reference returned by getitem and
     # binds it to the name 'c'.  In this case the element inside the list.
     ref c = list[0]

     # Once assigned, the `ref` itself cannot change, so all reads and writes
     # mutate the value referenced.  In this case, the list item changes.
     c += 1

     print(list[0]) # Value changed!
```

Mojo handles these patterns by adding them to the target grammar, building on
the grammar productions above:

```text
target ::= "var" target_list
         | "ref" target_list
         | ... other productions above ...
```

The combination of these means that you can define composite patterns and mix
them with `var` (and `ref`):

```mojo
def fancier_patterns():
    # Assign two variables at once
    var a, b = (foo(), bar())

    # Declare two uninitialized variables
    var a2, b2 : Int, String

    # Unpack a tuple
    var c, d, e = some_tuple_returning_function()

    # Scope one var locally, one to function scope.
    (var f), g = (foo(), bar())

    # The parens here are weird and pointless, but valid.
    (var h) = foo()

    # Function-scoped typed pattern
    i: Int8 = 42

    # Right side type infers from the left side to know the element type of the
    # list.
    j: List[Int] = []

    # Can unpack into more complicated lvalues too.
    a.field, b[1][i] = some_tuple_returning_function()
```

This also means that `var` and `ref` "statements" in functions are actually not
special statements in Mojo: they are just variants of the existing assignment.
Statement.

### Today's limitations

Mojo does not yet support the `[a, b, c] = foo()` syntax, nor does it support
the star syntax `a, *b = foo()` yet. Unpacking and is currently hard coded to
the standard library `Tuple` type, but we can generalize it in the future to
work with other types when there is demand.

## Where target patterns can be used

We've shown simple example with the `=` assignment operator above, but target
patterns are supported in a few other places, compositionally:

```mojo
def more_target_examples():
    tuple_list = [(1, 2), (3, 4)]
    # For statement
    for i, j in tuple_list: pass
    # List/dict/set comprehensions.
    my_list = [x+y for x, y in tuple_list]
    # ref bindings work in comprehensions too.
    my_list2 = [x.field for ref x in expensive_values()]

    # Walrus operator too.
    var x, y = (1, 2)
    var a, b = (x, y) := (3, 4)

    # except clause of a try/except is a target.
    try:
        ...
    except target:
        ...

    # Compound assignment and typical LValues do NOT allow targets because the
    # value must be initialized before it can be passed in.
    var invalid += 42 # error!
```

Note that while `try/except` statements technically use targets, this is pretty
useless in practice, because the only type you can throw is an `Error` which
can't be unpacked.

## Pattern scoping behavior

Now that we know where targets are allowed, let's talk about how they behave. As
we mentioned above, `var` and `ref` patterns allow you to directly control the
scoping of a variable binding, and this behavior is consistent in `def` and
`fn`. However, Mojo allows assignments to patterns without `var` or `ref` as
well, and here the behavior is perhaps surprising:

```mojo
def examples():
  a = 4  # Function scoped implicit definition or reassignment.

  # 'for' statement binds to a local scope.
  for b in [1, 2]: pass
  use(b) # Error: b is not in scope here, unlike Python

  c = 42
  # 'c' in the list shadows 'c' in the function and doesn't collide.
  list = [c*c for c in [1, 2, 3]]
  print(c) # This prints 42 like in Python.
```

When names are bound if `for` statements, they are bound into a local scope with
semantics that mirror the `read` convention: returned references are bound to a
read-only reference (avoiding a copy, and preventing mutation) and returned
values are bound into an owned temporary and an immutable reference is live
across the loop body. This is why you see behavior like this:

```mojo
def examples():
  # Valid even though the elements are non-copyable, because 'a' is an immutable
  # reference.
  for a in CollectionOfNonCopyableValues():
     use(a)

  for b in [1, 2]:
     b = 42 # error, cannot assign into immutable reference.

  list = [1, 2]
  for ref c in list:  # c is a ref
     c += 1 # Ok, mutable reference!
  assert(list == [2, 3])

  for var d in [1, 2]: # d is a var
     d += 42 # Ok, modifying the local 'd' var, doesn't affect the list.

  for e in range(10): # range iterator returns Int by copy, not by reference.
     e = 42 # error, cannot assign into immutable reference.
```

## Controversial behavior

I'm aware of the following controversial behavior:

1) Some argue that implicit variable assignment in a `fn` should be an error,
   because it is likely to be a typo. The Mojo compiler now warns about
   unused stores, so many of the common bugs are now detected, but it does feel
   inconsistent with the use of targets in `for` which *are* implicitly scoped.

2) Alternatively, if we allow implicit definitions in a `fn` we could treat them
   as implicit scoped, which would allow things like this without a type
   conflict:

   ```mojo
   def test(cond: Bool):
     if cond:
        a = 42
        use(a)
     if cond:
        a = "foo"  # Different `a` so it can have a different type.
        use(a)
    ```

3) Defaulting to `var` in a `for` pattern in a `fn` can lead to surprising
   behavior, because the copy is mutable, but a mutation of the name changes
   the copy, not the original collection:

   ```mojo
   def test(mut int_list: List[Int]):
       # Error cannot copy element into `a`.
       for a in collection_of_noncopyable_values(): ...

       for a in int_list:
          # Valid, but mutates the copy, not the element in the list.
          a += 1

       for ref a in int_list:
          # Valid, and changes the element of the list.
          a += 1
   ```

   Some propose that we should default to capturing the element as an immutable
   reference (as in the `read` argument convention) when the `__next__` member
   returns a reference. A `__next__` returning a value directly would need to
   be defined: it needs to be placed in storage, but that could be projected as
   an immutable reference too? We couldn't do this in a `def` because Python
   allows local mutation.

4) You could go further and suggest that we allow `mut` and `read` keywords in
   a target list. I don't see strong motivation to do this though - `mut` would
   be confusing with `var` (but is a mutable reference, not an owning copy) and
   `read` should probably just be the default, so I don't see a reason to allow
   it to be explicitly written.

5) For comprehensions in a `def` should scope their variables by default, this
   is just a simple bug to be fixed.

## Alternatives considered

Several alternate designs were considered: This section captures some of them.

## Why not allow multiple type annotations?

["yinon" asked](https://forum.modular.com/t/variable-bindings-proposal-discussion/1579/2)
why we don't support `var a2: Int, b2: String`.

In Mojo, type annotations are part of the assignment grammar, not part of the
recursive expression grammar, so they can only be used on the outer part of a
pattern, indeed even `(var a2: Int) = 4` isn't valid, but `(var a2): Int = 4`
is. We could relax that in principle, but it wouldn't scale well. Let's
explore why:

First, people would expect assignment to work, which could be done if type
patterns are at the right precedence level:

```mojo
var a2: Int, b2: String = fn_returning_int_and_string()
```

Given this, we would then be asked to support the next "obvious" adjacent thing:

```mojo
var a2: Int = fn_returning_int(), b2: String = fn_returning_string()
var a2 = fn_returning_int(), b2 = fn_returning_string()
```

However, this is a fundamentally different design than Python or Mojo use for
their expression grammars, because now we’re taking `=` and using it as an
expression instead of a statement.

If we don't want to change that (in the foreseeable future), then there isn't a
strong argument to support things like:

```mojo
var a2: Int, b2: String = fn_returning_int_and_string()
```

First, people will ask for the next obvious thing, but second, reasonable people
could be confused about whether b2 is being initialized or a2/b2 together (it
needs to be both). If we don't want to support this, then it doesn't seem like
a good idea to support `var a2: Int, b2: String` unless and until we decide we
want to scale this all the way.

For the foreseeable future, we will keep things simple and narrow, rather than
providing a partially paved path. If we decide this is important enough to
address, we should scope solving the full problem.
