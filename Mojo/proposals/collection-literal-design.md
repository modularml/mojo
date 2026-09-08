# Collection Literal Design

Date: May 18, 2025
Status: Implemented

This doc describes the design and behavior of collection literals in Mojo. It is
intended to explain how types can work with collection literals and how corner
cases are handled in one end-to-end place. It does not go into the design
details of specific collections like `List` or `Dict`.

## Background

Mojo inherits ergonomic collection literals from Python, for example, we want
things like this to work as a python programmer would expect:

```mojo
    mylist = [1, 2, 3]
    assert(mylist[1] == 2)

    myset = {1, 2, 3}
    assert(2 in myset)

    mydict = {"foo": 1, "bar": 2}
    assert(mydict["foo"] == 1)
```

Mojo has `List` and `Set` and `Dict` collection types, but we need to wire up
the collection literal syntax to work with these properly.

While we want collection literals to default to the standard library types, we
also want collection literals to work with other types as well. This is because
we want to enable specialized collections, e.g. lists that contain elements
inline, dictionaries specialized for string keys, and we also want to
interoperate well with Python.

```mojo
def take_inline_array(arr: InlineArray[Int, 18]): ...
def take_python(obj: PythonObject): ...

def example():
    # This should form an InlineArray directly. It shouldn't create a List and
    # copy it.
    take_inline_array([1, 2, 3])

    # This should pass a python list with python integers, string, set and dict
    # inside of it.
    take_python([1, "str", {2, 3, 4}, {"foo": 1, "bar": 2}])
```

This gives us the power of easy of use of ergonomic defaults, while also
enabling the construction of efficient datatypes and interoperability with
existing APIs.

## Defaulting rule

If Mojo knows the contextual type for a collection literal, it will use that
type to construct the literal. Otherwise it will default to the corresponding
standard library type defined in the collections package: `List`, `Set`, `Dict`
etc.

## List literals

In order to support list literal syntax, a type implements an initializer that
takes a `__list_literal__: NoneType` argument. The `NoneType` marker ensures
that this overload won't be picked accidentally, because some types may need to
support multiple literals (e.g. `PythonObject` needs to support set literals
and list literals, and the initializers do different things). For example,
`List` has this initializer:

```mojo
struct List[T: Copyable & Movable, hint_trivial_type: Bool = False](...):
     def __init__(out self, owned *values: T, __list_literal__: NoneType = None):
        """Constructs a list from the given values.
        ...
```

This allows `List` to be built from a list literal, and because the
`__list_literal__` has a default value, you can also use the same initializer
with explicit syntax like `List[Int](1, 2, 3)` if you'd like.

While it is conventional to define this initializer with a variadic, this isn't
required:

```mojo
# A type that allows initializers with 2 or 3 integers only.
struct TwoAndThreeList:
   def __init__(out self, a: Int, b: Int, __list_literal__: NoneType): pass
   def __init__(out self, a: Int, b: Int, c: Int, __list_literal__: NoneType): pass

def test():
  # These are ok.
  var a: TwoAndThreeList = [1, 2]
  var b: TwoAndThreeList = [1, 2, 3]

  # Error: no matching function in initialization
  var c: TwoAndThreeList = [1, 2, 3, 4]
```

Similarly, you can also define things that take heterogenous types:

```mojo
struct StringIntListLiteral:
   def __init__(out self, a: String, b: Int, __list_literal__: NoneType): pass

def test():
  # Weird but accepted.
  var a: StringIntListLiteral = ["foo", 2]
```

While this is possible, this isn't something you should actually do. The reason
that this is important to support is `PythonObject` which takes a heterogenous
collection of values that conform to `PythonConvertible`.

## Set literals

Set literals work the same way as list literals, but choose an initializer with
the `__set_literal__` keyword argument.

## Dictionary literals

Like other forms of literals, a type enables support for Dictionary literals by
implementing a constructor. Here is `Dict` for example:

```mojo
struct Dict[K: KeyElement, V: Copyable & Movable](...):
    def __init__(
        out self,
        owned keys: List[K],
        owned values: List[V],
        __dict_literal__: NoneType,
    ): ...
```

As with other literal types, the `__dict_literal__` argument avoids ambiguity
with other constructors. The constructor takes a list of keys and a list of
values. These are passed separately (rather than as a list of tuples) because
this enables more flexibility in type merging for slightly different types,
e.g. we want `{k1: 4.0, k2: 5}` to produce a value of type `Float64` even though
the literals have different types.

The actual type of the `keys` and `values` lists are flexible: the compiler
passes the keys and values as a list literal, so all the rules for list literals
apply to these arguments.

## Initializer lists

Mojo supports a syntax extension vs Python to express C++-style "initializer
lists". This is a list of argument values that is bound up into an expression
which is applied to a contextual type. This can be useful if you have a
complicated type that you don't want to spell.

```mojo
def foo(x: SomeComplicatedType): ...

# Example with normal initializer.
foo(SomeComplicatedType(1, kwarg=42))
# Example with initializer list.
foo({1, kwarg=42})
```

While this is a minor convenience in this case, it can be more significant when
working with more complex types that have lots of parameters that would
otherwise require complicated uses of `type_of(x)`.

## Ambiguity resolution between Set Literals and Initializer Lists

The above approach has an ambiguity: does `{a, b}` create a set of two elements
with `T.__init__(a, b, __set_literal__=None)` or does it invoke an initializer
with just `T.__init__(a, b)`?

The approach used by the Mojo compiler is as follows based on whether it has
a contextually inferred type or not. If not, the compiler assumes that the
literal must be a Set literal. It:

1) requires at least one element, rejecting `var x = {}` because it cannot infer
   the element type of `Set`.
2) rejects attempts to use keyword arguments, e.g. `var x = {1, kwarg=2}` is
   known to not be a set, but we cannot know what type to create.
3) It unifies the elements provided (using `__merge_with__` and implicit
   conversions using the same algorithm as list literals), and then creates an
   instance of the `Set` type.

If there is a known contextual type, the compiler uses the following approach:

1) If the initializer list is empty, and if the type allows construction from a
   dictionary literal, that constructor is used. This ensures that `{}` turns
   into a dict with `PythonObject`.
2) If the type supports the set literal initializer, and if no keyword arguments
   are used, the element types are unified and the set literal constructor is
   used. Note that this disables initializer-list syntax for set-like types,
   but you can work around this by calling the initializer explicitly, or
   providing a keyword argument.
3) Otherwise, the elements are passed in to the `T.__init__` method as an
   initializer list.

## Collection Comprehensions

Mojo supports Python-style collection "comprehensions", which are a collection
literal with a single expression, followed by one or more postfix generator
clauses indicating how to produce the elements. Some examples:

```mojo
  var list1 = [x*y for x in range(10) for y in other_int_list]
  var list2: YourList = [x for x in range(10) if x & 1]
  var set = [x for x in range(10)]
  var dict = {x : x*x for x in range(10) if x & 1}
```

Collection comprehensions desugar into the same code you'd get by writing nested
`for` loops and `if` statements, and uses the same defaulting rules as other
literals above. For example, the examples above desugar into:

```mojo
  var list1 = List[Int]()
  for x in range(10):
      for y in other_int_list:
         list1.append(x*y)

  var list2 = YourList()
  for x in range(10):
      if x & 1:
          list2.append(x)

  var set = Set[Int]()
  for x in range(10):
      set.add(x)

  var dict = Dict[Int, Int]()
  for x in range(10):
      if x & 1:
          dict[x] = x*x
```

This implies that this will only work with collection types that are default
constructible, and have the insertion methods corresponding to their type:
`append` for arrays, `add` for sets, and `__setitem__` for dicts. Mojo doesn't
require the collections to be copy or movable or impose any other requirements
on them.
