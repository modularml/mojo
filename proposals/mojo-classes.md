# Classes in Mojo

**Author: Stef Lindall**\
**Status: Draft**\
**Date: August 14, 2024**\
**Last updated: August 14, 2024**\

A frequent refrain about Mojo's semantics differing from Python's is that
"Mojo doesn't yet have classes", which will behave more like Python types.
It is _not_ necessarily a goal of Mojo to precisely implement Python classes;
however, in order to decide how Mojo classes should work, it's critical to at
least understand how Python classes work, so we can make informed choices about
tradeoffs when we choose to differ.

This whitepaper attempts to be a path marker in that direction. It is split into
two halves with different goals:

1. Fully describe the Python object model.

    This should be entirely objective, with the goal that if we implemented the
    described behavior in Mojo, it would pass 100% of the relevant CPython
    conformance test suite. If you know of edge cases where it is inaccurate,
    please raise them.

2. A set of proposals for Mojo to implement classes.

    This will naturally include some degree of my own perspective. The goal
    isn't to be objective, but to try to identify the critical vs incidental
    complexity in the existing Python object model, judged by its relative
    usefulness in the broader Python ecosystem, and attempt to measure tradeoffs
    in implementing the behavior in Mojo vs Mojo's own language goals.

This document is _not_ a full summary of the Python Data Model defined here:
<https://docs.python.org/3/reference/datamodel.html>. In particular it 1)
attempts to omit things which are part of the CPython implementation but not
standard (though this is a pretty soft boundary in practice) and 2) does not
make any attempt to exhaustively list the set of `__magic__` dunder attributes
recognized by Python, especially as they relate to specific protocols not
central to the object model (for example, iteration or numerics).

## The Python object model in detail

### Scope -- What is the Python object model?

Concretely, the object model defines:

- For an object, what is its type?
- How are new types defined?
- When I access an attribute on an object, what happens? What is the type of
    that expression?

I will attempt to aggressively ignore any behavior which does not directly
serve answering these questions.

### Glossary

You probably know these terms already, but to make sure we're saying precisely
the same thing, they're defined here.

- `object`: The base type of all objects in Python, and a name used to describe
instances of python objects.
  - Internally it's represented as a pair:
    - `__class__: type`
    - `__dict__: dict[str, object]`
      - (this is different for objects defining `__slots__`, but conceptually
        still true)
    - Informally, instances are often named `obj` in code to avoid a name
        conflict with `object`
  - `isinstance(x, object)` for all `x` in Python (including `object`)
- instance: A synonym for an object
- `type`: A heavily overloaded term here.
    1. An object instance describing a type
    2. A function which returns an object's `__class__`
    3. The `type` metatype: The following are all true.
        - `type is type(object)`
        - `type is type(type)`
        - `isinstance(type, object)`
        - `isinstance(object, type)`
        - `type.__mro__ == (type, object)`
  - Informally, instances are often named `cls` in code to avoid a name conflict
    with `type` (`class` is a keyword)
- `attribute`: A named member of an `instance`
  - Accessed by the syntax `instance.attribute`
  - Conceptually, an object's attributes are exactly the members of `__dict__`
  - Python provides mechanisms to customize what a type considers to be an attribute
- `__dict__`: The attributes of an object (if it doesn't define `__slots__`)
- `__class__`: An attribute of objects which resolves to the type of that object
- `__bases__`: An attribute of type objects which contains their direct parent types
- `__mro__`: "Method resolution order" -- An attribute of type objects which
    contains a totally ordered sequence of all their ancestor types. Python uses
    the C3 linearization (see appendix).
- metaclass: A word for `type(type(x))`
- `issubclass(sub, cls)`: A builtin function defined on types. Generally true if
    `cls in sub.__mro__`. The object model provides mechanisms to customize this
    behavior.
- `isinstance(obj, cls)`: A builtin function. Generally true if
    `issubclass(type(obj), cls)`. The object model provides mechanisms to
    customize this behavior.
- protocol: An implicit interface or set of behaviors conformed to by a type
- descriptor: An object of some type which implements the "descriptor protocol",
    which allows attributes to be lazily evaluated.
- namespace: A `__dict__` without an associated `__class__`
- `locals()`: The locally scoped namespace dictionary
- `globals()`: The module scoped namespace dictionary
- name: A string key in a namespace
  - NB: Notice that a name _does not have a type_!
- annotation: A syntactically valid python expression that is never evaluated by
    the object model.
- `__annotations__`: A namespace member mapping names to annotation values.

### Memory model

All Python objects are reference counted, including compiler builtins like `None`.
Since types are implicitly references, there's not problem defining recursive types:

```python
class LinkedListNode:
    next: Optional[LinkedListNode]  # this is fine!!
```

Most of the rest of the memory model are CPython implementation details:

- An object can't be explicitly deleted.
  - It will be garbage collected when its reference count goes to 0, or when the
    cycle detector detects that it is part of a cycle with no external references.
  - When an object is deleted, if it has a `__del__` method, that method _may_
    be called. If it throws, the exception is ignored.
- The only way to explicitly allocate an object is with `object.__new__(cls)`.
- `obj.__weakref__` contains a list of weak references which point at `obj`

### Object model

Following is a conceptual implementation of Python's object model, as though it
were implemented directly in Python. It attempts to be accurate, in other words
if we implemented precisely this object model, we would be directly compatible
with Python types. The goal is to provide a shared basis of understanding of
what compatibility looks like, as well as a reference for an eventual implementation.

#### `object`

```python
## NOTE: Attribute accesses in this implementation are sometimes fudged
##       as direct access, when they couldn't be because of recursion.
class object(metaclass=type):
    def __new__(cls):
        # Can't be implemented in pure python
        # Allocates a new Python object with an empty __dict__, and __class__ = cls

    def __init__(self):
        pass

    @property
    def __class__(self):
        # Isn't implemented in pure python.
        # Returns the __class__ from the underlying C object.

    def __dir__(self):
        return list(self.__dict__)

    def __getattribute__(self, attr):
        for base in type(self).__mro__:
            if attr in base.__dict__:
                classvar = base.__dict__[attr]
                # Resolve data descriptor; See appendix
                if hasattr(classvar, ("__set__", "__delete__")):
                    return classvar.__get__(self, base)
                # Found non-data-descriptor
                break
        else:  # No classvar defined, do instance lookup
            if attr in self.__dict__:
                return self.__dict__[attr]
            raise AttributeError
        # resolve non-data descriptor
        if hasattr(classvar, "__get__"):
            return classvar.__get__(self, base)
        return classvar

    def __setattr__(self, attr, value):
        for base in type(self).__mro__:
            if (value := base.__dict__.get(attr)) and hasattr(value, "__set__"):
                value.__set__(self, value)
                return
        else:
            self.__dict__[attr] = value

    def __delattr__(self, attr):
        for base in type(self).__mro__:
            if (value := base.__dict__.get(attr)) and hasattr(value, "__delete__"):
                value.__delete__(self)
                return
        if attr in self.__dict__:
            del self.__dict__[attr]
        else:
            raise AttributeError

    # Other methods related to str/hash/pickling/comparison/etc omitted
    ...
```

#### Attribute access

Attribute access `instance.attribute` is syntax sugar for
`getattr(instance, "attribute")`, with the following implementation:

```python
def getattr(obj, attr, default=MISSING):
    # Try resolving through __getattribute__
    try:
        return obj.__getattribute__(attr)
    except AttributeError as e:
        exc = e

    # Try resolving through __getattr__
    try:
        if hasattr(obj, "__getattr__"):
            return obj.__getattr__(attr)
    except AttributeError as e:
        exc = e

    # Nothing found, return default or raise
    if default is not MISSING:
        return default
    raise exc

def setattr(obj, attr, value):
    obj.__setattr__(attr, value)
```

#### Methods

Functions in python implement the descriptor protocol. The `function` can be
thought of like

```python
class function:
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return method(self, instance)

    def __call__(self, *args, **kwargs):
        # execute the function
```

allowing them to be used as free functions, except when accessed as an attribute
of an instance, in which case they become a bound method; conceptually:

```python
@dataclass
class method:
    __func__: function
    __self__: object

    def __call__(self, *args, **kwargs):
        self.__func__(self.__self__, *args, **kwargs)
```

`@property`, `@staticmethod` and `@classmethod` decorators don't require any
special behavior, and may be implemented in pure python as descriptors

#### `type`

`type` is a function which returns an object's type:

```python
## `type` has this special overload
def type(obj):
    return obj.__class__
```

`type` is also the base metatype.

`type` is primarily responsible for defining how inheritance works. Python uses
the C3 type linearization algorithm by default to construct a type's method
resolution order.

Beyond that, `type` may be subclassed to create different metaclasses, with
customized behavior for new type creation, namespaces, or customizing
`isinstance` and `issubclass` behavior (particularly useful for things like `Protocol`.)

Conceptually, `type` could be implemented as

```python
class type:
    def __init__(self, name, bases, namespace, **kwargs):
        self.__name__ = name
        # See appendix for implementation. In ~all python code this does
        # resolve_bases is a no-op, but it supports inheriting from non-types.
        self.__bases__ = resolve_bases(bases)
        self.__mro__ = self.mro(bases)
        for name, attr in namespace.items():
            setattr(self, name, attr)

            # Set descriptor names
            if hasattr(attr, ("__get__", "__set__", "__delete__")) and hasattr(attr, "__set_name__"):
                attr.__set_name__(self, name)

        self.__init_subclass__(**kwargs)

    def __init_subclass__(self, **kwargs):
        pass

    def mro(self):
        # See appendix for implementation of C3 mro
        return [self] + c3_mro(self.__bases__)

    @property
    def __dict__(self):
        # Not implementable in pure python. This is where
        # all accesses of `__dict__` bottom out.

    @classmethod
    def __prepare__(cls, **kwargs):
        return {}

    def __call__(self, *args, **kwargs):
        # This is the real implementation of what we more
        # naturally think of as a "constructor" call
        new_instance = self.__new__(*args, **kwargs)
        new_instance.__init__(*args, **kwargs)
        return new_instance

    # NOTE: Attribute access here fudged, the true implementation
    #       won't recursively call `__getattribute__.
    def __getattribute__(self, attr):
        # __getattribute__ has different behavior for types:
        #  - descriptor doesn't pass an `instance`
        #  - iterate self.__mro__ instead of type(self).__mro__
        for base in self.__mro__:
            if attr in base.__dict__:
                value = base.__dict__[attr]
                if hasattr(value, "__get__"):
                    value = value.__get__(None, self)
                return value
        else:
            raise AttributeError

    def __instancecheck__(self, instance):
        # Called by `isinstance`
        return issubclass(type(instance), self)

    def __subclasscheck__(self, subclass):
        # Called by `issubclass`
        return self in subclass.__mro__

    def __subclasshook__(self, subclass):
        return NotImplemented

    def __subclasses__(self):
        # returns the list of types T having self in T.__mro__
        ...

    # This seems to be vestigial
    @property
    def __base__(self):
        return None if self is object else self.__bases__[0]

    # Other methods like str/repr omitted
    ...
```

#### Class creation

```python
class NewType(Base1, Base2, metaclass=type, **kwargs):
    <body>
```

is syntactic sugar for

```python
## Set up the class namespace
qualname = locals().get("__qualname__")
namespace = metaclass.__prepare__(**kwargs)
namespace["__module__"] = locals().get("__module__")
namespace["__qualname__"] = f"{qualname}.NewType" if qualname else "NewType"
namespace["__annotations__"] = {}

## See appendix for min_type implementation
metaclass = min_type({metaclass, *(type(base) for base in bases)})

## Execute the class body to populate the remainder of the namespace
exec(compile(body), globals=globals(), locals=namespace)

## Actually create the type
NewType = metaclass(name="NewType", bases=(Base1, Base2), dict=namespace, **kwargs)
```

### Descriptor protocol

Among protocols, the descriptor protocol is undoubtedly the one most central to
Python's object model, evidenced by its heavy use in the above implementation.

There isn't a formal protocol definition in the implementation, but it would look
like this:

```python
class Get(Protocol):
    def __get__(self, obj, cls=None): ...

class Set(Protocol):
    def __set__(self, obj, value): ...

class Delete(Protocol):
    def __delete__(self, obj): ...

NonDataDescriptor = Get
DataDescriptor = Set | Delete
Descriptor = DataDescriptor | NonDataDescriptor
```

Data descriptors and non-data descriptors differ in that a data descriptor takes
precedence over an instance attribute (one in `__dict__`) while a non-data
descriptor does not.

Descriptors may also implement `__set_name__(cls, name)`, which is called during
type creation.

### Generics

Python has two syntaxes for defining type parameters. It has a mojo-like type
parameter syntax, introduced in [PEP 695](https://peps.python.org/pep-0695/),
as well as `typing.TypeVar` and `typing.Generic`.

#### Type parameters

Using the type parameter syntax is syntactic sugar for defining `TypeVar`s and
inheriting from `Generic`, but is also allowed on functions and methods.

These look approximately like Mojo's parameters:

```python
class ClassA[T: str]:
    def method1(self) -> T:
        ...

def func[T](a: T, b: T) -> T:
    ...

type ListOrSet[T] = list[T] | set[T]
```

- `type` implements `__getitem__` where `type[Any] == type`, and `type[T]`
    represents `type(T)` in a generic context.
- `type` implements `__or__`, which allows type unions to be defined as `X | Y`.
- `__origin__` on a parameterized type holds the un-parameterized generic type
- `__type_params__` holds a tuple of type parameters of a generic type or function
  - Though not specified, `__parameters__` appears to alias `__type_params__`
- `__args__` holds the values of the type parameters of a parameterized generic
    type or function
- `__value__` is in the spec as holding the "resolved" type but doesn't appear
    to be implemented in CPython
- There appears to be no way to retrieve the annotations of type parameters

> [!IMPORTANT]
> `T: str` in Python is a _covariance_ declaration, whereas Mojo currently defines
> this as a _type_ declaration. In other words, in Python `class ClassA[T: str]:`
> means that `issubclass(T, str)`, while in Mojo this would currently mean
>`isinstance(T, str)`. This represents an important conflict to resolve in implementation.

#### `typing.Generic` and `typing.TypeVar`

`Generic` is of type `GenericMeta`, which implements the machinery for generic
types, such as `__class_getitem__`. This may be seen as a historical artifact,
as `type` itself is now subscriptable to support generic typing.
[PEP 560](https://peps.python.org/pep-0560/) explicitly identifies removing the
dependency on `GenericMeta` as a motivation for the development of
`__class_getitem__` and `__mro_entries__`.

`Generic` types

### `typing.Protocol`

`Protocol` is a type which can be subclassed to provide a Mojo-trait-like type:
any type with the provided interface will type-check as that type in type
checkers like Pylance and MyPy, and with `isinstance` if the type is
additionally annotated with `@runtime_checkable`.

[MyPy's Protocols and structural subtyping](https://mypy.readthedocs.io/en/stable/protocols.html)
docs have many good examples. Protocols are the most literal way to type check
Python's "duck typing" standard: types are never required to inherit from a
protocol (and generally shouldn't; while Protocols can implement default method
implementations, duck-typed "subclasses" don't inherit them, and `Protocol`
has its own metaclass, so this is all-around better achieved with a mixin type).

### Abstract base classes

ABCs were introduced alongside metaclasses in 2007 in Python 3.0. Python type
hints were introduced 7 years later, and `typing.Protocol` was released in 2017.
You can see ABCs as a specific set of predefined protocols, along with mixin
types. In fact, many of them are named similarly to Mojo traits: `Hashable`,
`Sized`, etc.

Of particular interest are the `Sequence`, `Mapping`, and `Set` types, which
provide mixin methods to define types implementing these protocols more easily.

ABC also defines the `@abstractmethod` and related decorators, which more
directly map to a traditional OO model. Types inhereting classes annotated with
these methods must either declare themselves to also be abstract, or implement
the required methods, otherwise they will raise during class creation.

### Metaclass conflicts

Python needs to choose a single type to use as a metaclass when creating a new
type. It does this in a sensible way: among all candidate metaclasses
(metaclasses used by bases, and any specified metaclass), it tries to use the
most derived one, and if there isn't a unique such type, it errors.

This is a sensible behavior, but it means that inheriting from a metaclass is a
_very_ expensive thing to do for a library: if someone uses your types, they
can't use _any_ other library which also uses metaclasses.

This is a particular pain point with `typing.GenericMeta` and
`collections.abc.ABCMeta`. Since these types are implementing typing for
"normal" python types, they (1) can't be used together, and (2) can't be used
with any libraries which use mataclasses, for instance `enum` or most ORMs.

This is in fact why `@dataclass` is implemented as a decorator, when a metaclass
would have been a much more natural fit. See
[PEP 557](https://peps.python.org/pep-0557/) for more details, which
specifically identifies dataclasses as a natural application of metaclasses, and
_not_ having a metaclass being a central decision in the design.

### `__slots__`

There's a separate opt-in object model pattern if a class defines
a `__slots__` class variable, eg.

```python
class Foo:
    __slots__ = ["bar", "baz"]
```

I'll link the [docs here](https://docs.python.org/3/reference/datamodel.html#slots)
on how slots behave, since it goes a bit outside the scope of this document.
Critically these objects lack a `__dict__`, and have a number of restrictions on
usage, including not being allowed to assign arbitrary attributes to instances,
and strong limitations on subclassing. This comes with reduced memory usage and
improvements in attribute access time.

Since types with `__slots__` behave _almost_ exactly like objects without them,
other than restrictions, and could be implemented with an appropriate metaclass
in the existing object model, they don't require substantial support and I won't
cover them more deeply here.

### Python Crimes

There's many behaviors supported in the Python object model that the community
has widely agreed are bad practice. These have vanishingly small usage in
practice, and the language provides other usage patterns to accomplish most
things they might be useful for.

#### Assigning to `__class__`

- This is the only way to mutably change the type of an object.
- In practice I've only used this extremely rarely, and in retrospect never for
    a good reason.
- It's always possible to instead update a reference to point to a shallow or
    deep copy as a new type instead. The only semantic difference is that it
    doesn't update other referrers.
- Conceivably it might be useful for certain hot-reloading patterns, updating
    references in the interpreter in-place to a new instance of the type. PyPy
    has experimented with this but the common Python hot-reloaders don't
    actually do it.
- This has a storied history; PyPy has already removed this behavior, and
    Python 3 raises a TypeError for certain cases of this.

#### Assigning to `__bases__` or `__mro__`

- I've never actually seen this done or done this.
- Again this falls into the category of "plausibly you might do this for
    hot-reloading, in practice no one does".

#### Assigning to `__dict__`

- This is a surefire way to burn yourself. I don't know of any widely used
    patterns that do this or might want to.

#### Assigning arbitrary attributes to `__dict__`

- This is obviously supported through `__getattr__` et al, but it's additionally
    just directly supported by all types not implemented with `__slots__`.
- In some sense this is the "expected" behavior, thinking of Python as a
    scripting language: when calling `__init__`, a priori we don't know what
    attributes a class _should_ have, so we can't place expectations on those attributes.
- There's a wide tail of other use cases for this behavior, such as
  - Decorators frequently assign a new attribute to a function or type to "tag"
    it with metadata. This is a very fragile pattern I'm not a fan of, but is
    relatively widely used.
  - Quick and dirty caching:
    `def x(self): return self._x if hasattr(self, "_x") else (self._x := _compute_x())`
- I'm convinced if this pattern didn't exist, everything it's used for could
    ~easily be done another way, however I am also prepared to be surprised by
    how much code actually does this.

#### Inheriting from objects that aren't types

- There's some support in the language for this through `__mro_entries__`.
    [PEP 560](https://peps.python.org/pep-0560/) discusses this. It is primarily
    useful because of metaclass conflicts, which became more common with the
    addition of generic types into python typing depending on `GenericMeta`.
- I doubt this is widely used, but I don't have data to back that up. It's
    likely we could design for generic types more directly from the start to
    avoid this issue (though harder to avoid the metaclass conflict issue more generally).

#### Types which provide `__bases__` or `__mro__` dynamically via a descriptor or `__getattr__`

- I've never heard of anyone doing this but it is clearly chaotic evil.
- I bet you could easily make this mechanism alone turing complete so it
    undoubtedly has amazing use cases, and I also am confident this will never
    be practiced. You could certainly implement an approximation of this many
    other ways.

#### Dynamic inheritance: `class Foo(Bar if x else Baz):`

This looks insane but is actually really useful in practice. For instance

- A CSV-loader interface might want to read the CSV header, and then construct a
    new row type which inherits from `collections.namedtuple(headers)`.
- Similarly, Pandas `Dataframe` has dynamic attributes for the name of each
    column in the dataset, which are data-dependent.
- A core use case for class decorators is to return a new subclass of the
    decorated class.
- It's sometimes useful to create an "anonymous" subclass of a type, created in
    some function scope.

#### Full `exec` dynamism: `exec("class Foo: ...")`

- `collections.namedtuple` used to be implemented this way
- I believe that the modern object model is expressive enough that there aren't
    any regular use cases of this anymore.

#### [Liskov substitution principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle) violations

Python subtypes don't prevent you from overridding parent type attributes with
incompatible attributes that would make them illegal to pass as a parent type.
In practice there's lots of code that accidentally does this, although MyPy
recognizes it as a typing error.

## Proposals to implement Mojo classes

This section has considerably more of a perspective than the previous. The goal
is to provide a concrete proposal for how Mojo could move forward implementing
classes which is faithful both to Python's object model and to Mojo's own
language goals.

### Types as types vs types as functions

Python may be thought of as a strongly typed, dynamically type functional language.
Objects have types, names do not. Types may never be converted into other types
(except the specific "Python Crime" of assigning to `__class__`). While it appears
imperative on the surface, all python types are both objects and functions, and
the "constructor" of a type in fact doesn't even need to do that; for instance,
`__new__` may be overloaded to return singleton instances or interned objects.
Instead, it's often more useful to think of types as a function which transforms
a value into a value of another type. While this perspective is pretty far from
what we normally think of as object-oriented programming, it makes a lot of the
more common "python crimes" listed below make more sense.

By contrast, Mojo is statically, weakly typed. Names have types, and only objects
of those types may be associated with that name. Mojo allows implicit conversion
of types into other types. Mojo constructors are exactly that in the traditional
sense: they allocate and initialize memory.

This contrast is in some sense the deepest disparity between Python and Mojo,
and in order to be successful in its goals, Mojo classes will need to fully
embrace Python's perspective on these topics.

### High level perspective: Mojo's goals

The Python 3 language spec is a moving target. Many attempts at alternative
language implementations exist, but as CPython is co-developed with the Python
spec, they are necessarily approximations.

Mojo also has its own goals. It wants to be a compiled, performant, full-stack
language, while building a healthy Pythonic-language ecosystem of libraries
shared between it and CPython.

Mojo and Python in this sense have a lot of shared goals. We should, at a
minimum aim for most Python libraries to "just work" on both Mojo and CPython.
However, realistically around the edges and in uncommon cases, we will need to
make separate decisions about precise implementation details, and Mojo can't
realastically pass the entire CPython compatibility test suite.

We should identify some _subset_ of it that we intend to pass, make this a clear
contract with our users, work closely with CPython in spec development in these
areas, and make it easy for users and library authors to write code which is
compatible with both. The _natural_ way to write Python and Mojo should be the
same on this set of language features. The vibe should be "you're using Python?
your use case will almost certainly _just work_ on Mojo", possibly with a small
set of compatibility changes that will continue to allow it to work in Python,
and that we help automate (think python's `six` library).

The truth is that there are decisions in Python that we should realistically
consider _not_ carrying forward; warts which exist for historical reasons but
are too much work to rip out, and that a vanishingly small percent of the Python
ecosystem relies on.

### Proposed differences for Mojo's object model

It would be incredibly valuable to Mojo, as a compiled language, to be able to
reason more about the set of possible things that may be done with an object.
Identifying tradeoffs of where simplifying the object model allows us a better
implementation, vs the cost in compatibility and expressiveness, is the core
job of this proposal.

I'm framing these tradeoffs primarily with the intuition that moving attribute
and method resolution to compile time where possible is the most valuable
benefit, and that we shouldn't change what isn't broken except in service of
this goal.

Following is an initial proposal of a Mojo object model for classes, which
retains most of the dynamic use cases of Python objects, while allowing
hopefully the vast majority of existing cases to "just work" with compile-time
method and attribute resolution.

#### Any Mojo type which is a `class` is implicitly wrapped by an `Arc`

Existing Mojo `fn` and `def` semantics are unchanged:

- If an argument is a `struct`, it retains its current behavior
- If an argument is a `class`, it will behave as an owned reference, and that
    reference will be reference-counted
  - Attribute lookup behaves as it would for a direct reference
  - It is implicitly convertible to an `Arc[Object]` for lower level access

#### Implement Python's object model for classes, with compile-time method resolution

`__getattribute__` and `__getattr__` may still be used to implement dynamic
attributes, but in the common case attributes are resolved statically.

- `__dict__` may not be assigned to directly
  - For compatibility, it may be accessed, but these function as the variants of
    `setattr` and `getattr` described below.
- `__class__`, `__bases__`, and `__mro__` are immutable and may not be assigned to
- Class attributes must have a static type
  - All attributes declared in the class body.
  - Classes inherit the attributes of all types in their `__mro__`.
  - If two types in the `__mro__` share an attribute name, they must either be
    the same type, or must be linear ancestors of each other, in which case they
    take the most derived type.
- Classes are separated into implementation categories:
  - Type 1: Classes which define neither `__getattr__` nor `__getattribute__`
    in their `__mro__`
    - Assigning to or accessing an undefined attribute is a compile error.
    - `__setattr__` may be defined, but will raise an `AttributeError` if called
        with any unknown attribute.
    - `__delattr__` may not be defined
  - Type 2: Classes which define `__getattr__` but not `__getattribute__`.
    - Behave like Type 1 classes, except where the compiler would fail td
        resolve an attribute lookup, it instead emits a call to `__getattr__`.
    - `__delattr__` may be defined, but will raise a `TypeError` if called witd
        an attribute that resolves via Type 1 attribute lookup.
    - `__setattr__` may be defined with no restrictions
  - Type 3: Classes which define `__getattribute__`.
    - Any attribute access on these types will emit a call to `__getattribute__`.
    - `__delattr__` may be defined in the same way as Type 2 classes.
    - `__setattr__` may be defined with no restrictions
- We have a variant of the `getattr`, `hasattr`, `setattr`, and `delattr`
    functions which dynamically attempt to perform a Type 1 lookup on the object
    using its runtime type, or raise an`AttributeError`.
- Dynamic inheritance is allowed, but any type variable used as a base class
    must be specified as a [type parameter](https://docs.python.org/3/reference/compound_stmts.html#type-params)
    and the compiler must be able to infer its concrete type at compile time.

#### More invasive or opinionated changes

##### Simply the typing model around type parameters and Protocols

- Remove Abstract Base Classes and decorators. Re-implement ABC types as Protocols.
- Remove `typing.Generic`, only support type parameters.
- A class inheriting from Protocol is treated exactly as a trait definition.
- All traits and protocols behave as though they are annotated with `@runtime_checkable`

##### Require types to obey the Liskov substitution principle

- Attributes and method return types in child classes must be covariant to the
    same in their parent classes.
  - Attributes may be replaced with descriptors satisfying this princple.
- Method arguments must be contravariant to the same in their parent classes.
  - They may define additional optional arguments.
  - They may make previously required arguments into default arguments.
  - Since all functions in Python may raise Exception, any new raises are always
    legal.

##### Allow method overloading

- All members of a method overload set must be defined in the class body.
- If a subclass overrides an overloaded method, it forms a new overload set.
  - A subclass overriding only some overloads of a method will implicitly have
    additional overloads in its overload set calling super() methods.

##### Simplify descriptor semantics

Eliminate the distinction between non-data and data descriptors. Define a
descriptor as having a `__get__` with the correct interface. Descriptors follow
the same precedence rules as other name lookups.

##### Help avoid common metaclass conflicts

- If metaclasses are not linearizable when creating a new type, attempt to
create an anonymous subclass of conflicting metaclasses.

### Open questions

- How do we resolve the difference in semantics of Python type parameters
    defining covariance vs Mojo type parameters defining type?
- Python uses magic attributes for type parameters, while Mojo treats them as
    type attributes. What's the best way to resolve this in implementation?
- Mojo's type system should support some degree of higher-kinded types compared
    to Python. Is that in scope for this proposal?

## Appendix

### Out-of-line reference implementations

Find the unique most-derived type among input types, or `TypeError`.

#### min_type

```python
def min_type(types):
    assert types
    min_type = types.pop()
    for type in types:
        if issubclass(type, min_type):
            min_type = type
        elif not issubclass(min_type, type):
            # Inherited metaclasses must be totally orderable
            raise TypeError
    return min_type
```

#### C3 Method resolution ordering

This looks a bit complicated, but it's not actually too bad. See the
[MRO documentation](https://docs.python.org/3/howto/mro.html) and
[C3 linearization paper](https://dl.acm.org/doi/10.1145/236337.236343) for more details.

The Method Resolution Order is a "linearization" of
the ancestor classes of `cls`, which has the property
of being "monotonic": For any types `P` and `Q` in `Base.__mro__`
where `P` appears before `Q`, then for all types `Child` having `Base`
in `Child.__mro__`, `P` must also appear before `Q` in `Child.__mro__`.

This naive implementation of the C3 algorithm is O(N^2)
for simplicity.

```python
def c3_mro(bases):
    linearization = []
    mros = [list(base.__mro__) for base in bases]
    # We reduce the total number of elements in `mros`
    # by at least 1 per iteration, so time is bounded.
    while mros:
        # Search fo a good head
        for mro in mros:
            head = mro[0]
            for other_mro in mros:
                # Good iff it doesn't appear in any other tails
                if mro is other_mro: continue
                if head in other_mro[1:]: break
            else:  # no break, good head
                # Add head as the next base
                linearization.append(head)
                # Filter head out of all remaining lists
                mros = [(remaining := [t for t in mro if t is not head]) for mro in mros if remaining]
                break
        else: # no break, no good head
            raise TypeError(f"MRO conflicts among bases {bases}")
    return linearization
```

#### resolve_bases

If any non-type bases have a `__mro_entries__` attribute,
call it and replace them with its results.

```python
def resolve_bases(bases):
    def _bases(base):
        if isinstance(base, type) or not hasattr(base, "__mro_entries__"):
            return [base]
        return base.__mro_entries__(bases)

    return tuple(itertools.chain.from_iterable(map(_bases, bases)))
```

## References

- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Descriptor Guide](https://docs.python.org/3/howto/descriptor.html)
- [Python typing](https://docs.python.org/3/library/typing.html)
- [Type parameters for python generics](https://docs.python.org/3/reference/compound_stmts.html#type-params)
- [Method resolution order](https://docs.python.org/3/howto/mro.html)
- [C3 linearization](https://dl.acm.org/doi/10.1145/236337.236343)
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html)
- [PEP 484 - Type hints](https://peps.python.org/pep-0484/)
- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [PEP 544 - Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
- [PEP 560 - Core support for typing module and generic types](https://peps.python.org/pep-0560/)
- [PEP 557 - Data Classes](https://peps.python.org/pep-0557/)
- [PEP 3115 - Metaclasses in Python 3000](https://peps.python.org/pep-3115/)
- [PEP 3119 - Introducing Abstract Base Classes](https://peps.python.org/pep-3119/)
- [MyPy - Protocols and structural subtyping](https://mypy.readthedocs.io/en/stable/protocols.html)
- [Liskov substitution principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle)
- [Python garbage collection](https://devguide.python.org/internals/garbage-collector/index.html)
