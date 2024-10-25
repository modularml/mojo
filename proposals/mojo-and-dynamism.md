# Mojo and Dynamism

Mojo has the lofty goal of being a simple, powerful, and easy-to-use language
like Python but with features that allow programmers to reach the performance of
C. One of Mojo's approaches is to start by adopting the syntax of Python and
provide an incremental typing-for-performance story where the performance of
Python code can be incrementally improved by adding type annotations, explicit
variable declarations and error handling, switching `def` to `fn`, and so on.
By making things like types explicit, dynamism is removed from the program and
the compiler can generate faster code.  The relationship between Mojo and
dynamism has to be carefully managed to meet the goals of the language. The
point of this post is to measure where that relationship stands now, what it
will need going forward as Mojo grows more features, and develop a framework for
managing that relationship.

## Classes and Dynamism

One feature that Mojo lacks at the moment are classes, an inherently dynamic
feature that provides inheritance and runtime polymorphism, the foundations of
object-oriented programming.

Classes are implemented differently in many languages. In Python, for example,
classes are much more flexible than in languages like C++ or Java -- they are
more similar to classes in Javascript or Smalltalk. Methods can be defined and
then deleted, and even conditionally defined! For example, this is valid Python
code:

```python
define = True

class C:
    print("hello") # prints 'hello'
    if define:
        def f(self): print(10)
    else:
        def f(self): print(20)


C().f() # prints '10'
```

In fact, the body of a Python class is just code that is executed, and the
resulting local variables are bound to the attributes of a class object. When
calling a class object, it returns a new object with a reference to the class
object, in which it can perform attribute lookup. In addition, functions that
would be member functions have their first argument bound to the new class
instance through the Python [descriptor
mechanism](https://docs.python.org/3/howto/descriptor.html#invocation-from-a-class).

Mojo adopting the syntax of Python means we have to support the full "hash-table"
dynamism in classes for compatibility with Python, but reference semantic classes
are also important for systems programming and application programming, where this
level of dynamism isn't needed and is actively harmful.  We need to decide how to
handle this.

One approach is to provide a decorator on class definitions (which can be opt-in
or opt-out) to indicate whether the class is "fully dynamic" as in Python or
whether it is "constrained dynamic" (e.g. has virtual methods that may be
overridden but cannot have methods added or removed).

"Constrained dynamic" Mojo classes will use vtables for a more limited but more
efficient constrained dynamism than full hash table lookups.  In addition to raw
lookups, constrained dynamic classes can use "[class hierarchy
analysis](https://dl.acm.org/doi/10.5555/646153.679523)" to devirtualize and
inline method calls, which are not valid for "fully dynamic" classes.

Swift has a similar issue, where the developers wanted to have constrained
dynamism by default but needed full dynamism when working with Objective-C code:
Objective-C is based on the Smalltalk object model and thus has the same issues
as Python.  Swift solved this by adding an opt-in
[@objc](https://swiftunboxed.com/interop/objc-dynamic/) decorator, which
provides full compatibility with Objective-C classes.  Swift implicitly applies
this decorator to subclasses of Objective-C or `@objc` classes for convenience.

If we chose to follow this design in Mojo, we could introduce a `@dynamic`
decorator, in which the class is an instance of a hash table and the body is
executed at runtime:

```python
@dynamic
class C:
    def foo(): print("warming up")
    foo() # prints 'warming up'
    del foo
    def foo(): print("huzzah")
    foo() # prints 'huzzah'
```

We could of course make dynamic be the default, and have a decorator like
`@strict` to opt-in to constrained dynamism as well.  Regardless of the bias, we
absolutely need to support full dynamism to maintain compatibility with Python.

An implementation question here would be "when does the body get executed?" when
the class is defined at the top-level. In this case, the class `C` could be
treated as a global variable with a static initializer that is executed when the
program is loaded. This ties into a discussion about how to treat global
variables and top-level code in general, which will come in a subsequent
section. Naturally, if the class is never referenced, the body is never parsed
and the static initializer is never emitted.

### Syntactic Compatibility and `@dynamic`

A primary goal of Mojo is to [minimize the syntactic
differences](https://docs.modular.com/mojo/why-mojo.html#intentional-differences-from-python)
with Python. We also have to balance that need with what the right default for
Mojo is, and this affects the bias on whether this decorator is "opt-in" or
"opt-out".

We find it appealing to follow the Swift approach by making "full dynamic" an
opt-in choice for a Mojo class. This choice would add another syntactic
divergence between Mojo and Python, but it is one that can be alleviated with an
automatic mechanical transformer from Python code to Mojo code (e.g. to deal
with new keywords we take). In this case, all Python classes will be translated
by sticking `@dynamic` on them, and they can be removed for incremental boosts
to performance.

An alternate design is to require opt-in to "constraint dynamism" by adding a
`@strict` (or use another keyword altogether) for vtable dynamism.  We can
evaluate tradeoffs as more of the model is implemented.

## `def` and Dynamism

In Mojo, the goal of `def` is to provide a syntactic feature set that enables
compatibility with Python. It allows, for example, omitting type annotations,
implicit variable declarations, implicit raises, etc. But the Mojo `def` is not
the same as a Python `def`. A commonly reported issue is that Mojo scoping rules
differ from Python's. In Python, local variables are scoped at the function
level, but Python also supports behavior like:

```python
def foo(k):
    for i in range(k):
        print(i)
    # Mojo complains that `i` is not defined, but this code should compile and
    # dynamically raise an `UnboundLocalError` depending on the value of `k`!
    print(i)
```

Python functions also have a notion of which names are supposed to be bound to
local variables. In the following example, `bar` knows `i` refers to a captured
local variable in `foo`, whereas `baz` tries to retrieve a value for `i` in its
local variable map.

```python
def foo():
    i = 2
    def bar():
        print(i)
    def baz():
        print(i)
        i = 10
    bar() # prints '2'
    baz() # throws an 'UnboundLocalError'
```

This gets at the heart of how Mojo should treat implicitly declared variables in
`def`s. The short answer is: exactly how Python does. `def`s should carry a
function-scoped hash table of local variables that is populated and queried at
runtime. In other words, lookup of implicitly-declared variables would be
deferred to runtime. On the other hand, the function does need to have a notion
of what variable *could* be available in the function, in order to emit
`UnboundLocalError`s as required. Of course, the compiler can optimize the table
away and do all the nice stuff compilers do if possible.

Difficulty arises when discussing `def`s themselves. Although `def`s should
internally support full hashtable dynamism, what kind of objects are `def`s
themselves? For instance:

```python
def foo():
    bar()

def bar():
    print("hello")

foo() # prints 'hello'

def bar():
    print("goodbye")

foo() # should this print 'goodbye'?
```

In Mojo today, the first time the name lookup of `bar` is resolved, it is baked
into a direct call to the first `bar`. Therefore, shadowing of `bar` does not
propagate into the body of `foo`. On the other hand, if all `def`s were treated
as entries in a hashtable, then it would.

One middle-ground approach would be to treat `bar` as a mutable global variable
of type `def()` (one for each possible overload of `bar`). The dynamism can be
escalated with a `@dynamic` decorator that removes static function overloading.
However, both of these approaches risk creating confusing name lookup rules. For
instance, would the following be allowed?

```python
@dynamic
def bar(a): pass

def bar(a: Int): pass
```

This gets into the "levels of dynamism" Mojo intends to provide, and how that
relates to `def`s. The reality is that `def`s in Mojo today only resemble Python
`def`s on the surface. They share similar syntax, but Mojo `def`s are really
extra syntax sugar on top of `fn` and are altogether a different beast than
Python `def`s.

## Four Levels of Dynamism

To summarize, in order to support incremental typing-for-performance, Mojo will
have to support everything from strict, strongly-typed code to full Python
hashtable dynamism but with syntax that provides a gradual transition from one
end to the other.

Given all that has been discussed and what the language looks like today, Mojo's
dynamism is moving into four boxes:

1. Compile-time static resolution.
2. Partial dynamism.
3. Full hashtable dynamism.
4. ABI interoperability with CPython.

The fourth category isn't explored here, but will important when/if we support
subclassing imported-from-CPython classes in Mojo, because that will fix the
runtime in-memory representation to what CPython uses.

The highest level of dynamism and the most faithful compatibility doesn't come
from Mojo itself, it comes from Mojo's first class interoperability with
CPython.  This in effect will be Mojo's escape hatch for compatibility purposes
and is what gives Mojo access to all of Python's vast ecosystem.  Below that,
Mojo will provide an emulation of Python's hash-table dynamism that is a
faithful but not quite identical replication of Python behavior (no GIL, for
example!).  Building this out will be a huge undertaking, and is something Mojo
should do over time.

The most important thing to remember is that Mojo is not a "Python compiler".
The benefit of sharing the same syntax as Python, however, means seamless
interop is on the table:

```python
@python
def python_func(a, b=[]):
    return a + [2] + b

fn mojo_func():
    try:
        print(python_func([3]))
    except e:
        print("error from Python:", e)
```

The goal of the "levels of dynamism" is to provide an offramp, starting by
removing the `@python` decorator from `python_func`.
