# Lazy Mojo Parsing: Static vs. Dynamic

## Motivation

The core design goals of Mojo that influence the design of the parser are (1) to
be "just like Python" but (2) to extend the Python programming model and its
behaviour into the compile time domain to provide incremental performance and
enable new features, like metaprogramming and static function overloading.

Python, for example, does not require forward declarations of functions used
before they are defined: the function referenced is looked up in the globals map
dynamically when the relevant code executes:

```mojo
def foo(): # define a global object of 'function' type called 'foo'
    bar()  # lookup 'bar' in the '__globals__' map and call it

def bar(): pass # define a global object called 'bar'

foo() # lookup 'foo' in the globals map and call it
```

Python essentially performs name resolution of `bar` at runtime using the state
of the globals table when `foo` is invoked. Mojo needs to support the same
behaviour but be able to bring the name resolution up to compile time, for
instance, so that `bar()` can be emitted as a direct call!

Let's take a look at how Mojo's parser achieves this today and some the current
issues with it. We will also propose next steps for the parser and Mojo to
expand the supported feature set and more closely align with Python.

### Current Implementation Approach

Mojo's parser is inherently lazy. In simple terms, Mojo source code is parsed
only as needed. A lazy parser is an architecturally important feature of Mojo.
It enables more responsive language tooling to be built on top of the language,
such as diagnostics, go-to-definition, and fix-its. It makes parsing large
amounts of source code efficient, especially when there are many imported
packages, in contrast to a language like C++, where a huge amount of code in
header files must be parsed before parsing each source file. And it is a
reflection of the semantics of Python name lookup and resolution.

Mojo's lazy parsing is especially powerful because of significant whitespaces.
One of the advantages of significant whitespace is that the body of a
declaration -- a function, for example -- can be skipped without even lexing it
(with a few exceptions, such as multi-line strings). This means huge amounts of
declarations can be processed and registered in a name table with little effort.

### Three-Phase Parsing and Type Checking

Today, Mojo implements a 3-phase parser, in which the name, signature/type, and
body/value of a declaration are incrementally resolved as needed. In Mojo
parlance, a "declaration" is a named language thing that has a type and can be
referenced in Mojo code: modules, functions, structs, and variables. Concretely,
when Mojo parses the body of a declaration, only the names of contained
declarations are parsed and registered in a name table. Referencing a
declaration requires only their type to be parsed. And finally, all bodies of
referenced declarations are parsed to complete IR generation.

Many features naturally fall out of this system. Recursive imports and automatic
forward declaration, for example, work because referencing a declaration does
not require parsing its body. It also allows Mojo alias declarations to be
parsed non-lexically. For example, consider a pair of recursive functions:

```mojo
def foo():
    bar()

def bar():
    foo()
```

The signatures of the functions are parsed first and the bodies are resolved,
later, allowing the functions to reference each other without forward
declarations.
The same 3-phase parsing mechanism is used throughout different levels of Mojo
code, including in "imperative" regions like function bodies: the declarations
of variables, nested functions, and nested structs are parsed first, before the
full body of the function is resolved. Consider this example:

```mojo
def foo():
    var x = bar()

    def bar() -> Int:
        return 10

    print(x)
```

This code seems non-sensical: `x` is defined by calling `bar`, even though `bar`
is defined after `x`! This works, however, because the definitions of
declarations even inside a function body are lazily resolved. The reality is
that imperative regions like function bodies have different considerations,
because e.g. they can capture values and we
would like to be able to reference variables within imperative code before the
function body is fully resolved.

### What Doesn't Work

Three-phase resolution leads to several challenges with the current design. The
3-phase parser on the surface achieves much of the desired behaviour of the
parser, but in reality it's held together by a bunch of hacks:

- Stray expressions in top-level code or inside struct bodies can give terrible
  "recursive declaration" errors.
- The parser relies on an IR dominance check to ensure regular variables are
  not used before they are defined.
- Scopes for explicitly declared local variables within lexical blocks require
  nested "dummy" declarations to work, rather than a simple scoped hash table.

It is also not fully consistent with how Python works, and in fact actively
impedes compatibility with Python; that Mojo scoping rules differ from Python's
is a widely reported issue. In Python, local variables are scoped at the
function level, but Python also supports behaviour like:

```mojo
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

```mojo
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
`def`s. The short answer is: exactly how Python does. Mojo also needs to support
structs nested inside other structs and functions, fully dynamic Python classes,
and global variables.

## Local Variable Scoping and Implicitly Declared Variables

To solve these problems, we will move to a design where
code will be parsed imperatively with a scoped hash table
but certain declarations will still be resolved lazily. Importantly, explicit
`var` and `let` declarations will no longer be parsed lazily: lazy parsing of
these declarations doesn't make sense because they represent imperative code
with lexical scoping (concepts that don't exist in Python).

This means `var` and `let` declarations are simply residents of a scoped hash
table within each function body, naturally allowing shadowing while rejecting
use-before-def:

```mojo
def foo(a: Bool):
    var x = 10
    if a:
        let x = "hello"
        print(x)
```

This also composes with lazy parsing of nested functions and captures by
correctly rejecting references to variables that are not yet defined:

```mojo
def foo():
    def bar():
        print(a)

    bar() # error: 'a' is not defined
    var a = 10

def foo():
    def bar():
        print(a)

    var a: Int
    bar() # parses, but the lifetime checker will complain later
```

In these cases, `var` and `let` declarations are statically resolved by the
parser. But what about implicitly declared variables (which as you may recall,
are only allowed inside `def`s). Well, just like Python, `def`s need to carry a
function-scoped hash table of local variables that is populated and queried at
runtime. In other words, lookup of implicitly-declared variables is deferred to
runtime. On the other hand, the function does have a notion of what variable
*could* be available in the function, in order to emit `UnboundLocalError`s as
required. Of course, the compiler can optimize the table away and do all the
nice stuff compilers do if possible.

### Improving Dynamic Type Compatibility

Given the stronger guarantees and solidified model for variables explicitly
declared with var/let, we also need to address implicitly-declared variables in
`def` functions, which are constrained by compatibility with existing Python
code. Mojo needs to default implicitly-declared variables to be
`object` type, so that they can be reassigned to values of any type:

```mojo
def foo():
    a = "hello"
    a = 10
    a = [2.3]
```

This fits into Mojo's incremental performance story: implicitly declared
variables are fully dynamic and provide huge amounts of expressibility, but to
get guaranteed performance, programmers can add explicit `var` and `let`
declarations. Of course, we can also use MLIR optimizations to "unbox"
dynamically typed values in simple cases without loss of generality as well.

As an aside, while Python supports type annotations on implicitly-declared
variables, they need to be ignored for compatibility with Python. For example,
Python happily accepts an expression as a type annotation, including things like
the following code:

```python
x: 1 + 2 = 42  # The type is 1+2??
print(x)  # prints '42'
```

Together, this will give Mojo proper scoping rules for explicitly-declared
variables, fully dynamic and Python-compatible behaviour for implicitly-declared
variables in `def`s, while maintaining the lazy parsing of function bodies,
structs, and modules.

## Dynamic Classes

Python classes are much more flexible than a language like C++ or Java, they
are more similar to classes in Javascript or Smalltalk. Methods can be defined
and then deleted, and even conditionally defined! For example, this is valid
Python code:

```python
define = True


class C:
    print("hello")  # prints 'hello'
    if define:

        def f(self):
            print(10)
    else:

        def f(self):
            print(20)


C().f()  # prints '10'
```

In fact, the body of a Python class is just code that is executed, and the
resulting local variables are bound to the attributes of a class object. When
calling a class object, it returns a new object with a reference to the class
object, in which it can perform attribute lookup. In addition, functions that
would be member functions have their first argument bound to the new class
instance.

In Mojo, we need to support full "hash-table" dynamism in classes for
compatibility with Python, but reference semantic classes are also important for
systems programming and application programming, where this level of dynamism
isn't needed and is actively harmful. We need to decide how to handle this.

One approach is to provide a decorator on class definitions (which can be opt-in
or opt-out) to indicate whether the class is "fully dynamic" as in Python or
whether it is "constrained dynamic" (e.g. has virtual methods that may be
overridden but cannot have methods added or removed).

"Constrained dynamic" Mojo classes will use vtables for a more limited but more
efficient constrained dynamism than full hash table lookups. In addition to
raw lookups, constrained dynamic classes can use "[class hierarchy
analysis](https://dl.acm.org/doi/10.5555/646153.679523)" to devirtualize and
inline method calls, which are not valid for "fully dynamic" classes.

Swift has a similar issue, where the developers wanted to have constrained
dynamism by default but needed full dynamism when working with Objective-C code:
Objective-C is based on the Smalltalk object model and thus has the same issues
as Python. Swift solved this by adding an opt-in
[@objc](https://swiftunboxed.com/interop/objc-dynamic/) decorator, which
provides full compatibility with Objective-C classes. Swift implicitly applies
this decorator to subclasses of Objective-C or `@objc` classes for convenience.

If we chose to follow
this design in Mojo, we could introduce a `@dynamic` decorator, in which the
class is an instance of a hash table and the body is executed at runtime:

```mojo
@dynamic
class C:
    def foo(): print("warming up")
    foo() # prints 'warming up'
    del foo
    def foo(): print("huzzah")
    foo() # prints 'huzzah'
```

We could of course make dynamic be the default, and have a decorator to opt-in
to constrained dynamism as well. Regardless of the bias, we absolutely need to
support full dynamism to maintain compatibility with Python.

The next question is "when does the body get executed?" when
the class is
defined at the top-level. In this case, the class `C` could be treated as a
global variable with a static initializer that is executed when the program
is loaded. This ties into a discussion about how to treat global variables and
top-level code in general, which will come in a subsequent section. Naturally,
if the class is never referenced, the body is never parsed and the static
initializer is never emitted.

### Syntactic Compatibility and `@dynamic`

A primary goal of Mojo is to
[minimize the syntactic differences](https://mojolang.org/docs/why-mojo.html#intentional-differences-from-python)
with Python. We also have to balance that need with what the right default for
Mojo is, and this affects the bias on whether this decorator is "opt-in" or
"opt-out".

We find it appealing to follow the Swift approach by making "full dynamic" an
opt-in choice for a Mojo class. This choice would make Mojo not strictly a
superset of Python (because you may need to add the decorator in some cases),
but it is already a non-goal of Mojo is syntactic compatibility with
Python. Instead, we expect to provide an automatic mechanical transformer from
Python code to Mojo code (e.g. to deal with new keywords we take). In this case,
all Python classes will be translated by sticking `@dynamic` on them, and they
can be removed for incremental boosts to performance.

An alternate design is to require opt-in to "constraint dynamism" by adding a
`@strict` (or use another keyword altogether) for vtable dynamism. We can
evaluate tradeoffs as more of the model is implemented.

### Initialization of Struct and Constrained-Dynamic classes

While we need full compatibility for fully-dynamic classes, and thus must
execute struct initializers in a fully dynamic way, we still need to decide how
initializers work for structs and constrained dynamic classes, consider this
example:

```mojo
struct C:
    def foo(): print("duh what")
    foo() # prints 'duh what'
    del foo
    def foo(): print("huzzah")
    foo() # prints 'huzzah'
```

What does this code do? How can Mojo support this dynamism while
retaining performance and compiler guarantees?

In the spirit of moving runtime computation to compile time, as is the theme of
Mojo's metaprogramming system and the compile-time lookup resolution of
explicitly-declared variables, the body of the struct should be evaluated at
compile-time by the Mojo interpreter, and the result is that it populates the
name table of the struct. The interpreter could even be allowed to call `print`
if necessary. At the end of the body, the parser will have a fully formed and
static struct for the rest of the code.

The concept of the bodies of structs being parsed and interpreted to be resolved
is architecturally satisfying, but is ultimately an implementation/feature set
detail of how Mojo structs work. It is orthogonal to the overall purpose of this
document.

### Four Levels of Dynamism

To summarize, in order to support incremental typing-for-performance, Mojo will
have to
support everything from strict, strongly-typed code to full Python hashtable
dynamism but with syntax that provides a gradual transition from one end to the
other.

Given all that has been discussed and what the language looks like today,
Mojo's dynamism is moving into four boxes:

1. Compile-time static resolution.
2. Partial dynamism.
3. Full hashtable dynamism.
4. ABI interoperability with CPython.

The fourth category isn't explored here, but will important when/if we support
subclassing imported-from-CPython classes in Mojo, because that will fix the
runtime in-memory representation to what CPython uses. This work is not
explored in this document.

## Name Shadowing and Dynamism

A feature request to allow shadowing of `let` bindings within function bodies
was posted in <https://github.com/modular/mojo/issues/5>. This begs a larger
question of, given lexically parsed but lazily resolved declarations, whether
lexical name shadowing should be allowed for anything, even functions and
structs:

```mojo
def foo():
    print(10)

foo() # prints '10'

def foo():
    print(20)

foo() # prints '20'
```

This extension would be low-cost at the risk of creating potentially confusing
code to read, and it would match the desired behaviour of Mojo in the REPL
environment. Allowing this kind of redefinition is a feature built in to the
Mojo REPL implementation (the backend to the Jupyter notebook interface) but it
comes with a fair share of sharp edges because it lacks first-class language
support.

Difficulty arises when discussing `def`s themselves. Although `def`s should
internally support full hashtable dynamism, what kind of objects are `def`s
themselves? For instance:

```mojo
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

A middle-ground approach would be to treat `bar` as a mutable global variable
with type `def()`, with escalated dynamism if tagged with `@dynamic`. This gets
into the "levels of dynamism" Mojo intends to provide.

## Top-Level Code

Top-level code has two "modes" in Mojo: in standalone Mojo files and in the
REPL. This section is concerned with the former mode.

Top-level code in Mojo is in a weird place mainly because of what it represents
and the consequences of being a compiled language, needing things like symbol
tables, data sections, and so on. There are interesting questions like how to
support Python-style globals mixed in top-level code, how imports work, and how
globals work.

One idea is that top-level could *should be treated the same* as any other body
of code in the program. The semantics of a self-contained program should be the
same if the top-level code is wrapped inside an `def main()`. What differs is
how declarations in top-level code are represented in the generated IR: using
symbols instead of stack-allocated variables or parameters. For example, Mojo
could support the following:

```mojo
def foo():
    _ = x

var x = 10

for i in range(10):
    x += i

if __name__ == "__main__":
    foo()
```

Following the same lazy resolution rules as within a function body, the
declaration of `foo` is registered by the body is not parsed. Then, the
definition of `x` is fully parsed along with the top-level `for` loop and the
conditional call to `foo`. That call causes the signature of `foo` to be
resolved. Finally, the body of `foo` is resolved and the reference to `x` is
replaced with a reference to a global. The dynamic top-level code is inserted
into an initializer that is run when the file is executed.

Although `foo` is really a closure, in top-level code, it gets promoted to a
function with captured variables replaced with global value references. The
dynamic initializer code is placed inside an entry point in the module.

More difficult questions arise when mixing fully dynamic `def`s and static
`def`s in the same name scope:

```mojo
@dynamic
def foo(a): # this 'foo' is dynamically name bound in a global hashtable
    print(a)

def foo(a): # this 'foo' is statically name bound and overloadable
    print(a)

@dynamic
def bar():
    foo(10) # which 'foo' does this call?
```

This same problem will exist inside the body of a function, if top-level code
should be treated the same as a function body. So, this must be solved
regardless of the semantics of top-level code. On solution would be for the
parser to look for a statically matching function to call, and if that fails
emit a hashtable lookup when in a `@dynamic` `def`.

Global variables are another feature that have to be developed regardless. One
of the user-visible sharp edge right now is that functions cannot access
variables declared directly in cells:

```mojo
# [1]
a = 10

# [2]
def foo():
    print(a) # error: 'a' is not defined
```

Given this, it's likely top-level code will stay as the status quo in the
near-term as these other features are built out.

## Putting Everything Together

Consider a more involved example (assume `print` is builtin magic). This could
be one outcome where top-level code is treated like a function body, mixing
fully-dynamic objects and static ones:

```mojo
# foo.mojo
def foobar() -> Int:
    return 10

var abc = foobar()

@dynamic
class C:
    print("woah")
    def f(self): print("huzzah")

def baz() -> Int:
    return 11

@value
struct S:
    var y: Int
```

```mojo
# main.mojo
import foo as M

print(foo.abc)
foo.abc += M.baz()

def main():
    M.C().f()
    print(foo.abc)

if __name__ == "__main__":
    main()
```

1. The parser starts in `main.mojo` and processes the import statement. It goes
   to parse `foo.mojo`, skipping over the body of `foobar` and parses the
   initialization of the global `abc` to a function call to `foobar`. The
   signature of `foobar` gets resolved.

2. Because dynamic classes are the same as global variables with an initializer,
   the body of `C` is immediately parsed and the initialization code is inserted
   into the module of `foo`.

3. The declaration of `baz` is registered by the body is skipped. Same for `S`.

4. The module `foo` is mapped into the name table of the main file as `M` but
   the value of the module is a special `MPValue`, which represents a partially
   resolved module value. This allows modules to be imported without fullying
   resolving everything inside them. The initializer of `foo` is called at the
   import statement.

5. Top-level code like `print(foo.abc)` is parsed and placed inside the
   initializer of `main.mojo`. The reference to `abc` resolves to a global
   symbol.

6. The attribute reference `M.baz` on the `MPValue` causes the signature of
   `baz` to be resolved.

7. The declaration of `main` is registered but the body is not parsed.

8. The conditional code `__name__ == "__main__"` is parsed and the signature of
   `main` gets resolved.

9. The bodies of `foobar`, `baz`, and `main` are fully resolved. `S` remains
   unresolved because it was not referenced.

When `main.mojo` is compiled as an executable (and not a library), `__name__`
is set to `"__main__"` and the initializer code is placed inside a `main`
function.

The lynchpin here is mixing lazy resolution with imperative parsing:
declarations are lazily resolved using the current state of the name table when
code is parsed imperatively.

### Dynamic Modules

Mojo will allow modules to be dynamically instantiated, creating a runtime
dictionary value with all module elements mapped in. This enables backwards
compatibility with Python, enabling code like:

```mojo
import foo

let M = foo # MPValue is fully resolved and becomes an RValue
```

However, when a module contains static types like structs or non-`@dynamic`
classes -- language constructs that lack runtime representations --
dynamic instances of the module will lack entries for these items. This will
allow any module ported from Python to be dynamic while enabling said module
to be incrementally ported to Mojo.

### Nesting `fn` and `def`

What should the behaviour be when nesting an `fn` inside a `def`?

```mojo
@dynamic
def foo():
    def bar():
        print(a)

    a = 10
    bar()
```

`fn`s lack the dynamic lookup behaviour of `def`s, so it can be argued that this
code should not be allowed: it would emit an `'a' is not defined` error. On the
other hand, it can be argued that the local variables of `foo` can be implicitly
captrued by `bar`. It could go either way, and the resulting behaviour is likely
not super important. Whatever the behaviour that falls out of the implementation
can be the one Mojo goes with.

## Action Items

1. The parser should make a small but significant architectural change: looking
   up declarations in the name table of a parent declaration should no longer
   require the parent declaration to be fully resolved: it should use the state
   of the name table when the reference is encountered imperatively.

2. The initializers of `var` and `let` declarations are parsed imperatively
   instead of lazily. Only declarations with suites should be parsed lazily:
   structs and functions.

3. In the short-term, `alias` declarations can be made lexical
   *if this is easier*. Non-lexical aliases are not used very frequently
   anyways. This can always be added back when the system settles down more.

4. Parsing imperative code should track a scoped hash table in the parent
   declaration for resolving child declarations. This includes nested functions.
   A nested `fn` should be treated like a local variable, but `def`s within a
   `def` perhaps should be treated like implicitly-declared variables, for
   compatibility with Python.

    ```mojo
    def foo(a: Bool):
        if a:
            def foobar() -> Int: return 10
        else:
            def foobar() -> Int: return 20

        foobar() # error: 'foobar' is not defined

    def bar(a):
        if a:
            def foobar(): return 10
        else:
            def foobar(): return 20

        foobar() # ok
    ```

5. All these things combined will allow many of the aforementioned hacks to be
   removed from the parser. The next step will be tackling top-level code and
   proper lazy resolution of modules. In parallel, implicitly-declared variables
   in `def`s can be changed to a dynamic hash table and basic dynamic classes
   can be built out.

6. The parser should treat top-level code like the body of any function or
   (dynamic) class, with the difference being that functions and variables are
   emitted to global symbols, instead of `lit.varlet.decl`. On the other hand,
   the parser can just emit code as usual and a subsequent pass will transform
   the IR into a symbol-based one.

7. The parser should emit calls to module initializers during import and
   implement an `MPValue` (name subject to change) that allows elements of a
   module to be lazily resolved. Concrete resolution of declarations will begin
   from top-level code.
