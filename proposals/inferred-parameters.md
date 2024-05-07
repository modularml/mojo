# Inferring Parameters from Other Parameters

A common feature in programming language with generics is the ability to infer
the value of generics/templates/parameters from the argument types. Consider
C++:

```cpp
template <typename T>
void inferMe(T x) {}

int x = 1;
inferMe(x);
// Equivalent to
inferMe<int>(x);
```

Mojo is a parametric language and also supports this feature in a variety of use
cases that make code significantly less verbose:

```python
fn infer_me[dt: DType, size: Int](x: SIMD[dt, size]): pass

infer_me(Int32())
# Equivalent to
infer_me[DType.int32, 1](Int32())
```

But Mojo pushes these needs a step further. As a language that encourages heavy
parameterization, dependent types are very common throughout the language.
Consider:

```python
fn higher_order_func[dt: DType, unary: fn(Scalar[dt]) -> Scalar[dt]](): pass

fn scalar_param[dt: DType, x: Scalar[dt]](): pass
```

Language users commonly encounter cases where dependent types could infer their
parameter values from other parameters in the same way from argument types.
Consider `scalar_param` in the example above: `dt` could be inferred from the
type of `x` if `x` were passed as an argument, but we have no syntax to express
inferring it from `x` as a parameter since the user is required to pass `dt` as
the first parameter.

```python
scalar_param[DType.int32, Int32()]() # 'dt' parameter is required
```

This has been requested multiple times in various forms, especially given the
new autoparameterization feature. The current tracking feature request:

- <https://github.com/modularml/mojo/issues/1245>

## Proposal

In the above example, we want to be able to infer `dt` instead of explicitly
specifying it:

```python
scalar_param[Int32()]()
```

Laszlo Kindrat and I proposed several options to remedy this and members of the
“Mojo Language Committee”  met to discuss these ideas, summarized below.

We decided to move forward with the following option. Mojo will introduce a new
keyword, `inferred`, as a specifier for parameters only. `inferred` parameters
must precede all non-inferred parameters in the parameter list, and they
**cannot** be specified by a caller — they can **only** be inferred from other
parameters. This allows us to express:

```python
fn scalar_param[inferred dt: DType, x: Scalar[dt]](): pass

scalar_param[Int32()]() # 'dt' is skipped and 'Int32()' is bound to 'x'
```

Where `dt` is inferred from `x`. The decision to choose a keyword instead of
introducing a new punctuation character [like Python does for keyword-only
arguments](https://docs.python.org/3/tutorial/controlflow.html#special-parameters)
is because a keyword clearly indicates the intent of the syntax, and it’s easy
to explain in documentation and find via internet search.

## Aside: Inferring from Keyword Parameters

Related but separate to the proposal, we can enable parameter inference from
other parameters using keyword arguments. This allows specifying function (and
type) parameters out-of-order, where we can infer parameters left-to-right:

```python
scalar_param[x=Int32()]() # 'dt' is inferred from 'x'
```

We should absolutely enable this in the language, since this does not work
today. However, with respect to the above proposal, in many cases this still
ends up being more verbose than one would like, especially if the parameter name
is long:

```python
scalar_param[infer_stuff_from_me=Int32()]()

# One would like to write:
scalar_param[Int32()]()
```

So this feature is orthogonal to the `inferred` parameter proposal.

## Alternatives Considered

Several alternative ideas were considered for this problem.

### Non-Lexical Parameter Lists

This solution would alter the name resolution rules inside parameter lists,
allowing forward references to parameters within the same list. The above
example would be expressed as:

```python
fn scalar_param[x: Scalar[dt], dt: DType](): pass
```

Where any parameter is inferable from any previous parameter. The benefits of
this approach are that the order of parameters at the callsite match the order
in the declaration: `scalar_param[Int32()]()`

This alternative was rejected because:

1. Non-lexical parameters are potentially confusing to users, who normally
    expect named declarations to be lexical. Relatedly, we are moving towards
    removing non-lexical parameters in general from the language.

2. This would incur a huge implementation burden on the compiler, because the
    type system needs to track the topological order of the parameters.

### New Special Separator Parameter

This solution is fundamentally the same as the accepted proposal, but differs
only in syntax. Instead of annotating each parameter as `inferred`, they are
separated from the rest using a new undecided sigil (`%%%` is a placeholder):

```python
fn scalar_param[dt: DType, %%%, x: Scalar[dt]](): pass
```

The benefit of this approach is this matches the [Python
syntax](https://docs.python.org/3/tutorial/controlflow.html#special-parameters)
for separating position-only and keyword-only parameters. It also structurally
guarantees that all infer-only parameters appear at the beginning of the list.

This alternative was rejected because:

1. There was no agreement over the syntax, and any selected sigil would
    introduce additional noise into the language.

2. `inferred` clearly indicates the intent of the syntax, and can be found via
    internet search, and is overall easier to explain syntax than introducing a new
    argument separator.

### Special Separator Parameter at the End

This is a variation on the above, where the infer-only parameters would appear
at the end of the parameter list, and subsequent parameters would be allowed to
be non-lexical:

```python
fn scalar_param[x: Scalar[dt], %%%, dt: DType](): pass
```

The benefit of this approach is that the parameters appear in the same position
at the callsite. This alternative was rejected for a combination of the reasons
for rejecting a new separator and non-lexical parameters.

### Segmented Parameter Lists

This proposal would allow functions to declare more than one parameter list and
enable right-to-left inference of the parameter “segments”. The above would be
expressed as:

```python
fn scalar_param[dt: DType][x: Scalar[dt]](): pass
```

The callsite would look like

```python
scalar_param[Int32()]()
```

And call resolution would match the specified parameter list to the last
parameter list and infer `dt`. This proposal was rejected because

1. The right-to-left inference rules are potentially confusing.

2. This is an overkill solution to the problem, because this opens to door to
arbitrary higher-order parameterization of functions.
