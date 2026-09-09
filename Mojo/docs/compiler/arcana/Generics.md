This doc explains all the non-obvious parts of handling generics in the parser.

## `IndexRefAttrInterface`, Depths, and Indexes (IRAIDAI)

These two aliases have equal types.

```mojo
alias A: def[T: AnyType](x: T)->None = ...
alias B: def[Y: AnyType](x: Y)->None = ...
```

It's a bit easier to know this if the names are erased away, and the param-refs
(`x: T`'s `T` and `x: Y`'s `Y`) instead use relative positions (so to speak) to
describe the parameter-decls (`T: AnyType` or `Y: AnyType`) they're referring
to. Something like:

```mojo
alias A: def[_: AnyType](x: *(0,0))->None = ...
alias B: def[_: AnyType](x: *(0,0))->None = ...
```

`IndexRefAttrInterface`s, like those `*(0,0)`s, are made of two parts:

- depth: Which containing signature (`ParameterScopeTypeInterface`,
  `ParameterScopeAttrInterface`) contains the parameter we're referring to.
  Non-negative integer. Zero means the nearest containing signature (like
  above), one means the signature containing that one, etc. Note they cannot
  refer to any op's parameter-decls, and you cannot always use a depth to refer
  to surrounding scopes, see DCRTODS.
- index: index of the parameter decl in that signature (non-negative integer).

Another example, to illustrate a non-zero depth:

```mojo
alias bar: def[
    D: DType,
    N: Int,
    f: def[Y: AnyType](Y, SIMD[N, D])->None
](...) = ...
```

The `SIMD[N, D]`'s `N` is a #kgen.param.index.ref<1, 0> : !lit.struct<@Int>,
(sometimes written as `*(1,0)`) because it's not referring to the containing
signature but instead a signature outside of it. In other words, there is 1
signature between the param-ref and the param-decl it's referring to, so its
depth is 1.

Be careful when calculating that `depth`, it's easy to get wrong.

Note: not all param-refs use indexes and depths. There are also normal
`ParamDeclRefAttr`s which still refer to things by name.

## Depths Cannot Refer To Op-Declared Signatures (DCRTODS)

Given this alias:

```mojo
def foo[X: AnyType](x: X):
    alias bar: def[Y: AnyType](X, Y) = ...
```

Given IRAIDAI, one might assume that `bar`'s type is
`def[AnyType](*(1,0), *(0,0))`, because types generally don't contain
`ParamDeclRefAttr`s. However, this is wrong.

The type is actually `def[AnyType](X, *(0,0))`.

So why does that `Y` mention get replaced with `*(0,0)`, but `X` doesn't?

It's because of this rule:

- When referring to an op's parameter-decl, use `ParamDeclRefAttr`.
- When referring to a type's or attr's parameter-decl, use `ParamIndexRefAttr`.

Here are the parameter-decls involved:

- `X` is declared by the `def foo` op.
- `bar` is declared by the `alias bar` op.
- `Y: AnyType` is declared by the `fn` type.

`X` and `bar` are declared by ops, so we use `ParamDeclRefAttr`s to refer to
them.

`Y` is declared by a type, so we use a `ParamIndexRefAttr` to refer to it.

Rule of thumb: depths can only refer to `GeneratorType` and `GeneratorAttr`s
that contain them. For everything else (like the `X`), use `ParamDeclRefAttr`.

## ParameterScopeTypeInterface Affects IndexRefAttrInterface Depths (PSTIAIRAID)

Recall how `IndexRefAttrInterface`'s `depth` field is a number that describes
*which* surrounding signature's param-decl a param-ref is referring to (see
IRAIDAI).

If a `depth` is 0, then it's referring to the nearest enclosing signature, like
the `T` in this:

```mojo
alias A: def[T: AnyType](x: T)->None = ...
```

If a `depth` is 1, then it's referring to the signature containing that one,
like the `N` in the `SIMD[N, D]` in this:

```mojo
alias bar: def[
    D: DType,
    N: Int,
    f: def[Y: AnyType](Y, SIMD[N, D])->None
](...) = ...
```

To manage this, we make all signatures inherit from
`ParameterScopeTypeInterface`, and our code specifically watches for
`ParameterScopeTypeInterface` when dealing with depths.

For example, if we're looking for all mentions of `T` in this Mojo snippet:

```mojo
alias bar: def[
    T: AnyType,
    L: List[T],
    f: def[Y: AnyType](Y, List[T])->None
](...) = ...
```

...which is interpreted like this:

```mojo
alias bar: def[
    _: AnyType,
    _: List[*(0,0)],
    _: def[_: AnyType](*(0,0), List[*(1,0)])->None
](...) = ...
```

...we can't just look deeply for `*(0,0)`. If we did, we'd find the first `T`
mention (good) and the `Y` mention (bad).

Instead, as we're looking, we need to keep track of "how many signatures deep"
we are as we're searching. While looking in the outermost signature, we should
look for `*(0,0)`s, but when our search dives into the inner signature, we
should then be looking for `*(1,0)`s.

One might call this "depth-aware searching".

This is particularly important when looking inside anything that might contain
a signature, even indirectly.

For this reason, think twice when using `AttrTypeReplacer` or `AttrTypeWalker`,
and consider using something like `IndexParameterReplacer`, `ParameterReplacer`,
`ParameterEvaluator`, `ParserParameterEvaluator`, or `IndexRefRemapper` which
are all depth-aware.

For example, from `IndexParameterReplacer`'s comments:

```mojo
/// Handing this `depth` to replaceImpl is the main point of this class,
/// it enables the replaceImpl implementation to know how deep into
/// signature scopes we currently are in our recursive walk.
/// For example, they can check `depth == 0` to know if they're in the
/// original scope, or they can check that an
/// `indexRef.getDepth() == depth` to know if that indexRef is referring to
/// a parameter-decl from the original scope.
```

They have special logic to handle the different depths, described a little in
STCHDDDOS.

**Also, beware: MLIR's built-in `.walk` method IS NOT depth-aware!**

## Same Type Can Have Different Depths Depending On Scope (STCHDDDOS)

Looking at the example from PSTIAIRAID:

```mojo
alias bar: def[
    T: AnyType,
    L: List[T],
    f: def[Y: AnyType](Y, List[T])->None
](...) = ...
```

...which is interpreted like this:

```mojo
alias bar: def[
    _: AnyType,
    _: List[*(0,0)],
    _: def[_: AnyType](*(0,0), List[*(1,0)])->None
](...) = ...
```

Notice how `List[T]` now appears as two different types:

- `List[*(0,0)]`
- `List[*(1,0)]`

...even though they're the same type.

This means that **equal types are not always pointer-comparable.**

...which is ironic, since that was the whole point of depths (per IRAIDAI).

This needs special handling in various places, search STCHDDDOS.

When comparing these two, we'll need to either:

- Decrement the depths of things in deeper signatures. In other words, decrement
  the depth in `List[*(1,0)]`.
- Increment the depths of things in outer signatures. In other words, increment
  the depth in `List[*(0,0)]`.

Only then can you compare the two types to see if they're equal.

One key place to watch out for is when instantiating a parameterized value. For
example (STCHDDDOS-A), given a generator value `<index> *(0,0) + 1` (Given an
index typed value, return one plus that value), we can bind an index reference
from the outer scope such as `bind_params(<index> *(0,0) + 1, *(0,1))`. Folding
this `bind_params` operator gets us a new generator value that has zero input
parameters `<> *(1,1) + 1`. It's important to note that this is still a
generator value. An explicit "instantiation" step is required to unwrap the
generator, and this unwrap step will require decrementing all index references
in the generator body as it is now under one fewer level of generator scopes. A
similar situation exists with the `apply` operator when the callee is an
instantiated parametric function (STCHDDDOS-B). The resulting function type
needs to have its index references decremented (also referred to us
"up-binding").
