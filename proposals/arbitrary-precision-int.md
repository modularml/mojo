## Summary

We propose standardizing an arbitrary precision runtime signed integer type, `BigInt`, with the aim of using this to build an arbitrary precision integer variant of `object`.

## Motivation

Mojo intends to be a superset of Python, so it should have the same semantics on working python code when possible. Having arbitrary precision by default is also useful in explorative coding, where it might not be clear yet how big the integers can get. Integer overflow is also easy to encounter as a beginner, since many online exercises use functions like the factorial and fibonacci functions, which can easily overflow a 64 bit integer for small inputs. Making integers behave as close to mathematical integers by default as possible reduces surprises in these contexts.

There are also contexts where arbitrary precision integers are just necessary to support dynamic APIs, such as in abstract algebra.

## Implementation guidance

`BigInt` should support the same arithmetic API as `Int`, conversion from `IntegerLiteral` and `Int`, and fallible conversion to `Int`. The internal representation should be `List[Int]`, where the least significant words are stored at a lower index.

The implementation should prioritize simplicity over performance, at least initially.

## Drawbacks

Implementing and maintaining `BigInt` has a cost, and it will likely be significantly less performant than implementations in other languages, at least initially.

## Alternatives

One alternative is using `PythonObject` to represent integers. Flexible support for use of python libraries will likely require `object` to have a `PythonObject` variant anyways, so there would be little additional cost to also supporting integers this way. The drawback of this alternative is that it would make python a runtime dependency for a lot of very simple Mojo programs down the line, even those that do not want to directly interact with existing python libraries.

Another alternative is to use an existing arbitrary precision integer library from another performant compiled language. This would make it easier to provide a performant implementation. The drawbacks are that if the library is in a language not yet used by the compiler this would require adding another compiler as a dependency to the toolchain, and even if the library is in C++ it would still add a dependency to the toolchain. Some libraries, such as GMP, could cause licensing issues.

## Future Possibilities

- Make the integer variant in `object` arbitrary precision, possibly falling back to a non-allocating version for small integers.
- Make integers, at least in `def`s, realize to `object` by default.
