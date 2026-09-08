# Bounds checking and removing negative indexing

**Status**: Implemented.

## Background

Negative indexing prevents us from enabling bounds checking by default. Without
bounds checking, indexing bugs become silent memory corruption. This is the core
motivation for removing it.

Today, collections allow negative indexing e.g. `x[-1]` to get the last element.
This requires every index operation to normalize negative values (simplified):

```python
def __getitem__(self, var i: Int):
    if i < 0:
        i += len(self)
```

On CPU, this branch is cheap but not free. On GPU it's actively harmful, GPU
architectures execute in lockstep warps/wavefronts, and divergent branches cause
both paths to execute serially. An extra branch on every element access in a
tight kernel is a measurable performance regression.

More critically, on GPU we disable bounds checking for this exact reason,
branching overhead is unacceptable on hot paths. This means a negative index on
GPU won't get normalized, it will silently access out-of-bounds memory. If users
expect CPU-style negative indexing to work on GPU, they get undefined behavior
with no error and no warning.

Removing negative indexing lets us enable bounds checking by default on CPU with
a single, clean invariant: any index outside `[0, len-1]` is invalid. This
catches a broad class of bugs, including accidental negative indices, with zero
additional overhead beyond the bounds check itself. On GPU, bounds checking is
folded out entirely, and the indexing path is branchless.

The ergonomic cost is minimal, `x[-1]` becomes `x[len(x) - 1]`, and we can add a
`.last()` or `.from_end(i)` method if desired. In GPU kernels, indices are
almost never hardcoded literals. They're computed from thread IDs, block
dimensions, and strides, so negative indexing provides zero benefit there.

Note this design does not impact MAX ops or kernel code. Any op that supports
negative indexing will continue to do so. It's only for subscripting on our
stdlib collections, to standardize bounds checking so that users get the
behavior they expect, with optimal performance tradeoffs.

## Proposed Design

### Compile Time

The vast majority of negative indices are literals. Mojo distinguishes
compile-time integer literals from runtime integers, allowing us to add a
compile-time check via comptime assert. The error shows the indexing call site
directly above the error:

```mojo
def main():
    var x = [1, 2, 3]
    print(x[-1])
```

```text
/tmp/main.mojo:3:12: note: call expansion failed
    print(x[-1])
            ^
constraint failed: negative indexing is not supported, use e.g. `x[len(x) - 1]` instead
```

This makes migration easy and fast, users get the file and line number for each
call site they need to update, with an error describing how to do it.

### Runtime

Negative runtime index values are much less common than literals such as `-1`.
Rather than adding a separate check for them, bounds checking handles it
naturally. The `UInt` conversion in the bounds check turns negative values into
large unsigned values that fail the `< len` comparison, so negative indices are
rejected for free, with no extra branch:

```mojo
def main():
    var x = [1, 2, 3]
    var i = -1
    print(x[i])
```

```text
At: /tmp/main.mojo:4:12: Assert Error: index -1 is out of bounds, valid range is 0 to 2
```

For cases where a negative runtime index was intentional (e.g. a sliding window
offset), the error message and call site make it clear how to migrate: change
`x[offset]` to `x[len(x) + offset]`.

### Scope

The following collections will be updated to have bounds checking by default on
CPU, and the compile-time check for negative `IntLiteral` indices:

- List
- Span
- InlineArray
- String
- StringSlice
- Deque
- IntTuple

`PythonObject` for interop with Python will remain unchanged and retain the same
behaviour.

This work should be completed before Mojo 1.0 to avoid breaking changes.

## Implementation

Remove `normalize_index` from each collections `__getitem__` method, using
`check_bounds` instead. For example in `List`:

```mojo
struct List:
    @always_inline
    def __getitem__(ref self, idx: Some[Indexer]) -> ref[self] Self.T:
        check_bounds(idx, len(self), call_location())
        return (self._data + idx)[]

    # Comptime error to help with migration off negative indexing
    @always_inline
    def __getitem__(ref self, idx: IntLiteral) -> ref[self] Self.T:
        comptime assert IntLiteral[idx.value]() >= 0, \
        "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        check_bounds(idx, len(self), call_location())
        return (self._data + idx)[]
```

Where `check_bounds` is:

```mojo
@always_inline
def check_bounds[
    cpu_default: Bool = True
](idx: Some[Indexer], size: Int, loc: SourceLocation):
    debug_assert[
        assert_mode="safe" if cpu_default and not is_gpu() else "none"
    ](
        UInt(index(idx)) < UInt(size),
        "index ",
        index(idx),
        " is out of bounds, valid range is 0 to ",
        size - 1,
        loc=loc,
    )
```

The `UInt` cast is key, a negative `Int` is reinterpreted as large `UInt` which
exceeds `len`. So the single comparison `UInt(index) < UInt(len)` rejects both
negative and too-large indices in one branch.

For GPU this is off by default, but you can turn on the GPU bounds checks for
testing (we use this for bazel tests):

```bash
mojo build -D ASSERT=all main.mojo
```

If you want to turn off all bounds checking including CPU, you can use:

```bash
mojo build -D ASSERT=none main.mojo
```

The `call_location()` in the `debug_assert` shows the call site from the user's
code where they supplied an invalid index because `__getitem__` is
`@always_inline`. Some collections require method updates to `@always_inline` to
support this. This is essential for users debugging their code to discover where
they're providing the incorrect index, and for easy migration off negative
indexing.

The `debug_assert` message is minimal to avoid generating extra IR for this hot
path. The user still gets all the information required with the index value,
valid range, and source location.

The `normalize_index` function will be deprecated next release and removed in
the release after, to give users who have it in their collections time to
migrate off it.

Note: removing the `Indexer` trait and settling on one concrete indexer type is
outside the scope of this design, and will be considered separately.

## Alternatives Considered

### Deprecate with `debug_assert` error in `normalize_index`

Add a `debug_assert` deprecation message for negative indices on CPU in
`normalize_index`. This is worse because the user sees an error pointing at the
normalization code, not at the call site where the negative index was passed.
It's also not a real deprecation, as the program crashes at runtime rather than
warning. And it introduces overhead while the deprecation is active, since the
normalization branch remains on the hot path.

### Deprecate with print warning

This adds additional IR and overhead to a hot path for a deprecation. The
message is only shown at runtime, not compile time, and obscures the
application's output.

## Conclusion

There is no good path for deprecation of negative indexing, so we provide good
errors that enable quick migration with bounds checking on by default for CPU.
