# String iterators > List allocations APIs

date: 2026-03-17, status: proposed

## Motivation

Oftentimes the biggest performance penalty is memory allocation. The
Python-inspired String API we currently have also allocates very often. A lot
of string operations are also commonly concatenated until a final phase when
the contiguous result string is actually needed. There are also many times where
one does some manipulations of a series of strings and then only needs it for
comparing the result, which shouldn't require any allocation.

## Concrete proposal

Disclaimer: This is a post 1.0 proposal.

Let's make all of these APIs return `StringSlice` iterators:

```mojo
struct String:
    def split(self) -> SplitNoneIterator: ...
    def split(self, chrs: StringSlice) -> SplitCharsIterator: ...
    def split_lines(self) -> SplitLinesIterator: ...
    def replace(self, old: StringSlice, new: StringSlice) -> ReplaceIterator: ...
    def lower(self) -> LowerIterator: ...
    def upper(self) -> UpperIterator: ...
    def join(self, it: Some[StringIterator]) -> JoinIterator: ...
```

The iterators should also have the ability to do:

```mojo
trait StringSliceIterator:
    comptime _origin: Origin

    # make it fully indexable and sliceable
    def __getitem__(idx: Some[Indexer]) -> StringSlice[Self._origin]: ...
    def __getitem__(idx: ContiguousSlice) -> ContiguousStringSliceIterator: ...
    def __getitem__(idx: SteppedSlice) -> SteppedStringSliceIterator: ...
    # other string iterator-yielding APIs
    def split(self) -> SplitNoneIterator: ...
    def split(self, chrs: StringSlice) -> SplitCharsIterator: ...
    def split_lines(self) -> SplitLinesIterator: ...
    def replace(self, chars: StringSlice) -> ReplaceIterator: ...
    def lower(self) -> LowerIterator: ...
    def upper(self) -> UpperIterator: ...
    def join(self, it: Some[StringSliceIterator]) -> JoinIterator: ...
    # make it fully Comparable
    def __eq__(self, other: Some[StringSliceIterator]) -> Bool: ...
    ...
```

There are other APIs we haven't implemented yet that would also fall in this
category of iterator-friendly like Python's `capitalize()` and `title()`.

## Examples of improvements this enables

```mojo
# this would previously allocate 3 times (and replace goes through the whole
# string to count occurrences)
def write_pretty_csv_cols(columns: StringSlice, mut writer: Some[Writer]):
    writer.write(columns.replace("_", " ").replace(",", ", ").title())

# this would previously reallocate several times for the split function
# and once more for the upper one
def write_pretty_hexadecimal(data: StringSlice, mut writer: Some[Writer]):
    # incoming format: some_data;hex_value;some_data;some_data;some_data;some_data; ...
    writer.write(columns.split("=")[1].upper())
```
