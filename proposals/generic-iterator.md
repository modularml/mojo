# Iterator trait

As started by @jayzhan211 in issue [#2629](https://github.com/modularml/mojo/issues/2629),
an Iterable trait and Iterator implementation would be very useful. Especially since
currently every stldlib type has its own iterator and support for each has to be added
independently if a function wants to take iterable arguments.

## What is proposed?

```mojo
trait HasNext[T: CollectionElement]:
    fn __next__(self) -> T:
        ...


trait HasNextLen[T: CollectionElement]:
    fn __next__(self) -> T:
        ...

    fn __len__(self) -> Int:
        ...


trait HasNextRaising[T: CollectionElement]:
    fn __next__(self) raises -> T:
        ...


trait HasOptionalNext[T: CollectionElement]:
    fn __next__(self) -> Optional[T]:
        ...


struct StaticSizedIterator[size: Int, T: CollectionElement, A: HasNext[T]]:
    var _has_next: A

    fn __init__(inout self, has_next: A):
        self._has_next = has_next

    fn __iter__(self) -> Iterator[T]:
        return Iterator(self)

    fn __next__(self) -> T:
        return next(self._has_next)


struct SizedIterator[T: CollectionElement, A: HasNextLen[T]]:
    var _has_next: A

    fn __init__(inout self, has_next: A):
        self._has_next = has_next

    fn __iter__(self) -> Iterator[T]:
        return Iterator(self)

    fn __next__(self) -> T:
        return next(self._has_next)

    fn __len__(self) -> Int:
        return len(self._has_next)


struct RaisingIterator[T: CollectionElement, A: HasNextRaising[T]]:
    var _has_next: A

    fn __init__(inout self, has_next: A):
        self._has_next = has_next

    fn __iter__(self) -> Iterator[T, A]:
        return Iterator(self)

    fn __next__(self) raises -> T:
        return next(self._has_next)


@value
struct Iterator[
    T: CollectionElement,
    A: Variant[
        HasOptionalNext[T],
        RaisingIterator[T],
        SizedIterator[T],
        StaticSizedIterator[T],
    ],
]:
    var _has_next: A
    var _idx: UInt

    fn __init__(inout self, has_next: A):
        self._has_next = has_next
        self._idx = 0

    fn __iter__(self) -> Self:
        return self

    fn __next__[I: HasOptionalNext[T]](self: Iterator[T, N]) -> Optional[T]:
        return next(self._has_next)

    fn __next__[
        N: HasNextRaising[T], I: RaisingIterator[T, N]
    ](self: Iterator[T, N]) -> Optional[T]:
        try:
            return next(self._has_next)
        except:
            return None

    fn __next__[
        N: HasNextLen[T], I: SizedIterator[T, N]
    ](self: Iterator[T, N]) -> Optional[T]:
        if len(self._has_next) == 0:
            return None
        return next(self._has_next)

    fn __next__[
        size: Int, N: HasNext[T], I: StaticSizedIterator[size, T, N]
    ](self: Iterator[T, N]) -> Optional[T]:
        # FIXME: the state of _idx should be contained here and not in a struct
        # attr. Once we have generators ?
        if self._idx == size:
            return None
        self._idx += 1
        return next(self._has_next)

    ...


trait HasIter[T: CollectionElement]:
    fn __iter__(self) -> Iterator[T]:
        ...
```

## What would be needed?

The for loop codegen implementation would need to check for None and break so
that pythonic syntax is preserved
```mojo
for i in List("something", "something"):
    print(i)
```

`@parameter for` should accept `StaticSizedIterator`

## Other details

To allow for functional patterns, the Iterator struct could have wrappers for
itertools
```mojo
struct Iterator[T: CollectionElement, A: Nextable[T]]:
    ...
    fn map(owned self, func: fn(value: T) -> T) -> Self:
        return map(func, self)

    fn filter(owned self, func: fn(value: T) -> Bool) -> Self:
        return filter(func, self)
    
    fn batched(owned self, amnt: Int) -> Self:
        return itertools.batched(self, amnt)
```

`zip` implementation would be something like 
```mojo
struct _zip[size: Int, *Ts: CollectionElement, *A: HasIter[Ts]]:
    var _iters: Tuple[*A]

    fn __init__(inout self, *values: *A):
        @parameter
        for i in range(size):
            self._iters[i] = iter(values[i])

    fn __next__(self) -> Optional[Tuple[*Ts]]:
        var items: Optional[Tuple[*Ts]]
        @parameter
        for i in range(size):
            item[i] = next(self._iters[i])
        return items

fn zip[*Ts: CollectionElement, *A: HasIter[Ts]](*values: *A) -> Iterator[*Ts]:
    return Iterator(_zip[len(values)](values))
```

And once we have capable enough generics for this code to be feasible,
implementing a generator would be as simple as taking a yielding function
```mojo
struct Generator[T: CollectionElement, *A: AnyType, **B: AnyType]:
    alias _coro = Coroutine[fn(*A, **B) -> Optional[T]]
    var _func: _coro
    var _args: A
    var _kwargs: B

    fn __init__(inout self, func: _coro, *args: A, **kwargs: B):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    fn __next__(self) -> Optional[T]:
        return self._func(self._args, self._kwargs).get()
```

That way:
```mojo
fn some_yielding_func() -> Optional[Int]
    ...

fn main():
    var iterator = Iterator(Generator(some_yielding_func))
    var concatenated = ", ".join(iterator)
```