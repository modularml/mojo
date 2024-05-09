from . import InlineArray


@value
struct _SpanIter[
    T: CollectionElement,
    is_mutable: __mlir_type.`i1`,
    lifetime: AnyLifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for Span.

    Parameters:
        T: The type of the elements in the span.
        is_mutable: Whether the reference to the span is mutable.
        lifetime: The lifetime of the Span.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var src: Span[T, is_mutable, lifetime]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(
        inout self,
    ) -> Reference[T, is_mutable, lifetime]:
        @parameter
        if forward:
            self.index += 1
            return self.src._refitem__(self.index - 1)
        else:
            self.index -= 1
            return self.src._refitem__(self.index)

    @always_inline
    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


@value
struct Span[
    T: CollectionElement,
    is_mutable: __mlir_type.i1,
    lifetime: AnyLifetime[is_mutable].type,
]:
    """A non owning view of contiguous data.

    Parameters:
        T: The type of the elements in the span.
        is_mutable: Whether the span is mutable.
        lifetime: The lifetime of the Span.
    """

    var data: UnsafePointer[T]
    var len: Int

    @always_inline
    fn __init__(inout self, *, data: UnsafePointer[T], len: Int):
        """Unsafe construction from a pointer and length.

        Args:
            data: The underlying pointer of the span.
            len: The length of the view.
        """
        self.data = data
        self.len = len

    @always_inline
    fn __init__(inout self, list: Reference[List[T], is_mutable, lifetime]):
        """Construct a Span from a List.

        Args:
            list: The list to which the span refers.
        """
        self.data = list[].data
        self.len = len(list[])

    @always_inline
    fn __init__[
        size: Int
    ](inout self, array: Reference[InlineArray[T, size], is_mutable, lifetime]):
        """Construct a Span from an InlineArray.

        Args:
            array: The array to which the span refers.
        """
        self.data = UnsafePointer(array).bitcast[T]()
        self.len = size

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the span. This is a known constant value.

        Returns:
            The size of the span.
        """
        return self.len

    @always_inline
    fn _refitem__[
        intable: Intable
    ](self, index: intable) -> Reference[T, is_mutable, lifetime]:
        debug_assert(
            -self.len <= int(index) < self.len, "index must be within bounds"
        )

        var offset = int(index)
        if offset < 0:
            offset += len(self)
        return (self.data + offset)[]

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the list length."""
        var adjusted_span = span

        if adjusted_span.start < 0:
            adjusted_span.start = len(self) + adjusted_span.start

        if not adjusted_span._has_end():
            adjusted_span.end = len(self)
        elif adjusted_span.end < 0:
            adjusted_span.end = len(self) + adjusted_span.end

        if span.step < 0:
            var tmp = adjusted_span.end
            adjusted_span.end = adjusted_span.start - 1
            adjusted_span.start = tmp - 1

        return adjusted_span

    @always_inline
    fn __getitem__[IntableType: Intable](self, index: IntableType) -> T:
        """Get a `Reference` to the element at the given index.

        Parameters:
            IntableType: The inferred type of an intable argument.

        Args:
            index: The index of the item.

        Returns:
            A reference to the item at the given index.
        """
        # note that self._refitem__ is already bounds checking
        return self._refitem__(index)[]

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        """Get a new span from a slice of the current span.

        Returns:
            A new span that points to the same data as the current span.
        """
        var adjusted_span = self._adjust_span(span)
        debug_assert(
            0 <= adjusted_span.start < self.len and 0 <= adjusted_span.end,
            "Slice must be within bounds.",
        )
        var res = Self(
            data=(self.data + adjusted_span.start), len=len(adjusted_span)
        )

        return res

    @always_inline
    fn __iter__(self) -> _SpanIter[T, is_mutable, lifetime]:
        """Get an iterator over the elements of the span.

        Returns:
            An iterator over the elements of the span.
        """
        return _SpanIter(0, self)
