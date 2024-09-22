from memory import UnsafePointer, stack_allocation, memcpy

struct Boxed[T: AnyType, address_space: AddressSpace = AddressSpace.GENERIC]:
    var _inner: UnsafePointer[T, address_space]

    fn __init__[T: Movable](inout self, owned value: T):
        self._inner = UnsafePointer[T, address_space].alloc(1)
        self._inner.init_pointee_move(value^)

    fn __init__[T: ExplicitlyCopyable](inout self, value: T):
        self._inner = UnsafePointer[T, address_space].alloc(1)
        self._inner.init_pointee_explicit_copy(value)

    fn __init__[T: ExplicitlyCopyable](inout self, other: Self):
        var copied_t = T(other[])
        Self(copied_t)

    #fn __init__[T: AnyTrivialRegType, other_address_space: AddressSpace](inout self, other: Boxed[T, other_address_space]):
     #   self._inner = UnsafePointer[T, address_space].alloc(1)
      #  memcpy[1](self._inner, other[])

    fn __moveinit__(inout self, owned existing: Self):
        self._inner = existing._inner
        existing._inner = UnsafePointer[T, address_space]()

    fn __getitem__(ref [_] self) -> ref [__lifetime_of(self)] T:
        # This should have a widening conversion here that allows
        # the mutable ref that is always (potentially unsafely)
        # returned from UnsafePointer to be guarded behind the
        # aliasing guarantees of the lifetime system here.
        # All of the magic happens above in the function signature

        debug_assert(self._inner, "Box is horribly broken, and __getitem__ was called on a destroyed box")

        self._inner[]

    fn __del__(owned self):
        self._destroy()

    fn _destroy(inout self):
        # check that inner is non-null to accomodate into_inner and other
        # consuming end states
        if self._inner:
            (self._inner).destroy_pointee()
            self._inner.free()
            self._inner = UnsafePointer[T, address_space]()

    fn into_inner(owned self) -> T:
        var r = (self._inner).take_pointee()
        self._inner.free()
        self._inner = UnsafePointer[T, address_space]()

        return r^
