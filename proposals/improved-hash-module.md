# Improve hash module

Current implementation of the hash module in standard library reflex the
implementation of the Python hash module, which by itself is a good idea,
but it has some flaws, which we should correct in Mojo.

## Flaws of the `Hashable` trait

The `Hashable` trait is designed as following:

```mojo
trait Hashable:
    fn __hash__(self) -> Int:
        ...
```

Which implies that a developer, who writes a new hashable struct needs
to return an `Int`.

- This API does not provide guidance to how this `Int` value needs to be computed
- It is impossible to exchange hashing algorithms on demand
- `Int` as a type is variable length (based on the CPU arch), which might cause issues
- Such trait design follows the call return principle which, when applied on complex
types, will lead to unnecessary computations and memory allocations

## Example

```mojo
@value
struct Person(Hashable):
    var name: String
    var age: UInt8
    var friends_names: List[String]

    fn __hash__(self) -> Int:
        var hashes = List[Int]()
        hashes.append(hash(self.name))
        hashes.append(hash(self.age))
        for friend in self.friends_names:
            hashes.append(hash(friend[]))

        # How to combine a hash of hashes ???
```

As you can see above we, computed hashes for all of the struct fields,
but we are uncertain how to combine those values in a way which produces
a good (non compromised) hash value. Python [docs](https://docs.python.org/3/reference/datamodel.html#object.__hash__)
suggest to pack fileds into a tuple and hash the tuple, but this is not
possible at this point in time in Mojo.

## Proposal

In order to improve the hash module and address the flaws of the `Hashable`
trait, we need to apply following steps.

## Introduce a `Hasher` trait

By introducing a `Hasher` trait we define an abstraction for the hashing algorithm
itself, which allows streaming creation of the hash value. Below is a possible API
of the `Hashable` trait.

```mojo
trait Hasher:
    """Trait which every hash function implementer needs to implement."""
    fn __init__(inout self):
        """Expects a no argument instantiation."""
        ...
    fn _update_with_bytes(inout self, bytes: DTypePointer[DType.uint8], n: Int):
        """Conribute to the hash value based on a sequence of bytes. Use only for complex types which are not just a composition of Hashable types."""
        ...
    fn _update_with_simd[dt: DType, size: Int](inout self, value: SIMD[dt, size]):
        """Contribute to the hash value with a compile time know fix size value. Used inside of std lib to avoid runtime branching."""
        ...
    fn update[T: Hashable](inout self, value: T):
        """Contribute to the hash value with a Hashable value. Should be used by implementors of Hashable types which are a composition of Hashable types."""
        ...
    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        """Used internally to generate the final hash value, should be simplified to `_finish(owned self) -> Scalar[hash_value_dt]`
        once trait declarations support parameters and we can switch to `trait Hasher[hash_value_dt: DType]`.
        This is beneficial as hash functions have different implementations based on the type """
        ...
```

## Implement a default `Hasher` in standard library

The standard library should provide a default `Hasher` implementation,
but it would be possible for the developers to implement, or choose other
hash algorithms, if they better fit their use case.

Bellow you can see a dummy implementation of a `DefaultHasher`

```mojo
struct DefaultHasher(Hasher):
    var hash: UInt64

    fn __init__(inout self):
        self.hash = 42
    fn _update_with_bytes(inout self, bytes: DTypePointer[DType.uint8], n: Int):
        ...
    fn _update_with_simd[dt: DType, size: Int](inout self, value: SIMD[dt, size]):
        ...
    fn update[T: Hashable](inout self, value: T):
        ...
    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        return self.hash.cast[dt]()
```

## Redesign the `Hashable` trait to follow the data flow principles

Given the `Hasher` trait, we can define the `Hashable` trait to adopt the
data flow paradigm instead of call return.

```mojo
trait Hashable:
    fn hash_with[H: Hasher](self, inout hasher: H):
        ...
```

## Example for `Hashable` struct implementation

The implementation of `Hashable`, where all the fields are `Hashable` is trivial
and could be easily synthesized by the compiler. But even if not, the data flow
API guides the developer toward a very simple solution.

```mojo
@value
struct Person(Hashable):
    var name: String
    var age: Int

    fn __hash__[H: Hasher](self, inout hasher: H):
        hasher.update(self.name)
        hasher.update(self.age)
```

## Parameterized the `hash` function with `Hasher` type

The `hash` function can be parameterized with the `Hasher` type, which gives the
users control over the hashing algorithm.

```mojo
fn hash[T: Hashable, H: Hasher = DefaultHasher](value: T) -> UInt64:
    var hasher = hasher_type()
    hasher.update(value)
    return hasher^._finish()
```

## Prove of concept

Bellow you can find a fully working POC implementation:

```mojo
from os.env import getenv, setenv
from random import random_si64

trait Hashable:
    """Trait which every hashable type needs to implement."""
    fn __hash__[H: Hasher](self, inout hasher: H):
        ...

trait Hasher:
    """Trait which every hash function implementer needs to implement."""
    fn __init__(inout self):
        """Expects a no argument instantiation."""
        ...
    fn _update_with_bytes(inout self, bytes: DTypePointer[DType.uint8], n: Int):
        """Conribute to the hash value based on a sequence of bytes. Use only for complex types which are not just a composition of Hashable types."""
        ...
    fn _update_with_simd[dt: DType, size: Int](inout self, value: SIMD[dt, size]):
        """Contribute to the hash value with a compile time know fix size value. Used inside of std lib to avoid runtime branching."""
        ...
    fn update[T: Hashable](inout self, value: T):
        """Contribute to the hash value with a Hashable value. Should be used by implementors of Hashable types which are a composition of Hashable types."""
        ...
    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        """Used internally to generate the final hash value, should be simplified to `_finish(owned self) -> Scalar[hash_value_dt]`
        once trait declarations support parameters and we can switch to `trait Hasher[hash_value_dt: DType]`.
        This is beneficial as hash functions have different implementations based on the type """
        ...

@value
struct MyInt(Hashable):
    """An example for the Int type."""
    var value: Int

    @always_inline
    fn __hash__[H: Hasher](self, inout hasher: H):
        hasher._update_with_simd(Int64(self.value))

@value
struct MyString(Hashable):
    """An example for the String type."""
    var value: StringLiteral

    @always_inline
    fn __hash__[H: Hasher](self, inout hasher: H):
        hasher.update(MyInt(len(self.value)))
        hasher._update_with_bytes(self.value.data().bitcast[DType.uint8](), len(self.value))

@value
struct Person(Hashable):
    """An example for a type composing Hashable types."""
    var name: MyString
    var age: MyInt

    fn __hash__[H: Hasher](self, inout hasher: H):
        hasher.update(self.name)
        hasher.update(self.age)

alias DefaultHasher = DJBX33A_Hasher[0]

@always_inline
fn my_hash[V: Hashable, hasher_type: Hasher = DefaultHasher](value: V) -> UInt64:
    """Example how the `hash` function should look like."""
    var hasher = hasher_type()
    hasher.update(value)
    return hasher^._finish()

@always_inline
fn _DJBX33A_SECRET() -> UInt64:
    """Example how secret and seed can be stored and retrieved."""
    try:
        var secret_string = getenv("DJBX33A_SECRET", "")
        return bitcast[DType.uint64](Int64(int(secret_string)))
    except:
        var value = random_si64(Int64.MIN, Int64.MAX)
        _ = setenv("DJBX33A_SECRET", str(value))
        return bitcast[DType.uint64](value)

struct DJBX33A_Hasher[custom_secret: UInt64 = 0](Hasher):
    """Example of a simple Hasher, with an option to provide a custom secret at compile time.
    When custom secret is set to 0 the secret will be looked up in env var DJBX33A_SECRET.
    In case env var DJBX33A_SECRET is not set a random int will be generated."""
    var hash_data: UInt64
    var secret: UInt64

    @always_inline
    fn __init__(inout self):
        self.hash_data = 5361
        @parameter
        if custom_secret != 0:
            self.secret = custom_secret
        else:
            self.secret = _DJBX33A_SECRET()

    @always_inline
    fn _update_with_bytes(inout self, bytes: DTypePointer[DType.uint8], n: Int):
        """The algorithm is not optimal."""
        for i in range(n):
            self.hash_data = self.hash_data * 33 + bytes.load(i).cast[DType.uint64]()

    @always_inline
    fn _update_with_simd[dt: DType, size: Int](inout self, value: SIMD[dt, size]):
        """The algorithm is not optimal."""
        alias size_in_bytes = size * dt.sizeof()
        var bytes = bitcast[DType.uint8, size_in_bytes](value)
        @parameter
        for i in range(size_in_bytes):
            self.hash_data = self.hash_data * 33 + bytes[i].cast[DType.uint64]()

    @always_inline
    fn update[T: Hashable](inout self, value: T):
        value.__hash__(self)

    @always_inline
    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        return (self.hash_data ^ self.secret).cast[dt]()

fn main() raises:
    var p = Person("Maxim", 43)
    print(p.name.value, p.age.value)

    var hasher = DJBX33A_Hasher()
    p.age.__hash__(hasher)
    print("My hasher 43", hasher^._finish())
    print("Std hash 43", hash(p.age.value))

    hasher = DJBX33A_Hasher()
    p.__hash__(hasher)
    print("Person", hasher^._finish())

    var h1 = my_hash(p)
    var h2 = my_hash[hasher_type=DJBX33A_Hasher[77777]](p)
    var h3 = my_hash(p)
    print("Person", h1, h2, h3)
```

## Compiler limitations

Current compiler does not allow parameters on trait definition.
A parametrization on Hasher trait for for hash value dtype would be
beneficial as a hashing algorithm might differ.
For example in [Fowler–Noll–Vo hash function](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV_hash_parameters)
parameters prime and offset basis dependend on hash value width.
