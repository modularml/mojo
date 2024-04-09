# Improve hash module

Current implementation of the hash module in standard library reflex the implementation of the Python hash module, which by itself is a good idea, but it has some flaws, which we should correct in Mojo.

# Flaws of the `Hashable` trait

The `Hashable` trait is designed as following:

```mojo
trait Hashable:
    fn __hash__(self) -> Int:
        ...
```

Which implies that a developer, who writes a new hashable struct needs to return an `Int`. 
- This API does not provide guidance to how this `Int` value needs to be computed
- It is impossible to exchange hashing algorithms on demand
- `Int` as a type is variable length (based on the CPU arch), which might cause issues
- Such trait design follows the call return principle which, when applied on complex types, will lead to unnecessary computations and memory allocations

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
As you can see above we, computed hashes for all of the struct fields, but we are uncertain how to combine those values in a way which produces a good (non compromised) hash value.

# Proposal

In order to improve the hash module and address the flaws of the `Hashable` trait, we need to apply following steps.

## Introduce a `Hasher` trait

By introducing a `Hasher` trait we define an abstraction for the hashing algorithm itself, which allows streaming creation of the hash value. Bellow is a possible API of the `Hashable` trait.

```mojo
trait Hasher:
    fn __init__(inout self):
        ...
    fn update[T: DType](inout self, value: SIMD[T, 1]):
        ...
    fn update(inout self, pointer: DTypePointer[DType.uint8], length: Int):
        ...
    fn finish(owned self) -> UInt64:
        ...
```

## Implement a default `Hasher` in standard library

The standard library should provide a default `Hasher` implementation, but it would be possible for the developers to implement, or choose other hash algorithms, if they better fit their use case.

Bellow you can see a dummy implementation of a `DefaultHasher`

```mojo
struct DefaultHasher(Hasher):
    var hash: UInt64

    fn __init__(inout self):
        self.hash = 42
    fn update[T: DType](inout self, value: SIMD[T, 1]):
        ...
    fn update(inout self, pointer: DTypePointer[DType.uint8], length: Int):
        ...
    fn finish(owned self) -> UInt64:
        return self.hash
```

## Redesign the `Hashable` trait to follow the data flow principles

Given the `Hasher` trait, we can define the `Hashable` trait to adopt the data flow paradigm instead of call return.

```mojo
trait Hashable:
    fn hash_with[H: Hasher](self, inout hasher: H):
        ...
```

## Example for `Hashable` struct implementation

The implementation of `Hashable`, where all the fields are `Hashable` is trivial and could be easily synthesized by the compiler. But even if not, the data flow API guides the developer toward a very simple solution

```mojo
@value
struct Person(Hashable):
    var name: String
    var age: UInt8
    var friends_names: List[String]

    fn hash_with[H: Hasher](self, inout hasher: H):
        # self.name.hash_with(hasher), when String is Hashable, otherwise:
        hasher.update(self.name._as_ptr().bitcast[DType.uint8](), len(self.name))
        # self.age.hash_with(hasher), when SIMD is hashable, otherwise
        hasher.update(self.age)
        # self.friends_names.hash_with(hasher), when List of Hashable types is Hashable, otherwise:
        for friend in self.friends_names:
            hasher.update(friend[]._as_ptr().bitcast[DType.uint8](), len(friend[]))
```

## Parameterized the `hash` function with `Hasher` type

The `hash` function can be parameterized with the `Hasher` type, which gives the users control over the hashing algorithm.

```mojo
fn hash[T: Hashable, H: Hasher = DefaultHasher](value: T) -> UInt64:
    var hasher = H()
    value.hash_with(hasher)
    return hasher^.finish()
```

Alternatively we could go with following API

```mojo
fn hash[T: Hashable, H: Hasher, hasher_factory: fn () -> H](value: T) -> UInt64:
    var hasher = hasher_factory()
    value.hash_with(hasher)
    return hasher^.finish()
```

Where we allow complex and context dependent `Hasher` initialization.

Similar parameterization should be applied for all hash based data structures like `Dict` and `Set`.
