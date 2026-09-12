# Draining Collections in Mojo

Josiah Laivins, September 18, 2025

Status: Proposal

This doc describes the design and behavior of draining collections in Mojo. This
proposes the `take_items` as a standard way of moving elements efficiently out
of their respective collections.  It walks through concerns related to the drain
iterator vs the source collection, and weird behavior that can come out of
interacting with collections with active drain iterators. Additionally it
explains a workaround for tuple unpacking using `Optional`. It also goes through
 compatibility with iter tools like `zip`.

## Motivation

Mojo has changed default copy behavior from
[ImplicitlyCopyable to ExplicitlyCopyable][1]. This has surfaced more clearly
places in the stdlib where we want to be moving values, not copying them.

One major part of the stdlib where copies occur is when we work with
collections. Moving elements out of collections involves clever, but unstable
workarounds to avoid copying elements as we move them out of the collection.

## Background

The example below shows that prior to making ExplicitlyCopyable the default,
we were copying values when taking ownership of items in a collection.

Given struct `Foo` that is (Explicitly) `Copyable` and `Movable`. Also
given 2 example functions `take_ownership`, one that takes a single `Foo`, and
one that takes a `String` and a `Foo` (as a key value pair).

```mojo
from collections import Set
from hashlib import Hasher
# Foo is used as an example so we can print the copies.
@fieldwise_init
struct Foo(Copyable & Movable & Hashable & EqualityComparable):
    var x: String

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self.x)

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        print('Foo.',self.x,' copied',sep='')

fn take_ownership(var item: Foo):
    print('Taking ownership of item:', item.x)

fn take_ownership(var key: String, var item: Foo):
    print('Taking ownership of: (key) ',key,' item:', item.x)
```

The straight forward way to interface with functions that take ownership
of elements in collections involves copying the elements.

```mojo

fn test_copyable_a_iteration():
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    for ref item in l:
        take_ownership(item.copy()) # Must be copied, not moved (via ^)

    print('Testing set')
    var set: Set[Foo] = {Foo(x="a"), Foo(x="b"), Foo(x="c")}
    for ref item in set:
        take_ownership(item.copy()) # Must be copied, not moved (via ^)

    print('Testing dict')
    var dict: Dict[String, Foo] = {"a": Foo(x="a"), "b": Foo(x="b"), "c": 
                                   Foo(x="c")}
    for ref item in dict.items():
        # Must be copied, not moved (via ^)
        take_ownership(item.key.copy(), item.value.copy()) 

```

Collections `List` and `Set` and `Dict` all have `pop`/`popitem` methods that
can move popped values (as of this writing `Set.pop` does a copy still).

How would we do this currently? We may try a while loop:

```mojo
fn test_while_loop_moving_items_out_of_collection():
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    while l:
        # List.pop will do a full collection move-left for every call to pop(0)
        take_ownership(l.pop(0))

    print('Testing set')
    var set: Set[Foo] = {Foo(x="a"), Foo(x="b"), Foo(x="c")}
    while set:
        # Set.pop calls require raising context. Hashable items get copied
        # since it copies the reference in the underlying `_DictKeyIter` 
        try:
            take_ownership(set.pop())
        except:
            print('Set is empty')

    print('Testing dict')
    var dict: Dict[String, Foo] = {"a": Foo(x="a"), "b": Foo(x="b"), 
                                   "c": Foo(x="c")}
    print('beginning iteration')
    while dict:
        try:
            # Dict.popitem calls require raising context. 
            # Executes a move, however must be in LIFO order.
            var item = dict.popitem()
            # DictEntry does not support move-based unpacking, so we need to 
            # copy the key, then take the value.
            take_ownership(item.key.copy(), item^.reap_value())
        except:
            print('Dict is empty')
```

This is made more complicated by combining with iter tools like `zip`:

```mojo
fn test_zip_from_2_lists():
    var keys = List[String]("a", "b", "c")
    var values = List[Foo](Foo(x="d"), Foo(x="e"), Foo(x="f"))
    var it = zip(keys^, values^)
    for item in it:
        # Copies happen during zip iteration, then additionally
        # must happen when passing to take_ownership. We are copying twice per 
        # iter!
        take_ownership(item[0].copy(), item[1].copy())
```

This is a common pattern, given a list of keys and a list of values, we want to
zip them together into a map-like object (Dict, Tree, etc.).

### How do other languages handle this?

Rust

> TODO: Would like input from people with Rust heavy backgrounds to clarify this
> is how Rust drain actually works, particularly "when do the source collections
> get cleared"?

Rust has a struct called
[`vec/Drain`][2] and
[`hashmap/Drain`][3]
that is used by collections like `Vec` and `HashMap` to drain the collection.

[Vec.drain](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.drain)

```rust
let mut v = vec![1, 2, 3];
let u: Vec<_> = v.drain(1..).collect();
assert_eq!(v, &[1]);
assert_eq!(u, &[2, 3]);

// A full range clears the vector, like `clear()` does
v.drain(..);
assert_eq!(v, &[]);
```

[HashMap.drain](https://doc.rust-lang.org/std/collections/struct.HashMap.html#method.drain)

```rust
use std::collections::HashMap;

let mut a = HashMap::new();
a.insert(1, "a");
a.insert(2, "b");

for (k, v) in a.drain().take(1) {
    assert!(k == 1 || k == 2);
    assert!(v == "a" || v == "b");
}

assert!(a.is_empty());
```

For sequential collections like Vec:

- `drain(..)` clears the full collection either on full iteration or when the
iterator is dropped (destroyed).
- `drain(start..end)` drains a range of elements either on full iteration or when
the iterator is dropped (destroyed).

For hashable collections like HashMap:

- `drain().take(n)` returns the first `n` elements. *Clears the entire collection
when the drain iter is dropped (destroyed).*
- `drain()` drains the full collection either on full iteration or when the
iterator is dropped (destroyed).

Note that the `Drain` struct has [`Drain.keep_rest`][4]
that avoids clearing unyielded elements when Drain is dropped.

## Proposal

The stdlib team has proposed to move forward with [Rust style draining of
collections][5].

Scope:

This proposal can expand to support additional
collections beyond `List`, `Set`, and `Dict`, however for brevity we will focus
on the basic collections for now. It also goes through compatibility with iter
tools like `zip`, however will not go into compatibility of all the iter tool
functions that could be added in the future that Python supports.

Goals:

- `take_items` implementation for `Set`, `List`, and `Dict` initially that aligns
with Rust's `Drain` struct that does not copy elements.
- Compatibility with iter tools like `zip`.

Non-Goals:

- Block on move-only tuple unpacking.

### User Experience Cases

In all cases, `ref` must be used to get mutable access to the item. `for` loops
by default iterate over immutable references (even if the source collection is
mutable).

In all cases the call to `take_items` takes mutable reference of the underlying
collection, and clears the source collection immediately on destruction of the
iterator. Cases such as `take_items(Slice(..))` takes mutable reference of the
subset of the collection, and updates the source collection on destruction of
the iterator.

- e.g.: `list.take_items(Slice(1, 2))` takes elements at indicies 1 and 2, and
one-time shifts left the remaining elements in the source list on destruction of
the iterator.

Basic flow from Collection -> TakeIterable -> for loop / iteration as
pseudo-code.

```mojo
collection = List("a", "b", "c") or Dict("a": 1, "b": 2, "c": 3) or \ 
Set(1, 2, 3)
take_iter = collection.take_items()
assert(len(collection) == 3) 

for ref item in take_iter:
    # Do something with item or break early
    # Note: What happens if we do `collection.pop(0)` here? Is this something 
    # to worry about?
    break
# On take_iter destruction, the collection (or its subset) is cleared / 
# updated.
assert(len(collection) == 0) 
```

A non-obvious case here is how to handle cases where iterated items are stored
in tuples such as in `zip`. The examples below, wrap the elements in `Optional`
so we can `take` the elements out. This allows the elements to be moved out of
tuples without deinitializing the tuple overall. As part of this proposal a
method such as `DictEntry.reap` will return a
`Tuple[Optional[Key], Optional[Value]]` that allows each element to be
independently moved out of the tuple without deinitializing the tuple overall.

- Ideally above would not be needed if structs (with a unpackable trait) and
tuples supported move unpacking. Reference issue
[#5330](https://github.com/modular/modular/issues/5330).

The basic case is a complete drain of each respective collection.

```mojo
def basic_user_experience_cases():
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    for item in l.take_items():
        take_ownership(item.take())
    assert_equal(len(l), 0)

    print('Testing set')
    var set: Set[Foo] = {Foo(x="a"), Foo(x="b"), Foo(x="c")}
    for ref item in set.take_items():
        take_ownership(item.take())
    assert_equal(len(set), 0)

    print('Testing dict')
    var dict: Dict[String, Foo] = {
        "a": Foo(x="a"), 
        "b": Foo(x="b"), 
        "c": Foo(x="c")
    }
    for ref item in dict.take_items():
        # Dict take_items returns the DictEntry struct, which does not 
        # automatically unpack itself.
        # Ideally mojo would support `__unpack__` like in #5330
        # However we can define method that returns a 
        # Tuple[Optional[Key], Optional[Value]]
        # using optional allows for moving the key and value out of the item 
        # without deinitializing the item overall.
        var key_value = item.reap()
        take_ownership(key_value[0].take(), key_value[1].take())
    assert_equal(len(dict), 0)
```

Draining only slices of the collections. This moves the elements out of the
List, then allows the list to do a single move-left on initialization instead of
per element.

```mojo
def slice_drain_user_experience_cases():
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    for ref item in l.take_items(Slice(1, 2)):
        take_ownership(item.take())
    assert_equal(len(l), 1)
```

Draining multiple lists via zip.

```mojo
    var keys = List[String]("a", "b", "c")
    var values = List[Foo](Foo(x="d"), Foo(x="e"), Foo(x="f"))
    var new_dict = Dict[String, Foo]()
    for ref item in zip(keys, values):
        new_dict[item[0].take()] = item[1].take()
```

Clearing collections on incomplete iteration.

```mojo
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    for ref item in l.take_items():
        break # Iterator clears the remaining elements on destruction
    assert_equal(len(l), 0)

    print('Testing set')
    var set: Set[Foo] = {Foo(x="a"), Foo(x="b"), Foo(x="c")}
    for ref item in set.take_items():
        break # Iterator clears the remaining elements on destruction
    assert_equal(len(set), 0)

    print('Testing dict')
    var dict: Dict[String, Foo] = {
        "a": Foo(x="a"), 
        "b": Foo(x="b"), 
        "c": Foo(x="c")
    }
    for ref item in dict.take_items():
      break # Iterator clears the remaining elements on destruction
    assert_equal(len(dict), 0)
```

## Questions

**Do we think Mojo's Trait API is flexible enough to make Drain/TakeIterator a
trait?**

**Should we allow the user to explicitly keep the remaining elements?**

Note this is contentious in [the rust issue][6].

```mojo
    print('Testing list')
    var l = List[Foo](Foo(x="a"), Foo(x="b"), Foo(x="c"))
    # Not sure what the iterator should do for keep_rest in this case.
    # When do we shift the remaining elements left?
    drain_iter = l.take_items()
    for item in drain_iter:
        break 
    drain_iter.keep_rest()
    assert_equal(len(l), 2)
```

**How should we handle cases where the drain destruction is delayed?**

Note, these edge cases can be partially avoided if the call to `take_items`
instantly takes ownership of the underlying collection before any iteration is
expected to occur.

```mojo
def test_dict_take_items():
    var d: Dict[String, Int] = {"a": 1, "b": 2, "c": 3}
    var iter = d.take_items() 

    # Compiler error: d should not be accessible here regardless whether we:
    #   - Eventually support xor mutability
    #   - Use d^.take_items(), deinitializing d.
    assert_equal(len(d), 0) 
    # User's fault for trying to delay the iterator's destruction here. 
    _ = iter 
```

Or even stranger:

```mojo
def test_list_take_items():
    var ls: List = ["a", "b", "c"]
    var iter = ls.take_items() 
    for element in iter:
        # Only removed the first item, and avoid moving the others left since 
        # that can be slow.
        break 
    
    # The items haven't been moved to the left yet, and the iterator has not 
    # been destroyed.  As far as List is concerned, what is actually in 
    # index 0?
    _ = ls.pop(0) 

    assert_equal(len(ls), 0) # <-- will raise
    # User's fault for trying to delay the iterator's destruction here. 
    _ = iter 
```

One solution is that `take_items` moves the whole collection into the iterator,
and clears the collection immediately.

**Should Dict.take_items()'s iterator just return Tuple[Optional[K],
Optional[V]] directly?**

The default `DictEntry` doesn't automatically unpacking itself. This proposal
introduces a `DictEntry.reap` method that returns a
`Tuple[Optional[K], Optional[V]]` to get around this.

```mojo
    var dict: Dict[String, Foo] = {
        "a": Foo(x="a"), 
        "b": Foo(x="b"), 
        "c": Foo(x="c")
    }
    for ref item in dict.take_items():
        var key_value = item.reap()
        take_ownership(key_value[0].take(), key_value[1].take())
    assert_equal(len(dict), 0)
```

Versus if mojo supported a `__unpack__` method on structs.

```mojo
    var dict: Dict[String, Foo] = {
        "a": Foo(x="a"), 
        "b": Foo(x="b"), 
        "c": Foo(x="c")
    }
    for var (key, value) in dict.take_items():
        take_ownership(key,value)
    assert_equal(len(dict), 0)
```

[1]: https://github.com/modular/modular/commit/0b49b8c8c462142bcc9c44e1231599ffa1ba969d
[2]: https://doc.rust-lang.org/std/vec/struct.Drain.html
[3]: https://doc.rust-lang.org/std/collections/hash_map/struct.Drain.html
[4]: https://doc.rust-lang.org/std/collections/hash_map/struct.Drain.html#method.keep_rest
[5]: https://github.com/modular/modular/pull/5328#discussion_r2360472206
[6]: https://github.com/rust-lang/rust/issues/101122
