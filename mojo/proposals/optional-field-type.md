# Optional Field Type

Author: Maxim Zaks  
Date: September 18, 2025

This document proposes the introduction of a new type in the standard library
that allows developers to define compile-time optional fields.

## Motivation

The memory footprint of a type can be divided into:

- **Inherent complexity**: required for representing the data itself (e.g., the
encoded contents of a string).  
- **Feature-based complexity**: additional fields to enable certain operations
(e.g., size and capacity fields for in-place mutation of a string).

Traditionally, reduced-feature variants are modeled as distinct types. However,
Mojo’s compile-time metaprogramming allows types to be parameterized. This makes
it possible to conditionally include or exclude fields, reducing memory
footprint without introducing many new types. We propose standardizing this
pattern with an `OptionalField` type.

## Proposed Type

```mojo
struct OptionalField[
    active: Bool, ElementType: Copyable & Movable
](Copyable, Movable):
    var field: InlineArray[ElementType, 1 if active else 0]

    fn __init__(out self):
        constrained[
            not active, "Constructor only available with no active field"
        ]()
        self.field = InlineArray[ElementType, 1 if active else 0](
            uninitialized=True
        )

    fn __init__(out self, var value: ElementType):
        constrained[active, "Constructor only available with active field"]()
        self.field = InlineArray[ElementType, 1 if active else 0](value^)

    fn __init__(out self, var value: Optional[ElementType]):
        @parameter
        if active:
            self.field = InlineArray[ElementType, 1 if active else 0](
                value.take()
            )
        else:
            self.field = InlineArray[ElementType, 1 if active else 0](
                uninitialized=True
            )

    @always_inline
    fn __getitem__(ref self) -> ref [self.field] Self.ElementType:
        constrained[
            Self.active, "Field is not active, you should not access it."
        ]()
        return self.field[0]
```

Inactive fields occupy 0 bytes, achieved through `InlineArray`. This
implementation is illustrative, not prescriptive.

## Usage Examples

### Compile-Time Optional Field

```mojo
from sys import size_of

struct Person[withAge: Bool](Copyable, Movable):
    var name: String
    var age: OptionalField[withAge, Int]

    fn __init__(out self, var name: String, var age: Optional[Int] = None):
        self.name = name^
        self.age = {age^}


    fn print_info(ref self):
        @parameter
        if withAge:
            print(
                "Name: ", self.name, ", Age: ", self.age[], 
                " (Size: ", size_of[Self](), " bytes)"
            )
        else:
            print(
                "Name: ", self.name, ", Age: N/A (Size: ", 
                size_of[Self](), " bytes)"
            )
```

```mojo
var a1 = Person[True]("Alice", 30)
var a2 = Person[False]("Bob")
a1.print_info()
a2.print_info()
```

Output:

```shell
Name:  Alice , Age:  30  (Size:  32  bytes)
Name:  Bob , Age: N/A (Size:  24  bytes)
```

With a runtime Optional, both instances would occupy 40 bytes, showing the
overhead avoided by OptionalField.

### Compile-Time Sum Type

```mojo
struct Address[AddressType: Int]:
    alias postal = 1
    alias email = 2
    alias phone = 3
    var _postal: OptionalField[
        AddressType == Self.postal, (String, String, String)
    ]  # Street, City, Country
    var _email: OptionalField[AddressType == Self.email, String]
    var _phone: OptionalField[AddressType == Self.phone, (String, String)]
    
    fn __init__(out self, var value: String):
        constrained[
            AddressType == Self.email, "Address type shoudl be email"
        ]()
        self._postal = OptionalField[
            AddressType == Self.postal, (String, String, String)
        ]()
        self._email = OptionalField[AddressType == Self.email, String](value^)
        self._phone = OptionalField[
            AddressType == Self.phone, (String, String)
        ]()

    fn __init__(out self, var value: (String, String)):
        constrained[
            AddressType == Self.phone, "Address type should be phone"
        ]()
        self._postal = OptionalField[
            AddressType == Self.postal, (String, String, String)
        ]()
        self._email = OptionalField[AddressType == Self.email, String]()
        self._phone = OptionalField[
            AddressType == Self.phone, (String, String)
        ](value^)

    fn __init__(out self, var value: (String, String, String)):
        constrained[
            AddressType == Self.postal, "Address type should be postal"
        ]()
        self._postal = OptionalField[
            AddressType == Self.postal, (String, String, String)
        ](value^)
        self._email = OptionalField[AddressType == Self.email, String]()
        self._phone = OptionalField[
            AddressType == Self.phone, (String, String)
        ]()

    fn postal_address(ref self) -> (String, String, String):
        constrained[AddressType == Self.postal, "Not a postal address"]()
        return self._postal[]

    fn email_address(ref self) -> String:
        constrained[AddressType == Self.email, "Not an email address"]()
        return self._email[]

    fn phone_number(ref self) -> (String, String):
        constrained[AddressType == Self.phone, "Not a phone number"]()
        return self._phone[]
    
    fn print_info(ref self):
        @parameter
        if AddressType == Self.postal:
            print(
                "Postal Address: ", self._postal[][0], self._postal[][1], 
                self._postal[][2], " (Size: ", size_of[Self](), " bytes)"
            )
        elif AddressType == Self.email:
            print(
                "Email Address: ", self._email[], 
                " (Size: ", size_of[Self](), " bytes)"
            )
        elif AddressType == Self.phone:
            print(
                "Phone Number: ", self._phone[][0], self._phone[][1],
                " (Size: ", size_of[Self](), " bytes)"
            )
        else:
            print("Invalid Address Type")
```

```mojo
var addr1 = Address[Address.postal]((("123 Main St", "Anytown", "USA")))
var addr2 = Address[Address.email]("alice@example.com")
var addr3 = Address[Address.phone](("+1", "555-123-4567"))

addr1.print_info()
addr2.print_info()
addr3.print_info()

print("Address2 Size: ", size_of[Address2](), " bytes")
```

Output:

```shell
Postal Address:  123 Main St Anytown USA  (Size:  72  bytes)
Email Address:  alice@example.com  (Size:  24  bytes)
Phone Number:  +1 555-123-4567  (Size:  48  bytes)
```

By contrast, a Variant-based representation:

```mojo
from utils import Variant

alias Address2 = Variant[
    (String, String, String),   # Postal
    String,                     # Email
    (String, String)            # Phone
]
```

has a fixed footprint (80 bytes in this case), regardless of which case is
active.

## Alternative: Decorator Syntax

A more natural syntax would use a decorator:

```mojo
struct Person[withAge: Bool](Copyable, Movable):
    var name: String
    @optional(withAge)
    var age: Int
```

This syntax is cleaner but requires compiler support. We propose introducing
`OptionalField` in the standard library first, and later replacing it with a
decorator-based syntax once available.

## Rationale and Trade-offs

- Compared to runtime `Optional`
  - Benefit: OptionalField avoids the extra memory overhead of storing a runtime
  tag or pointer.
  - Trade-off: The presence or absence of the field must be known at compile
  time. This reduces flexibility in exchange for efficiency.
- Compared to `Variant`
  - Benefit: `OptionalField` produces specialized layouts with smaller memory
  footprints for each case.
  - Trade-off: Variants centralize case handling and can represent multiple
  alternatives at runtime, while `OptionalField` only expresses compile-time
  choices.

Overall, `OptionalField` provides a lightweight mechanism for compile-time
memory optimization.
