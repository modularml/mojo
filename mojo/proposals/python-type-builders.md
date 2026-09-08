# Python type builders

## Goal

This document proposes an approach for enabling Mojo developers to define
[type objects](https://docs.python.org/3/c-api/typeobj.html).

Type objects are the CPython mechanism that allows binary extensions to fully
implement types that behave like native Python types. Libraries like `numpy`
rely on them to deliver powerful APIs — for example
[indexing](https://numpy.org/doc/stable/user/basics.indexing.html).

To participate in CPython's protocols (operators, `len()`, iteration,
subscripting, …), Mojo slot implementations must match the C function
signatures that the interpreter expects to call. Some slots also have a
sentinel contract — for example, a `tp_richcompare` slot can implement only
`<` and `==` and return `NotImplemented` for the others; CPython then derives
`>`, `>=`, `<=` and `!=` from the reflected operands. The exact set of slots
addressed by this proposal is enumerated in [Type Objects
protocols](#type-objects-protocols) below.

## Current state

The Mojo standard library already allows developers to expose Mojo modules as
native Python extensions — see ["Calling Mojo from
Python"](https://docs.modular.com/mojo/manual/python/mojo-from-python/). The
library provides ways to convert native scalar types between the two systems
and also provides ways to define Python modules and classes with user-defined
methods.

Today, however, those user-defined methods are bound by name as regular
attributes — not as `PyTypeObject` slots. A Mojo type can declare
`__getitem__` and it will be callable as `obj.__getitem__(i)`, but `obj[i]`,
`for x in obj`, `len(obj)`, `a + b`, and the rich comparison operators bypass
the user-supplied method entirely because CPython dispatches them through
type-slot function pointers rather than attribute lookup. This proposal
closes that gap by extending the existing Python integration to populate
those slots.

## Example

Columnar data — values laid out one column at a time rather than row-by-row —
is a foundational representation in data analytics and machine learning. It
underpins formats like Arrow and Parquet, DataFrame libraries such as pandas,
Polars, and cuDF, and the dense numeric tensors that most ML pipelines consume.

Under this proposal a `DataFrame` library would use the following Type Objects
protocols.

- **Mapping protocol** — `len(df)`, `df[i]`, `df[i] = (x, y)`, `del df[i]`
  via `mp_length` / `mp_subscript` / `mp_ass_subscript`.
- **Rich comparison** — `df < other`, `df == other` via `tp_richcompare`,
  with the remaining operators raising `PySlotError.not_implemented()` so
  CPython falls back to the reflected call on the other operand.
- **Number protocol** — `-df`, `abs(df)`, `bool(df)`, `df + other`,
  `df * scalar`, `df ** exp` via `nb_negative` / `nb_absolute` / `nb_bool` /
  `nb_add` / `nb_multiply` / `nb_power`. Operations against incompatible
  types return `NotImplemented`, letting Python raise `TypeError` naturally.

The code below is just for illustration purposes.

```mojo
struct DataFrame(Defaultable, Movable, Writable):
    """A simple columnar data structure storing 2-D points.

    x and y coordinates are stored in separate columns for cache-friendly
    access patterns. Used here to demonstrate the mapping, rich comparison,
    and number protocols.
    """

    var pos_x: Coord1DColumn
    var pos_y: Coord1DColumn
    var _bounding_box_area: Float64

    def __init__(out self):
        self.pos_x = []
        self.pos_y = []
        self._bounding_box_area = 0

    def py__getitem__(
        self, index: PythonObject
    ) raises PySlotError -> PythonObject:
        var i: Int
        try:
            i = Int(py=index)
        except e:
            raise PySlotError.type_error(String(e))
        var length = len(self.pos_x)
        if i < 0 or i >= length:
            raise PySlotError.index_error("index out of range")
        try:
            return Python().tuple(self.pos_x[i], self.pos_y[i])
        except e:
            raise PySlotError.runtime_error(String(e))

@export
def PyInit_columnar_mojo() -> PythonObject:
    """Entry point: create the Python extension module."""
    try:
        var b = PythonModuleBuilder("columnar_mojo")

        ref tb = (
            b.add_type[DataFrame]("DataFrame")
            .def_init_defaultable[DataFrame]()
        )
        var tpb = TypeProtocolBuilder[DataFrame](tb)
        _ = tpb.def_richcompare[DataFrame.rich_compare]()
        var mpb = MappingProtocolBuilder[DataFrame](tb)
        _ = (
            mpb.def_len[DataFrame.py__len__]()
            .def_getitem[DataFrame.py__getitem__]()
            .def_setitem[DataFrame.py__setitem__]()
        )
        var npb = NumberProtocolBuilder[DataFrame](tb)
        _ = (
            npb.def_neg[DataFrame.py__neg__]()
            .def_abs[DataFrame.py__abs__]()
            .def_bool[DataFrame.py__bool__]()
            .def_add[DataFrame.py__add__]()
            .def_mul[DataFrame.py__mul__]()
            .def_pow[DataFrame.py__pow__]()
        )

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))

```

## Type Objects protocols

The following protocols and their methods are addressed in this proposal.

### Type protocol

Top-level `PyTypeObject` slots that apply to every type. In this branch the
builder exposes rich comparison; other `tp_*` slots (init, dealloc, repr, hash,
…) are already populated in the Mojo stdlib.

See
[`PyTypeObject`](https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject).

| Builder method    | CPython slot                                                                                   | Python operator                  |
|-------------------|------------------------------------------------------------------------------------------------|----------------------------------|
| `def_richcompare` | [`tp_richcompare`](https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare) | `<`, `<=`, `==`, `!=`, `>`, `>=` |

### Mapping protocol

Slots that make a type behave like a dict-like mapping keyed by arbitrary
Python objects.

See
[`PyMappingMethods`](https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods).

| Builder method | CPython slot                                                                                           | Python operator                |
|----------------|--------------------------------------------------------------------------------------------------------|--------------------------------|
| `def_len`      | [`mp_length`](https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_length)               | `len(obj)`                     |
| `def_getitem`  | [`mp_subscript`](https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_subscript)         | `obj[key]`                     |
| `def_setitem`  | [`mp_ass_subscript`](https://docs.python.org/3/c-api/typeobj.html#c.PyMappingMethods.mp_ass_subscript) | `obj[key] = v`, `del obj[key]` |

### Sequence protocol

Slots for list/tuple-like types indexed by `Py_ssize_t`. Assignment with a
`null` value indicates deletion, so `def_setitem` covers both `__setitem__`
and `__delitem__`.

See
[`PySequenceMethods`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods).

| Builder method | CPython slot                                                                                              | Python operator            |
|----------------|-----------------------------------------------------------------------------------------------------------|----------------------------|
| `def_len`      | [`sq_length`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_length)                 | `len(obj)`                 |
| `def_getitem`  | [`sq_item`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_item)                     | `obj[i]`                   |
| `def_setitem`  | [`sq_ass_item`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_ass_item)             | `obj[i] = v`, `del obj[i]` |
| `def_contains` | [`sq_contains`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_contains)             | `x in obj`                 |
| `def_concat`   | [`sq_concat`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_concat)                 | `obj + other`              |
| `def_repeat`   | [`sq_repeat`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_repeat)                 | `obj * n`                  |
| `def_iconcat`  | [`sq_inplace_concat`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_concat) | `obj += other`             |
| `def_irepeat`  | [`sq_inplace_repeat`](https://docs.python.org/3/c-api/typeobj.html#c.PySequenceMethods.sq_inplace_repeat) | `obj *= n`                 |

### Number protocol

Arithmetic, bitwise, and numeric-conversion slots. Each binary operator has an
in-place counterpart (`def_iadd`, …) that backs the augmented assignment
operators. Binary slots may raise `PySlotError.not_implemented()` to return
`NotImplemented` so CPython tries the reflected operation on the other operand.

See
[`PyNumberMethods`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods).

Unary, conversion, and predicate slots:

| Builder method | CPython slot                                                                                | Python operator       |
|----------------|---------------------------------------------------------------------------------------------|-----------------------|
| `def_neg`      | [`nb_negative`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_negative) | `-obj`                |
| `def_pos`      | [`nb_positive`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_positive) | `+obj`                |
| `def_abs`      | [`nb_absolute`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_absolute) | `abs(obj)`            |
| `def_invert`   | [`nb_invert`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_invert)     | `~obj`                |
| `def_bool`     | [`nb_bool`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_bool)         | `bool(obj)`           |
| `def_int`      | [`nb_int`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_int)           | `int(obj)`            |
| `def_float`    | [`nb_float`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_float)       | `float(obj)`          |
| `def_index`    | [`nb_index`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_index)       | `operator.index(obj)` |

Binary slots (each has an `def_i<name>` in-place counterpart):

| Builder method                   | CPython slot                                                                                                                                                                                                                          | Python operator                                    |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `def_add` / `def_iadd`           | [`nb_add`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_add) / [`nb_inplace_add`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_add)                                                 | `a + b` / `a += b`                                 |
| `def_sub` / `def_isub`           | [`nb_subtract`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_subtract) / [`nb_inplace_subtract`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_subtract)                             | `a - b` / `a -= b`                                 |
| `def_mul` / `def_imul`           | [`nb_multiply`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_multiply) / [`nb_inplace_multiply`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_multiply)                             | `a * b` / `a *= b`                                 |
| `def_truediv` / `def_itruediv`   | [`nb_true_divide`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_true_divide) / [`nb_inplace_true_divide`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_true_divide)                 | `a / b` / `a /= b`                                 |
| `def_floordiv` / `def_ifloordiv` | [`nb_floor_divide`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_floor_divide) / [`nb_inplace_floor_divide`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_floor_divide)             | `a // b` / `a //= b`                               |
| `def_mod` / `def_imod`           | [`nb_remainder`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_remainder) / [`nb_inplace_remainder`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_remainder)                         | `a % b` / `a %= b`                                 |
| `def_divmod`                     | [`nb_divmod`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_divmod)                                                                                                                                               | `divmod(a, b)`                                     |
| `def_pow` / `def_ipow`           | [`nb_power`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_power) / [`nb_inplace_power`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_power)                                         | `a ** b` / `a **= b`                               |
| `def_matmul` / `def_imatmul`     | [`nb_matrix_multiply`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_matrix_multiply) / [`nb_inplace_matrix_multiply`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_matrix_multiply) | `a @ b` / `a @= b`                                 |
| `def_lshift` / `def_ilshift`     | [`nb_lshift`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_lshift) / [`nb_inplace_lshift`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_lshift)                                     | `a << b` / `a <<= b`                               |
| `def_rshift` / `def_irshift`     | [`nb_rshift`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_rshift) / [`nb_inplace_rshift`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_rshift)                                     | `a >> b` / `a >>= b`                               |
| `def_and` / `def_iand`           | [`nb_and`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_and) / [`nb_inplace_and`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_and)                                                 | `a & b` / `a &= b`                                 |
| `def_or` / `def_ior`             | [`nb_or`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_or) / [`nb_inplace_or`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_or)                                                     | <code>a &#124; b</code> / <code>a &#124;= b</code> |
| `def_xor` / `def_ixor`           | [`nb_xor`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_xor) / [`nb_inplace_xor`](https://docs.python.org/3/c-api/typeobj.html#c.PyNumberMethods.nb_inplace_xor)                                                 | `a ^ b` / `a ^= b`                                 |

## Implementation support code

The number protocol alone has ~30 slots, and every builder method admits
multiple user-signature flavors (pointer-receiver vs. value-receiver vs.
mut-receiver, raising vs. non-raising, returning `PythonObject` vs. any
`ConvertibleToPython` type). To keep that combinatorial explosion off the
public API, an internal `_SlotInstaller` struct in `adapters.mojo` hosts
all installation helpers as static methods (`_SlotInstaller.binary`,
`_SlotInstaller.unary_val`, `_SlotInstaller.ternary_conv_nr`, …), and a
sibling `_BfSlotInstaller` in `buffer.mojo` does the same for the buffer
protocol. The handful of builder `def_*` methods then each reduce to one
line that picks the right installer variant and forwards the user's
function as a template parameter.

### Adapter wrappers

Each CPython type slot has a fixed C function signature that the
interpreter calls directly (e.g. `tp_richcompare` is
`PyObject *(*)(PyObject *, PyObject *, int)`). Mojo cannot register a
user method against those signatures directly — methods take typed
receivers, raise `PySlotError`, and return `PythonObject` rather than a
raw `PyObjectPtr`. The adapter wrappers (`_unaryfunc_wrapper`,
`_binaryfunc_wrapper`, `_ternaryfunc_wrapper`, `_richcompare_wrapper`,
`_inquiry_wrapper`, `_mp_length_wrapper`, `_mp_subscript_wrapper`,
`_mp_ass_subscript_wrapper`, `_ssizeargfunc_wrapper`,
`_ssizeobjargproc_wrapper`, `_objobjproc_wrapper`, and
`_bf_getbuffer_wrapper`) sit between the two worlds: each is a small
`abi("C")` function that takes the raw `PyObjectPtr` arguments CPython
hands it, calls `_unwrap_self` to downcast `self`, invokes the
user-supplied `method` template parameter, and translates the result
back to whatever C type the slot's signature requires.

The wrappers also encode the per-slot error contract. For all slots, a
typed `except e: PySlotError` block branches on `e._variant` and either
calls `PyErr_SetString` with the matching `PyExc_*` global or — for
binary, ternary, and rich-compare slots, where the `not_implemented`
variant has Python-visible meaning — returns
`Py_NewRef(Py_NotImplemented())` so CPython falls back to the reflected
operation. Binary/ternary/richcompare wrappers also map any *prep-block*
failure (e.g. `_unwrap_self` failing because the LHS isn't our type
during a reflected call) to `Py_NotImplemented` rather than
`RuntimeError`, matching CPython's expectation for reflected dispatch.

### `_lift_*` and `_conv_*` helpers

User slot methods come in several shapes. A handler can take a
`UnsafePointer[T, MutAnyOrigin]` (the canonical form), a value receiver
`T`, or a `mut T`; it can be raising or non-raising; and number-protocol
return values can be either `PythonObject` or any type conforming to
`ConvertibleToPython`. Multiplying these axes out would force every
adapter wrapper to ship in a dozen variants. Instead, a single canonical
wrapper consumes a `def(UnsafePointer[T, MutAnyOrigin], …) raises
PySlotError -> PythonObject` shape, and a family of tiny `_lift_*` /
`_conv_*` template functions adapts each user signature to that shape
before the wrapper sees it.

The `_lift_*` helpers handle receiver and raising-ness — for example
`_lift_val_int_to_obj` takes a value-receiver, integer-arg, raising
method and produces a pointer-receiver version that calls
`method(ptr[], index)`. The `_conv_*` helpers handle the return type —
`_conv_ptr_r_binary` takes a method returning `R: ConvertibleToPython`
and produces one returning `PythonObject` by routing through `_to_py[R]`
(which itself translates `to_python_object`'s plain `Error` into
`PySlotError.value_error`). The builder `def_*` overloads then become
one-liners: `_SlotInstaller.binary_val[…]` already knows it needs
`_lift_val_obj_to_obj`, `binary_conv_r[…]` knows it needs
`_conv_ptr_r_binary`, and so on.

### `PySlotError` — the Mojo-native slot error type

CPython slots have a richer error contract than a single `Error` type can
express: a `tp_richcompare` slot may want to *raise* `TypeError`, *raise*
`ValueError`, or *return* `Py_NotImplemented`; a `sq_ass_item` slot picks
between `IndexError` and `TypeError`; and so on. Mojo today allows only one
error type per `raises` clause and only one `except` clause per `try` block (see
[Representing multiple error conditions](https://docs.modular.com/mojo/manual/errors#representing-multiple-error-conditions)),
so we collapse the contract into a single enumerated type. User slot methods
declare `raises PySlotError` and construct values via static factories:

```mojo
raise PySlotError.index_error("index out of range")
raise PySlotError.type_error("expected int")
raise PySlotError.not_implemented()
```

The wrapper's `except e:` block infers `e: PySlotError`, branches on
`e._variant`, and either calls
`cpython.PyErr_SetString(e.pyexc_global_name(), e.msg)` for the real-exception
variants or returns `Py_NewRef(cpython.Py_NotImplemented())` for the
`not_implemented` variant. The variants currently supported are `index_error`,
`type_error`, `value_error`, `key_error`, `attribute_error`, `overflow_error`,
`runtime_error`, and `not_implemented`; extending the set is a localized change
in `utils.mojo` plus one branch in `pyexc_global_name()`.
