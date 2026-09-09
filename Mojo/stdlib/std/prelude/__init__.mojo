# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Standard library prelude: fundamental types, traits, and operations auto-imported.

This package's contents form the basic vocabulary of Mojo programming that every
developer uses. It is implicitly imported to every Mojo program.

The `prelude` package contains the core types, traits, and functions that are
automatically imported into every Mojo program. It provides the foundational
building blocks of the language including basic types (Int, String, Bool),
essential traits (Copyable, Movable, Equatable), memory primitives (Pointer,
Span), and common operations (print, len, range). This package defines the
default namespace that makes Mojo code immediately usable without explicit
imports.
"""

from std.collections import (
    Dict,
    Array,
    KeyElement,
    List,
    Optional,
    ImmSpan,
    MutSpan,
    Span,
)
from std.collections.string import (
    Codepoint,
    ImmStringSpan,
    ImmStringSlice,
    MutStringSpan,
    MutStringSlice,
    StaticString,
    String,
    StringSpan,
    StringSlice,
    ascii,
    atof,
    atol,
    chr,
    ord,
)
from std.format import Writable, Writer, repr
from std.hashlib.hash import Hashable, hash
from std.io.file import FileHandle, open
from std.io.file_descriptor import FileDescriptor
from std.io.io import input, print

from std.builtin.anytype import (
    Some,
    SomeTypeList,
)
from std.builtin.bool import Bool, Boolable, all, any
from std.builtin.breakpoint import breakpoint
from std.builtin.builtin_slice import Slice, slice
from std.builtin.comparable import Comparable, Equatable
from std.builtin.debug_assert import debug_assert
from std.builtin.dtype import DType
from std.builtin.error import Error
from std.builtin.float_literal import FloatLiteral
from std.builtin.floatable import Floatable, FloatableRaising
from std.builtin.format_int import bin, hex, oct
from std.builtin.identifiable import Identifiable
from std.builtin.int import (
    Indexer,
    Intable,
    IntableRaising,
    index,
)
from std.builtin.int_literal import IntLiteral
from std.builtin.len import Sized, SizedRaising, len
from std.math.math import (
    Absable,
    Powable,
    Roundable,
    abs,
    divmod,
    max,
    min,
    pow,
    round,
)
from std.builtin.none import NoneType
from std.builtin.range import range
from std.builtin.rebind import rebind, rebind_var
from std.builtin.reversed import ReversibleRange, reversed
from std.builtin.simd_length import SIMDLength
from std.builtin.simd import (
    SIMD,
    BFloat16,
    Byte,
    Float4_e2m1fn,
    Float8_e4m3fn,
    Float8_e4m3fnuz,
    Float8_e5m2,
    Float8_e5m2fnuz,
    Float8_e8m0fnu,
    Float16,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Int256,
    Scalar,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    UInt256,
    UInt,
)
from std.builtin.sort import partition, sort
from std.builtin.string_literal import StringLiteral
from std.builtin.swap import swap
from std.builtin.tuple import Tuple
from std.builtin.type_aliases import Never
from std.builtin.value import (
    Defaultable,
    materialize,
    RegisterPassable,
    TrivialRegisterPassable,
)
from std.builtin.variadics import (
    TypeList,
    ParameterList,
    VariadicList,
    VariadicPack,
)
from std.documentation import doc_hidden
from std.iter import (
    Iterable,
    IterableOwned,
    Iterator,
    StopIteration,
    enumerate,
    iter,
    map,
    next,
    zip,
)
from std.memory import (
    alloc,
    AddressSpace,
    ImmOpaquePointer,
    MutOpaquePointer,
    OpaquePointer,
    OptionalPointer,
    ImmPointer,
    MutPointer,
    Pointer,
    UnsafePointer,
)
from std.origin import (
    AnyOrigin,
    ImmutAnyOrigin,
    ImmOrigin,
    MutAnyOrigin,
    MutOrigin,
    Origin,
    OriginSet,
    ImmStaticOrigin,
    UntrackedOrigin,
    ImmUntrackedOrigin,
    MutUntrackedOrigin,
    UnsafeAnyOrigin,
    MutUnsafeAnyOrigin,
    ImmUnsafeAnyOrigin,
)
from std.reflection import reflect
from std.traits import (
    AnyType,
    Copyable,
    Deinitable,
    ImplicitlyCopyable,
    Movable,
)
from std.traits.deinitable import deinit
