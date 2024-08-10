# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the prelude package.  This package provide the public entities
  that are automatically imported into every Mojo program.
"""

from builtin.anytype import AnyType
from builtin.bool import Boolable, ImplicitlyBoolable, Bool, bool, any, all
from builtin.breakpoint import breakpoint
from builtin.builtin_list import (
    ListLiteral,
    VariadicList,
    VariadicListMem,
    VariadicPack,
)
from builtin.builtin_slice import Slice, slice
from builtin.comparable import Comparable
from builtin.constrained import constrained
from builtin.coroutine import Coroutine, RaisingCoroutine, AnyCoroutine
from builtin.debug_assert import debug_assert
from builtin.dtype import DType
from builtin.equality_comparable import EqualityComparable
from builtin.error import Error
from builtin.file import open, FileHandle
from builtin.file_descriptor import FileDescriptor
from builtin.float_literal import FloatLiteral
from builtin.format_int import bin, hex, oct
from builtin.hash import hash, Hashable
from builtin.identifiable import Identifiable, StringableIdentifiable
from builtin.int import (
    Int,
    IntLike,
    Intable,
    IntableRaising,
    Indexer,
    index,
    int,
)
from builtin.int_literal import IntLiteral
from builtin.io import print
from builtin.len import Sized, UIntSized, SizedRaising, len
from builtin.math import (
    Absable,
    abs,
    divmod,
    max,
    min,
    Powable,
    pow,
    Roundable,
    round,
)
from builtin.none import NoneType
from builtin.object import Attr, object
from builtin.range import range
from builtin.rebind import rebind
from builtin.repr import Representable, repr
from builtin.reversed import ReversibleRange, reversed
from builtin.sort import sort, partition
from builtin.str import Stringable, StringableRaising, str
from builtin.string_literal import StringLiteral
from builtin.swap import swap
from builtin.tuple import (
    Tuple,
)
from builtin.type_aliases import (
    AnyTrivialRegType,
    ImmutableLifetime,
    MutableLifetime,
    ImmutableStaticLifetime,
    MutableStaticLifetime,
    LifetimeSet,
    AnyLifetime,
)
from builtin.uint import UInt
from builtin.value import (
    Movable,
    Copyable,
    ExplicitlyCopyable,
    Defaultable,
    CollectionElement,
    CollectionElementNew,
    StringableCollectionElement,
    EqualityComparableCollectionElement,
    ComparableCollectionElement,
    RepresentableCollectionElement,
    BoolableKeyElement,
    BoolableCollectionElement,
)
from builtin.simd import (
    Scalar,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    BFloat16,
    Float16,
    Float32,
    Float64,
    SIMD,
)
from builtin.type_aliases import AnyTrivialRegType

from collections import KeyElement, List
from collections.string import (
    String,
    ord,
    chr,
    ascii,
    atol,
    atof,
    isdigit,
    isupper,
    islower,
    isprintable,
)
from memory import UnsafePointer, Reference, AddressSpace
from utils import StringRef
from utils._format import Formattable, Formatter

# Private things
from builtin._documentation import doc_private
from utils._visualizers import lldb_formatter_wrapping_type

# Load-bearing ones to remove
from sys import alignof, sizeof, bitwidthof, simdwidthof
from memory import bitcast
from os import abort
from sys.ffi import external_call
