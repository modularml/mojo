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

from collections import KeyElement, List
from collections.string import (
    String,
    ascii,
    atof,
    atol,
    chr,
    isdigit,
    islower,
    isprintable,
    isupper,
    ord,
)
from hashlib.hash import Hashable, hash

from builtin.anytype import AnyType, UnknownDestructibility
from builtin.bool import Bool, Boolable, ImplicitlyBoolable, all, any, bool
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
from builtin.coroutine import AnyCoroutine, Coroutine, RaisingCoroutine
from builtin.debug_assert import debug_assert
from builtin.dtype import DType
from builtin.equality_comparable import EqualityComparable
from builtin.error import Error
from builtin.file import FileHandle, open
from builtin.file_descriptor import FileDescriptor
from builtin.float_literal import FloatLiteral
from builtin.floatable import Floatable, FloatableRaising, float
from builtin.format_int import bin, hex, oct
from builtin.identifiable import Identifiable, StringableIdentifiable
from builtin.int import (
    Indexer,
    Int,
    Intable,
    IntableRaising,
    IntLike,
    index,
    int,
)
from builtin.int_literal import IntLiteral
from builtin.io import input, print
from builtin.len import Sized, SizedRaising, UIntSized, len
from builtin.math import (
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
from builtin.none import NoneType
from builtin.object import Attr, object
from builtin.range import range
from builtin.rebind import rebind
from builtin.repr import Representable, repr
from builtin.reversed import ReversibleRange, reversed
from builtin.simd import (
    SIMD,
    BFloat16,
    Byte,
    Float8e5m2,
    Float8e5m2fnuz,
    Float8e4m3,
    Float8e4m3fnuz,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Scalar,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from builtin.sort import partition, sort
from builtin.str import Stringable, StringableRaising, str
from builtin.string_literal import StringLiteral
from builtin.swap import swap
from builtin.tuple import Tuple
from builtin.type_aliases import (
    AnyTrivialRegType,
    ImmutableAnyOrigin,
    ImmutableOrigin,
    MutableAnyOrigin,
    MutableOrigin,
    Origin,
    OriginSet,
    StaticConstantOrigin,
)
from builtin.uint import UInt
from builtin.value import (
    BoolableCollectionElement,
    BoolableKeyElement,
    BytesCollectionElement,
    CollectionElement,
    CollectionElementNew,
    ComparableCollectionElement,
    Copyable,
    Defaultable,
    EqualityComparableCollectionElement,
    ExplicitlyCopyable,
    Movable,
    RepresentableCollectionElement,
    StringableCollectionElement,
)
from documentation import doc_private
from memory import AddressSpace, Pointer

from memory.span import AsBytes
from utils import Writable, Writer
