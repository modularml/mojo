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
"""General utils: indexing, variants, static tuples, and thread synchronization.

The `utils` package provides foundational data structures and utilities used
throughout the Mojo standard library. It includes types for multi-dimensional
indexing, type-safe unions, fixed-size tuples, and thread synchronization
primitives. These tools solve common programming patterns that don't fit neatly
into other stdlib packages.

Use this package when you need low-level building blocks for data structures,
generic programming with sum types, or fine-grained control over threading and
indexing operations.

`coord` (`Coord`, `coord()`, `Idx`, and related APIs) lives in `std.utils.coord`
and is not re-exported from this package root so callers use explicit imports
and avoid widening every `from std.utils import` dependency surface.
"""

from .index import Index, IndexList, product
from .lock import BlockingScopedLock, BlockingSpinLock, SpinWaiter
from .static_tuple import StaticTuple
from .variant import Variant
