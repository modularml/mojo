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
"""Cryptographic and non-cryptographic hashing with customizable algorithms.

The `hashlib` package provides hashing functionality for computing hash values
of data. It defines the core hashing infrastructure through the `Hasher` trait
for implementing hash algorithms and the `Hashable` trait for types that can be
hashed. The package supports both compile-time and runtime hashing with
pluggable hash algorithm implementations.

Use this package for implementing hash-based data structures, creating hashable
types, computing checksums, or building custom hash algorithms. Types that
implement `Hashable` can be used as dictionary keys or in sets.
"""

from .hash import Hashable, hash
from .hasher import Hasher, default_comp_time_hasher, default_hasher
