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
"""Provides test types for validating reflection-based trait implementations."""


@fieldwise_init
struct SimplePoint(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """A simple struct using default reflection-based for traits."""

    var x: Int
    """The x coordinate."""
    var y: Int
    """The y coordinate."""

    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait
    # Uses default reflection-based __hash__ from Hashable trait


@fieldwise_init
struct NestedStruct(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """A struct with nested fields using default reflection-based for traits."""

    var point: SimplePoint
    """The nested struct."""

    var name: String
    """Another nested field."""

    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait
    # Uses default reflection-based __hash__ from Hashable trait


@fieldwise_init
struct EmptyStruct(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """A struct with no fields using default reflection-based for traits."""

    pass
    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait
    # Uses default reflection-based __hash__ from Hashable trait
