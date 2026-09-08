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
# RUN:  %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# Test that merging two type values through a comptime ternary preserves the
# common (intersection) trait bound *and* the conditional-conformance
# constraint carried by that bound.


# The field's trait bound. Deliberately does NOT require `Deinitable`.
trait Storage(Defaultable):
    pass


def predicate() -> Bool:
    pass


@explicit_destroy("StorageA must be explicitly destroyed.")
struct StorageA[T: AnyType](
    Deinitable where conforms_to(T, Deinitable),
    Storage, Movable where False,
):
    def __init__(out self):
        pass

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass


@explicit_destroy("StorageB must be explicitly destroyed.")
struct StorageB[T: AnyType](
    Deinitable where conforms_to(T, Deinitable),
    Storage, Movable where False,
):
    def __init__(out self):
        pass

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass


# CHECK-LABEL: lit.struct.decl @Container
@explicit_destroy("Container must be explicitly destroyed.")
struct Container[T: AnyType](
    Deinitable where conforms_to(T, Deinitable), Movable where False,
):
    # The merged `_Storage` type value must be a constrained trait bound.

    # CHECK: lit.alias.decl {{.*}}_Storage{{.*}}: !constrained_{{.*}}Deinitable{{.*}}Storage
    comptime _Storage = StorageA[Self.T] if predicate() else StorageB[Self.T]

    var _storage: Self._Storage

    def __init__(out self):
        self._storage = Self._Storage()

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        self._storage^.__deinit__()
