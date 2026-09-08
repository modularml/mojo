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

# Test errors for duplicate traits with conflicting conditional conformances
# in struct inheritance lists.
#
# When the same trait appears multiple times in a struct's inheritance list
# (either directly or through trait compositions), the conditional conformance
# constraints must agree.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===========================================================================
# Duplicate trait: unconditional + conditional
# ===========================================================================


trait DupTraitA:
    pass


struct DupUnconditionalAndConditional[T: Movable](
    DupTraitA,
    # expected-error @below {{trait ''DupTraitA'' appears multiple times in the conformance list with different constraints}}
    DupTraitA where conforms_to(T, Copyable),
    Movable,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


# ===========================================================================
# Duplicate trait: different conditional constraints
# ===========================================================================


trait DupTraitB:
    pass


struct DupDifferentConstraints[T: Movable](
    DupTraitB where conforms_to(T, Copyable),
    # expected-error @below {{trait ''DupTraitB'' appears multiple times in the conformance list with different constraints}}
    DupTraitB where conforms_to(T, Intable),
    Movable,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


# ===========================================================================
# Trait composition + standalone with conflicting constraints
# ===========================================================================
# A & B where cond gives both A and B the condition. Listing A again without
# a condition conflicts.


trait CompA:
    pass


trait CompB:
    pass


struct CompositionConflictsWithStandalone[T: Movable](
    CompA & CompB where conforms_to(T, Copyable),
    # expected-error @below {{trait ''CompA'' appears multiple times in the conformance list with different constraints}}
    CompA,
    Movable,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^
