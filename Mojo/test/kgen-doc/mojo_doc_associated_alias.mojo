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

# RUN: kgen-doc %s | FileCheck %s


trait TraitWithAlias:
    # CHECK-DAG: "summary": "It's a trait, with an alias."
    """It's a trait, with an alias."""

    # CHECK-DAG: "traits"
    # CHECK-DAG: "aliases"
    # CHECK-DAG: "kind": "alias",
    # CHECK-DAG: "name": "N",
    # CHECK-DAG: "summary": "This is the alias."
    comptime N: Int
    """This is the alias."""


def _is_positive(x: Int) -> Bool:
    return x > 0


trait IsEven:
    pass


trait TraitWithParametricAlias:
    """A trait carrying parametric associated aliases with trailing where clauses.
    """

    # Trailing where clause with a non-trait-conformance predicate is preserved
    # verbatim in the signature.
    # CHECK-DAG: "name": "AssocAliasWithWhere",
    # CHECK-DAG: "signature": "comptime AssocAliasWithWhere[N: Int] where _is_positive(N)"
    comptime AssocAliasWithWhere[N: Int]: Int where _is_positive(N)
    """An associated parametric alias with a non-mergeable trailing where clause.

    Parameters:
        N: A positive integer parameter.
    """

    # Trailing trait-conformance constraint merges into the parameter's type
    # bounds, so it does not appear as a separate `where` clause.
    # CHECK-DAG: "name": "AssocAliasWithTraitConformance",
    # CHECK-DAG: "signature": "comptime AssocAliasWithTraitConformance[T: AnyType & IsEven]"
    comptime AssocAliasWithTraitConformance[
        T: AnyType
    ]: AnyType where conforms_to(T, IsEven)
    """An associated parametric alias whose trailing trait-conformance constraint merges into the parameter bound.

    Parameters:
        T: A type that must conform to IsEven.
    """
