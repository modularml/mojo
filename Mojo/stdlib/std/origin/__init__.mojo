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
"""Defines Mojo's origin types.

An origin is a compile-time value that names the variable a reference
borrows from, and whether that reference permits mutation. The compiler uses
origins to check that a reference never outlives the value it points to,
enforce mutable exclusivity between references, and destroy values as soon as
their last use.

Most origins are inferred automatically—from `ref` arguments, from parametric
types like `Pointer` and `Span`, or from `origin_of()`. Reach for `Origin` and
its aliases (`ImmOrigin`, `MutOrigin`, `ImmUntrackedOrigin`,
`MutUntrackedOrigin`, `ImmStaticOrigin`) when an API needs to name an origin
explicitly.

For the full picture, see the
[lifetimes guide](https://mojolang.org/docs/manual/values/lifetimes/).
"""

comptime ImmOrigin = Origin[mut=False]
"""Immutable origin reference type."""

comptime MutOrigin = Origin[mut=True]
"""Mutable origin reference type."""

comptime AnyOrigin[*, mut: Bool] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.any.origin : !lit.origin<`, +mut._mlir_value, `>`
    ],
]()
"""An origin that might access any memory value.

Parameters:
    mut: Whether the origin is mutable.
"""

comptime ImmutAnyOrigin = AnyOrigin[mut=False]
"""The immutable origin that might access any memory value."""

comptime MutAnyOrigin = AnyOrigin[mut=True]
"""The mutable origin that might access any memory value."""

comptime UnsafeAnyOrigin[*, mut: Bool] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.any.origin : !lit.origin<`, +mut._mlir_value, `>`
    ],
]()
"""The universal origin: an unsafe origin that might alias any memory value.

Parameters:
    mut: Whether the origin is mutable.

Because a reference with this origin might alias any live value, it forces the
lifetime checker into its most conservative behavior, defeating the guarantees
the origin system is meant to provide:

- It extends unrelated lifetimes. Every other value in scope is kept alive for
  as long as the reference is live, even values it never points to, effectively
  halting ASAP destruction.
- It hides unused-variable warnings, since the compiler treats every in-scope
  variable as potentially aliased.
- It disables mutable exclusivity checking, since the compiler cannot prove
  which value the reference aliases.

**Safety:** This is a temporary compiler escape hatch from Mojo's early days,
not a capability to reach for. It will never be stabilized and is slated for
deprecation and removal. The `Unsafe` prefix marks every use as a place to
migrate away from; prefer a concrete origin so the compiler can continue to
track lifetimes and exclusivity.
"""

comptime ImmUnsafeAnyOrigin = UnsafeAnyOrigin[mut=False]
"""The immutable universal origin that might alias any memory value.

This is an unsafe escape hatch slated for removal. See `UnsafeAnyOrigin`.
"""

comptime MutUnsafeAnyOrigin = UnsafeAnyOrigin[mut=True]
"""The mutable universal origin that might alias any memory value.

This is an unsafe escape hatch slated for removal. See `UnsafeAnyOrigin`.
"""

comptime UntrackedOrigin[*, mut: Bool] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.union<> : !lit.origin<`,
        +mut._mlir_value,
        `>`,
    ],
]()
"""An origin the lifetime checker does not track, because it aliases no
existing value.

Parameters:
    mut: Whether the origin is mutable.

An untracked origin is the empty origin: it promises the reference aliases no
value the compiler is managing, so there is nothing for the lifetime checker to
track or extend. That is exactly the behavior you want when interfacing with
memory from outside the Mojo program. For example, the pointer returned by
`alloc()` carries an untracked origin, because the allocated block aliases no
Mojo-owned value.
"""

comptime ImmUntrackedOrigin = UntrackedOrigin[mut=False]
"""An immutable origin the lifetime checker does not track."""

comptime MutUntrackedOrigin = UntrackedOrigin[mut=True]
"""A mutable origin the lifetime checker does not track."""

# Static constants are a named subset of the global origin.
comptime ImmStaticOrigin = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.field<`,
        `#lit.static.origin : !lit.origin<false>`,
        `, "__constants__"> : !lit.origin<false>`,
    ]
]()
"""An origin for strings and other always-immutable static constants."""

comptime OriginSet = __mlir_type.`!lit.origin.set`
"""A set of origin parameters."""


comptime _lit_origin_type_of_mut[mut: Bool] = __mlir_type[
    `!lit.origin<`, mut._mlir_value, `>`
]


struct Origin[mut: Bool, _mlir_origin: _lit_origin_type_of_mut[mut], //](
    TrivialRegisterPassable
):
    """This represents a origin reference for a memory value.

    Parameters:
        mut: Whether the origin is mutable.
        _mlir_origin: The raw MLIR origin value.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Construct an Origin."""
        pass

    @always_inline("builtin")
    @implicit
    def __init__(v: Origin) -> ImmOrigin[_mlir_origin=v._mlir_origin]:
        """Implicitly convert an origin to an immutable one.

        Args:
            v: The origin to convert.
        """
        return {}

    @always_inline("builtin")
    @staticmethod
    def unsafe_mut_cast[
        dest_mut: Bool
    ]() -> Origin[
        _mlir_origin=__mlir_attr[
            `#lit.origin.mutcast<`,
            Self._mlir_origin,
            `> : !lit.origin<`,
            dest_mut._mlir_value,
            `>`,
        ]
    ]:
        """Cast this origin to a different mutability, potentially introducing
        more mutability, which is an unsafe operation.

        Parameters:
            dest_mut: The desired mutability of the resulting origin.

        Returns:
            The same origin but with a new specified mutability.
        """
        return {}

    comptime equals[rhs: Origin]: Bool = __mlir_attr[
        `#lit.origin.eq<`,
        Self._mlir_origin,
        `, `,
        rhs._mlir_origin,
        `> : i1`,
    ]
    """Is true if self is equal to rhs.  This predicate can only be
    used in 'where' clauses and other expressions evaluated at parse time.
    It may not be used in 'comptime if' and similar expressions.

    Parameters:
        rhs: The other origin to compare to.
    """

    comptime contains[element: Origin]: Bool = Self.equals[
        origin_of(Self._mlir_origin, element._mlir_origin)
    ]
    """Is true if self is a superset of element.  This predicate can
    only be used in 'where' clauses and other expressions evaluated at parse
    time. It may not be used in 'comptime if' and similar expressions.

    Parameters:
        element: The origin to check if it is a subset of Self.
    """

    comptime _get_owned_interior[name: StringLiteral] = Origin[
        _mlir_origin=__mlir_attr[
            `#lit.interior.origin<`,
            Self._mlir_origin,
            `, `,
            name.value,
            `> : `,
            type_of(Self._mlir_origin),
        ]
    ]()
    """Returns an interior sub-origin of this origin.

    Interior origins are an experimental feature that name storage that is owned
    by a container, usually reached through a pointer indirection or inlined
    into it. The base origin governs invalidation of the interior origin.

    Parameters:
        name: A compile-time string that identifies the interior object in
            diagnostics and invalidation tracking.
    """

    comptime _subtree = Origin[
        _mlir_origin=__mlir_attr[
            `#lit.origin.subtree<`,
            Self._mlir_origin,
            `> : `,
            type_of(Self._mlir_origin),
        ]
    ]()
    """Returns a subtree view over this origin and every origin derived from
    it.

    A subtree origin is an experimental feature that does not name one
    specific storage location. Instead, it represents a reference that could
    point to storage governed by this origin itself, or by any origin formed
    by applying field or interior origin projections beneath it. This lets an
    API abstract over the precise interior region being referenced while
    still preserving ownership and invalidation semantics.
    """
