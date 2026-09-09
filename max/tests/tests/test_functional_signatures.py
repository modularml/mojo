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
"""Verify ``inspect.signature(F.X)`` reports ``Tensor`` for functional ops.

The graph ops are annotated with ``TensorValueLike`` / ``StrongTensorValueLike``
/ ``TensorValue``. The functional surface wraps them and should display
``Tensor`` instead. These tests assert that the signature substitution
applied inside ``functional()`` works for a representative set of ops
(unary, multi-input, list-return, tuple-return, and binary with scalar
promotion).
"""

import inspect
import re
from collections.abc import Callable
from typing import Any

import pytest
from max.experimental import functional as F
from max.experimental.functional._signatures import TENSOR_VALUE_ALIAS_NAMES
from max.experimental.tensor import Tensor
from max.graph import TensorValue

# Names of the graph-op wrapper-type aliases that must NOT leak into any
# rendered signature or ``__annotations__`` on the functional surface.
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(TENSOR_VALUE_ALIAS_NAMES) + r")\b")

# ``inspect.formatannotation`` renders class annotations as
# ``{__module__}.{__qualname__}``; build the fragments dynamically so the
# tests survive moving these classes between modules.
_TENSOR = f"{Tensor.__module__}.{Tensor.__qualname__}"
_TENSOR_VALUE = f"{TensorValue.__module__}.{TensorValue.__qualname__}"

# (op, fragments that must appear in str(inspect.signature(op))).
SIGNATURE_CASES = [
    # Unary: TensorValueLike on input, TensorValue on return.
    (F.exp, [f"x: {_TENSOR}", f"-> {_TENSOR}"]),
    (F.relu, [f"x: {_TENSOR}", f"-> {_TENSOR}"]),
    # StrongTensorValueLike on input.
    (F.cast, [f"x: {_TENSOR}", f"-> {_TENSOR}"]),
    (F.argsort, [f"x: {_TENSOR}", f"-> {_TENSOR}"]),
    # Multi-input.
    (F.matmul, [f"lhs: {_TENSOR}", f"rhs: {_TENSOR}"]),
    # Iterable[TensorValueLike] container.
    (F.concat, [f"Iterable[{_TENSOR}]"]),
    # tuple return.
    (F.top_k, [f"tuple[{_TENSOR}, {_TENSOR}]"]),
    # list return — note: cond/while_loop are defined with
    # ``from __future__ import annotations`` so their annotations are
    # the literal source strings.
    (F.cond, ["-> 'list[Tensor]'"]),
    (F.while_loop, ["-> 'list[Tensor]'"]),
    # Scalar-promotion binary: ``wrapper`` defines its own
    # ``Tensor | float`` annotation (a string under future-annotations).
    (
        F.add,
        [
            "lhs: 'Tensor | float'",
            "rhs: 'Tensor | float'",
            "-> 'Tensor'",
        ],
    ),
]


@pytest.mark.parametrize("op,expected", SIGNATURE_CASES)
def test_signature_substitutes_tensor(
    op: Callable[..., Any], expected: list[str]
) -> None:
    rendered = str(inspect.signature(op))
    for fragment in expected:
        assert fragment in rendered, (
            f"{op.__name__}: expected {fragment!r} in {rendered!r}"
        )
    # Word-boundary check so ``Tensor`` itself isn't flagged when it's
    # a substring of ``TensorValueLike`` / ``TensorValue``.
    leak = _FORBIDDEN_RE.search(rendered)
    assert leak is None, (
        f"{op.__name__}: leaked {leak.group(0)!r} in {rendered!r}"
    )


@pytest.mark.parametrize(
    "op,expected",
    [
        # Bare ``TensorValue`` in a parameter is preserved (only ``*Like``
        # aliases get substituted in parameter position). Return position
        # still substitutes everything, so ``-> Tensor``.
        (
            F.repeat_interleave,
            [
                f"x: {_TENSOR}",
                f"repeats: int | {_TENSOR_VALUE}",
                f"-> {_TENSOR}",
            ],
        ),
        (
            F.layer_norm,
            [
                f"input: {_TENSOR_VALUE}",
                f"gamma: {_TENSOR}",
                f"-> {_TENSOR}",
            ],
        ),
    ],
)
def test_signature_preserves_bare_tensorvalue_in_params(
    op: Callable[..., Any], expected: list[str]
) -> None:
    """Position-aware rule: bare ``TensorValue`` in a parameter stays as
    ``TensorValue`` because the graph op's body genuinely needs one (e.g.
    ``repeat_interleave._promote_repeats`` does an ``isinstance`` check
    that ``Tensor`` would fail).
    """
    rendered = str(inspect.signature(op))
    for fragment in expected:
        assert fragment in rendered, (
            f"{op.__name__}: expected {fragment!r} in {rendered!r}"
        )


def test_binary_promotion_preserves_op_name() -> None:
    # Before this PR, ``F.add.__name__`` was ``"wrapper"`` because
    # ``_binary_with_scalar_promotion`` didn't call ``functools.wraps``.
    assert F.add.__name__ == "add"
    assert F.sub.__name__ == "sub"
    assert F.mul.__name__ == "mul"


def test_annotations_dict_substitutes_tensor() -> None:
    """Sphinx autodoc reads ``__annotations__`` directly, not via
    ``inspect.signature``, when rendering parameter and return-type
    sections. Lock the substitution in at the ``__annotations__`` level.
    """
    # Class-annotation case (graph op uses bare ``TensorValueLike`` /
    # ``TensorValue`` — no ``from __future__ import annotations``).
    assert F.abs.__annotations__ == {"x": Tensor, "return": Tensor}
    assert F.exp.__annotations__ == {"x": Tensor, "return": Tensor}

    # String-annotation case (graph op or wrapper uses
    # ``from __future__ import annotations``).
    assert F.cond.__annotations__["pred"] == "Tensor"
    assert F.cond.__annotations__["return"] == "list[Tensor]"
