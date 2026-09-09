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
"""Worker-method type-hint introspection, plus the `CascadeValue` boundary type."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Callable
from typing import Any, TypeAlias, Union, get_args, get_origin, get_type_hints

import numpy as np

CascadeValue: TypeAlias = Union[
    "np.ndarray",
    bytes,
    str,
    float,
    bool,
    int,
    dict[str, "CascadeValue"],
    list["CascadeValue"],
    None,
]


def arg_types(method: Callable[..., Any]) -> dict[str, object | None]:
    """Returns the parameter hints of a method keyed by name in order.

    Every declared parameter is present (unannotated ones map to None) so
    the i-th entry always corresponds to the i-th positional argument.
    ``*args`` / ``**kwargs`` catch-alls are excluded.
    """
    hints = get_type_hints(method)
    return {
        name: hints.get(name)
        for name, param in inspect.signature(method).parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def return_type(method: Callable[..., Any]) -> object | None:
    """Returns the resolved return hint of a method, or None if unresolvable."""
    try:
        return get_type_hints(method).get("return")
    except Exception:
        return None


def async_elem_type(hint: object | None) -> object | None:
    """Returns the element type of an async-iterable hint.

    i.e. the ``T`` in ``AsyncIterator[T]``, or None for anything else.
    """
    origin = get_origin(hint)
    if isinstance(origin, type) and issubclass(origin, AsyncIterable):
        args = get_args(hint)
        return args[0] if args else None
    return None


def stream_elem_type(method: Callable[..., Any]) -> object | None:
    """Returns the element type of an async-generator method's return hint."""
    return async_elem_type(return_type(method))
