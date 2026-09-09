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
"""Generic request parsing with standardized error handling.

This module provides FastAPI dependencies for parsing and validating request
objects that implement the `from_fastapi_request` pattern. It centralizes
error handling logic to convert parsing and validation errors into appropriate
HTTP responses.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import TypeVar, cast

from fastapi import HTTPException, Request
from pydantic import ValidationError

logger = logging.getLogger("max.serve")

# Type variable for request classes
T = TypeVar("T")


async def parse_request_generic(
    request: Request,
    parser_class: type[T],
) -> T:
    """Parses a FastAPI request with standardized error handling.

    This function provides a generic interface for parsing FastAPI requests
    into typed request objects. It handles common error cases (missing required
    fields, validation errors) and converts them into appropriate HTTP exceptions.

    The ``parser_class`` supplies a ``from_fastapi_request`` classmethod that
    turns a request into a typed object:

    .. code-block:: python

        import asyncio
        from dataclasses import dataclass

        from max.serve.dependencies.request_parsing import parse_request_generic

        @dataclass
        class EchoRequest:
            model: str

            @classmethod
            async def from_fastapi_request(cls, request) -> "EchoRequest":
                return cls(model="llama")

        parsed = asyncio.run(
            parse_request_generic(request=None, parser_class=EchoRequest)
        )

    .. invisible-code-block: python

        assert parsed.model == "llama"

    Args:
        request: The incoming FastAPI request object.
        parser_class: A class with a ``from_fastapi_request`` classmethod that
            accepts a Request and returns an instance of the class.

    Returns:
        A parsed and validated request object of type ``T``.

    Raises:
        HTTPException: With status code 400 (BAD_REQUEST) if parsing fails
            due to a ValueError (for example, missing required fields like request_id).
        HTTPException: With status code 422 (UNPROCESSABLE_ENTITY) if Pydantic
            validation fails (for example, invalid field types or constraints).
    """
    try:
        # Cast is safe because we expect parser_class to have from_fastapi_request
        # that returns an instance of T
        result = await parser_class.from_fastapi_request(request)  # type: ignore[attr-defined]
        return cast(T, result)
    except ValidationError as e:
        logger.error("Request validation failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}",
        ) from e
    except ValueError as e:
        logger.error("Request parsing failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        ) from e


def create_request_parser(
    parser_class: type[T],
) -> Callable[[Request], Awaitable[T]]:
    """Creates a FastAPI dependency for parsing a specific request type.

    This factory function creates a reusable dependency that can be used with
    FastAPI's dependency injection system. The returned dependency automatically
    parses and validates incoming requests, converting errors to HTTP exceptions.

    Wrap the parser in ``Depends`` and attach it to a route so the handler
    receives an already-parsed, validated request:

    .. code-block:: python

        from dataclasses import dataclass

        from fastapi import Depends, FastAPI, Request

        from max.serve.dependencies.request_parsing import create_request_parser

        @dataclass
        class EchoRequest:
            model: str

            @classmethod
            async def from_fastapi_request(cls, request) -> "EchoRequest":
                return cls(model="llama")

        ParseEchoRequest = Depends(create_request_parser(EchoRequest))

        app = FastAPI()

        @app.post("/responses")
        async def create_response(
            request: Request,
            echo_request: EchoRequest = ParseEchoRequest,
        ) -> dict:
            return {"model": echo_request.model}

    .. invisible-code-block: python

        assert any(
            getattr(route, "path", None) == "/responses"
            for route in app.routes
        )

    Args:
        parser_class: A class with a ``from_fastapi_request`` classmethod.

    Returns:
        An async function that can be used as a FastAPI dependency with
        ``Depends()``.
    """

    async def dependency(request: Request) -> T:
        """Dependency function that parses the request."""
        return await parse_request_generic(request, parser_class)

    return dependency
