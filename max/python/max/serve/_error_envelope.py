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

"""The OpenAI-shaped error envelope shared by every MAX Serve error response.

Lives apart from the exception handlers in ``api_server`` so the request-session
middleware, which sits outside Starlette's exception middleware and therefore
has to build its own response, can return the same shape.
"""

from __future__ import annotations

from typing import Any

from max.serve.schemas.openai import Error, ErrorResponse

_OPENAI_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def openai_error_body(status_code: int, message: str) -> dict[str, Any]:
    """Builds the OpenAI ``error`` envelope for ``status_code``.

    Args:
        status_code: HTTP status code the response will carry.
        message: Human-readable description of the failure.

    Returns:
        The serialized ``ErrorResponse`` body.
    """
    error_type = _OPENAI_ERROR_TYPES.get(
        status_code,
        "invalid_request_error" if status_code < 500 else "api_error",
    )
    return ErrorResponse(
        error=Error(
            code=str(status_code), message=message, param="", type=error_type
        )
    ).model_dump()
