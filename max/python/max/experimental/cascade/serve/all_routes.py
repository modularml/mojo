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
"""Provide a router that exposes all supported cascade inference routes."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from max.experimental.cascade.core import Runtime
from max.experimental.cascade.interfaces.gen_ai import GenAIInterface, Modality
from max.experimental.cascade.serve import chat_completions, open_responses


async def build_router(
    pipeline: GenAIInterface,
    runtime: Runtime,
    emit_reasoning_content: bool = False,
) -> APIRouter:
    """Auto-configure routes based on what the pipeline can generate.

    ``runtime`` is the one ``pipeline`` was deployed on; routes that need
    serving-side workers of their own deploy them there.
    """
    router = APIRouter()

    # The router is only built (and served) after the pipeline's workers are
    # deployed, so a plain 200 here signals readiness. This matches the
    # ``/health`` endpoint standard tooling (load balancers, k8s probes, and the
    # serving benchmark's readiness check) expects.
    @router.get("/health", response_class=PlainTextResponse)
    async def health() -> str:
        return "OK"

    output_modalities = pipeline.supported_output_modalities()

    if Modality.TEXT in output_modalities:
        router.include_router(
            await chat_completions.build_router(
                pipeline, runtime, emit_reasoning_content
            )
        )

    if Modality.IMAGE in output_modalities:
        router.include_router(open_responses.build_router(pipeline))

    return router
