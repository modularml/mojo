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
"""Interfaces for text generation and generation mixins."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..config import PipelineConfig

from max.pipelines.context import (
    GenerationStatus,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.kv_cache.paged_kv_cache import PagedKVCacheManagerInterface
from max.pipelines.modeling.types import (
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    RequestType,
    TextGenerationInputs,
)

from .pipeline_model import PipelineModelWithKVCache


@runtime_checkable
class GenerateMixin(Protocol[TextGenerationContextType, RequestType]):
    """Protocol for pipelines that support text generation."""

    _pipeline_model: PipelineModelWithKVCache[TextGenerationContextType]

    @property
    def kv_manager(self) -> PagedKVCacheManagerInterface:
        """Returns the KV cache managers for this pipeline."""
        ...

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        ...

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType, npt.NDArray[np.integer[Any]], RequestType
    ]:
        """Returns the tokenizer for this pipeline."""
        ...

    def execute(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Executes the pipeline for the given inputs."""
        ...

    def release(self, request_id: RequestID) -> None:
        """Releases resources for the given request."""
        ...

    def generate(
        self, prompts: RequestType | list[RequestType]
    ) -> list[TextGenerationOutput]:
        """Generates outputs for the given prompts."""
        if not isinstance(prompts, list):
            prompts = [prompts]

        async def _generate() -> list[TextGenerationOutput]:
            res = [
                TextGenerationOutput(
                    request_id=prompt.request_id,
                    tokens=[],
                    final_status=GenerationStatus.ACTIVE,
                )
                for prompt in prompts
            ]
            async for outputs in self.generate_async(prompts):
                for i, output in enumerate(outputs):
                    res[i].tokens.extend(output.tokens)
                    if output.is_done:
                        res[i].final_status = output.final_status
            return res

        return asyncio.run(_generate())

    async def generate_async(
        self, prompts: RequestType | list[RequestType]
    ) -> Any:
        """Generates outputs asynchronously for the given prompts."""
        if not isinstance(prompts, list):
            prompts = [prompts]

        context_batch: list[TextGenerationContextType] = []
        for prompt in prompts:
            context = await self.tokenizer.new_context(prompt)
            context_batch.append(context)

        data_parallel_degree = self.pipeline_config.model.data_parallel_degree

        # Create inputs to the model. If data parallelism is enabled, group them
        # by replica.
        batches: list[list[TextGenerationContextType]] = []
        batch_to_replica_idx: dict[RequestID, int] = {}
        batches = [[] for _ in range(data_parallel_degree)]
        for i, context in enumerate(context_batch):
            req_id = context.request_id
            # Use whatever replica the main models KVCache recommends.
            replica_idx = i % data_parallel_degree
            # Claim the slot for the KV cache manager
            self.kv_manager.claim(context, replica_idx=replica_idx)
            batches[replica_idx].append(context)
            batch_to_replica_idx[req_id] = replica_idx

        inputs = TextGenerationInputs(
            batches=batches,
        )

        # Generate outputs until all requests are done.

        done = 0

        try:
            while done < len(context_batch):
                for replica_batch in batches:
                    for ctx in replica_batch:
                        self.kv_manager.alloc(ctx)

                step_outputs = self.execute(inputs)

                # Filter out all responses for requests that are already released.
                # We can get a response for a request that is already released due to
                # the quirk of overlap scheduling where the pipeline may produce an extra
                # token after EOS. `batch_to_replica_idx` holds the requests still
                # running.
                step_outputs = {
                    request_id: output
                    for request_id, output in step_outputs.items()
                    if request_id in batch_to_replica_idx
                }

                outputs = []
                for request_id, output in step_outputs.items():
                    outputs.append(output)
                    if output.is_done:
                        done += 1
                        # Drop the request from the batch passed to the next
                        # call to execute, and from the still-running set the
                        # filter above reads: the two must stay in sync, or a
                        # post-EOS extra token gets past the filter and finds
                        # no context to release here.
                        replica_idx = batch_to_replica_idx.pop(request_id)
                        replica_batch = batches[replica_idx]
                        for idx, ctx in enumerate(replica_batch):
                            if ctx.request_id == request_id:
                                done_ctx = replica_batch.pop(idx)
                                break
                        else:
                            raise KeyError(
                                "Request ID not found in replica batch: "
                                f"{request_id}"
                            )

                        # Disjoint layers, not a double free: concrete
                        # `release` implementations leave the primary KV
                        # cache alone and free what the manager cannot --
                        # recurrent state slots and encoder-cache refs.
                        self.kv_manager.release(done_ctx)
                        self.release(request_id)

                if outputs:
                    yield outputs

                # Yield to the event loop.  If at no other point (e.g.
                # tokenizer.decode which we await earlier does not yield to the
                # event loop), it will be at this point that we'll receive a
                # CancelledError if our future was canceled (e.g., we received a
                # SIGINT).
                await asyncio.sleep(0)
            assert all(len(batch) == 0 for batch in batches)
        finally:
            # Release remaining requests if the generation was interrupted.
            for batch in batches:
                for context in batch:
                    if self.kv_manager.contains(context):
                        self.kv_manager.release(context)
                    self.release(context.request_id)
