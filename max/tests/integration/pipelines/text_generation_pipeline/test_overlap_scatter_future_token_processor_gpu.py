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

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.context.context import FUTURE_TOKEN
from max.pipelines.lib.interfaces import ModelInputs
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    AsyncBatch,
    RealizeFutureTokenProcessor,
)
from max.pipelines.modeling.types import TextGenerationInputs

OOB_IDX = np.iinfo(np.int32).min


@dataclass
class RaggedModelInputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer


def _create_context(
    active_tokens: list[int], chunk: int | None = None
) -> TextContext:
    context = TextContext(
        max_length=1024,
        tokens=TokenBuffer(np.array(active_tokens, dtype=np.int64)),
    )
    if chunk is not None:
        context.tokens.chunk(chunk)
    return context


def _to_ragged_inputs(
    contexts: list[list[TextContext]],
) -> tuple[list[int], list[int]]:
    flat_contexts = [ctx for batch in contexts for ctx in batch]
    tokens = np.concatenate(
        [context.tokens.active for context in flat_contexts]
    ).tolist()
    input_row_offsets = np.cumsum(
        [0] + [context.tokens.active_length for context in flat_contexts]
    ).tolist()
    return tokens, input_row_offsets


def _to_model_inputs(
    tokens: list[int], input_row_offsets: list[int]
) -> RaggedModelInputs:
    return RaggedModelInputs(
        tokens=_to_gpu(tokens),
        input_row_offsets=_to_gpu(input_row_offsets, dtype=np.uint32),
    )


def _to_cpu(arr: list[int] | list[list[int]], dtype: type = np.int64) -> Buffer:
    return Buffer.from_dlpack(np.array(arr, dtype=dtype))


def _to_gpu(arr: list[int] | list[list[int]], dtype: type = np.int64) -> Buffer:
    return _to_cpu(arr, dtype=dtype).to(Accelerator())


def _create_async_batch(
    batches: list[list[TextContext]],
) -> AsyncBatch[TextContext]:
    # generated_tokens is of shape (num_reqs,)
    generated_tokens: list[int] = []
    for batch in batches:
        for ctx in batch:
            generated_tokens.append(ctx.tokens.active[-1] + 1)
            ctx.update_with_future_token()
    return AsyncBatch(
        inputs=TextGenerationInputs(batches=batches),
        generated_tokens_device=_to_gpu(generated_tokens),
        generated_tokens_host=_to_cpu(generated_tokens),
        copy_event=MagicMock(),
    )


@pytest.fixture(scope="module")
def realize_future_token_processor() -> RealizeFutureTokenProcessor:
    device = Accelerator()
    return RealizeFutureTokenProcessor(
        session=InferenceSession(devices=[device]),
        devices=[DeviceRef.from_device(device)],
        num_speculative_tokens=0,
        enable_dp=False,
    )


def _scatter(
    realize_future_token_processor: RealizeFutureTokenProcessor,
    prev_batch: AsyncBatch[TextContext],
    curr_batch: list[list[TextContext]],
    model_inputs: RaggedModelInputs,
) -> tuple[list[int] | None, list[int]]:
    curr_inputs = TextGenerationInputs(batches=curr_batch)

    mappings = realize_future_token_processor._compute_mappings(
        prev_batch=prev_batch,
        inputs=curr_inputs,
    )
    if mappings is not None:
        prev_to_curr_map, _curr_to_prev_map = mappings
    else:
        prev_to_curr_map, _curr_to_prev_map = None, None
    realize_future_token_processor.realize_future_tokens(
        prev_batch=prev_batch,
        inputs=curr_inputs,
        model_inputs=model_inputs,
    )
    new_ragged_input_tokens = model_inputs.tokens
    return (
        prev_to_curr_map.to_numpy().tolist()
        if prev_to_curr_map is not None
        else None,
        new_ragged_input_tokens.to_numpy().tolist(),
    )


def test_basic(
    realize_future_token_processor: RealizeFutureTokenProcessor,
) -> None:
    req_a = _create_context([11, 12, 13])
    req_b = _create_context([21, 22, 23, 24])
    req_c = _create_context([31, 32])
    req_d = _create_context([41, 42])
    req_e = _create_context([51, 52])

    # replica1 has 2 reqs, replica2 has 0 reqs, replica3 has 1 reqs, replica4 has 0 reqs
    prev_batch = [[req_a, req_b], [], [req_c], []]
    prev_tokens, prev_input_row_offsets = _to_ragged_inputs(prev_batch)
    assert prev_tokens == [11, 12, 13, 21, 22, 23, 24, 31, 32]
    assert prev_input_row_offsets == [0, 3, 7, 9]

    async_batch = _create_async_batch(batches=prev_batch)
    generated_tokens = (
        async_batch.generated_tokens_host.to_numpy().flatten().tolist()
    )
    assert generated_tokens == [14, 25, 33]

    tests: list[
        tuple[list[list[TextContext]], list[int], list[int] | None, list[int]]
    ] = [
        (
            # test case where there is partial overlap between prev and curr batch
            [[], [req_b], [req_c, req_d], []],
            [FUTURE_TOKEN, FUTURE_TOKEN, 41, 42],
            [OOB_IDX, 0, 1],
            [25, 33, 41, 42],
        ),
        (
            # test case where reqs in curr batch are not in same order as prev batch
            [[req_e], [req_b, req_a], [req_d], []],
            [51, 52, FUTURE_TOKEN, FUTURE_TOKEN, 41, 42],
            [2, 1, OOB_IDX],
            [51, 52, 25, 14, 41, 42],
        ),
        (
            # test case where prev and curr batch have no overlapping reqs
            [[], [], [], [req_e, req_d]],
            [51, 52, 41, 42],
            None,  # all oob — _compute_scatter_future_tok_indices returns None
            [51, 52, 41, 42],  # no change
        ),
    ]
    for (
        curr_batch,
        expected_curr_ragged_inputs,
        expected_prev_to_curr_map,
        expected_new_ragged_inputs,
    ) in tests:
        curr_tokens, curr_input_row_offsets = _to_ragged_inputs(curr_batch)
        assert curr_tokens == expected_curr_ragged_inputs
        prev_to_curr_map, new_ragged_inputs = _scatter(
            realize_future_token_processor=realize_future_token_processor,
            prev_batch=async_batch,
            curr_batch=curr_batch,
            model_inputs=_to_model_inputs(curr_tokens, curr_input_row_offsets),
        )
        assert prev_to_curr_map == expected_prev_to_curr_map
        assert new_ragged_inputs == expected_new_ragged_inputs


def test_chunked_prefill(
    realize_future_token_processor: RealizeFutureTokenProcessor,
) -> None:
    req_a = _create_context([11, 12, 13])
    req_b = _create_context([21, 22, 23, 24, 25], chunk=3)
    assert req_b.tokens.active.tolist() == [21, 22, 23]
    req_c = _create_context([31, 32, 33, 34, 35, 36, 37, 38], chunk=4)
    assert req_c.tokens.active.tolist() == [31, 32, 33, 34]
    req_d = _create_context([41, 42, 43])
    req_e = _create_context([51, 52, 53])

    prev_batch = [req_a, req_b, req_c]
    prev_ragged_inputs, prev_input_row_offsets = _to_ragged_inputs([prev_batch])
    assert prev_ragged_inputs == [11, 12, 13, 21, 22, 23, 31, 32, 33, 34]
    assert prev_input_row_offsets == [0, 3, 6, 10]

    async_batch = _create_async_batch(batches=[prev_batch])
    generated_tokens = (
        async_batch.generated_tokens_host.to_numpy().flatten().tolist()
    )
    assert generated_tokens == [14, 24, 35]

    # Make sure that we correctly handle chunked requests
    # Note that the sampled tokens for chunked requests are dropped/ignored
    curr_batch = [req_d, req_b, req_e, req_a, req_c]
    curr_ragged_inputs, curr_input_row_offsets = _to_ragged_inputs([curr_batch])
    # fmt: off
    assert curr_ragged_inputs == [41, 42, 43, 24, 25, 51, 52, 53, FUTURE_TOKEN, 35, 36, 37, 38]
    assert curr_input_row_offsets == [0, 3, 5, 8, 9, 13]
    prev_to_curr_map, new_ragged_inputs = _scatter(
        realize_future_token_processor=realize_future_token_processor,
        prev_batch=async_batch,
        curr_batch=[curr_batch],
        model_inputs=_to_model_inputs(curr_ragged_inputs, curr_input_row_offsets),
    )
    assert prev_to_curr_map == [3, OOB_IDX, OOB_IDX]
    assert new_ragged_inputs == [41, 42, 43, 24, 25, 51, 52, 53, 14, 35, 36, 37, 38]
    # fmt: on


def test_no_overlap_returns_none_and_skips_scatter(
    realize_future_token_processor: RealizeFutureTokenProcessor,
) -> None:
    """When no requests overlap between batches, _compute_scatter_future_tok_indices
    returns None and realize_future_tokens returns the original tokens unchanged.

    This is the steady-state for prefill_only workers, where each batch contains
    new requests that never appeared in the previous batch.
    """
    req_a = _create_context([11, 12, 13])
    req_b = _create_context([21, 22, 23])
    req_c = _create_context([31, 32, 33])

    prev_batch = _create_async_batch(batches=[[req_a, req_b]])

    # curr_batch has no requests from prev_batch
    curr_ragged_inputs, curr_input_row_offsets = _to_ragged_inputs([[req_c]])
    assert curr_input_row_offsets == [0, 3]

    curr_inputs = TextGenerationInputs(batches=[[req_c]])

    # _compute_scatter_future_tok_indices returns None — no work to do.
    mappings = realize_future_token_processor._compute_mappings(
        prev_batch=prev_batch,
        inputs=curr_inputs,
    )
    assert mappings is None

    # realize_future_tokens returns original tokens unchanged.
    model_inputs = _to_model_inputs(curr_ragged_inputs, curr_input_row_offsets)
    realize_future_token_processor.realize_future_tokens(
        prev_batch=prev_batch,
        inputs=curr_inputs,
        model_inputs=model_inputs,
    )
    assert model_inputs.tokens.to_numpy().tolist() == curr_ragged_inputs
