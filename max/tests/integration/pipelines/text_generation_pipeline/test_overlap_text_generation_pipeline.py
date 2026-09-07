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


from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from llguidance import LLMatcher
from max.pipelines.context import (
    StructuredOutputRegionDelimiters,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.context import FUTURE_TOKEN
from max.pipelines.kv_cache.paged_kv_cache.block_manager import (
    _compute_seq_len,
)
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextGenerationPipeline,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _MAX_GRAPH_CAPTURE_BATCH_SIZE,
    _OOB_IDX,
    MAGIC_DRAFT_TOKEN_ID,
    AsyncBatch,
    _host_mirror_realized_drafts,
)
from max.pipelines.lib.pipeline_variants.utils import (
    StructuredOutputHelper,
    _count_token_subsequence,
)
from max.pipelines.lib.registry import get_pipeline_for_task
from max.pipelines.modeling.types import (
    PipelineTask,
    RequestID,
    TextGenerationInputs,
)
from max.support.math import ceildiv


@pytest.mark.parametrize(
    "content,special_tags,expected",
    [
        ([5, 1, 2, 5, 3, 5], [5], 3),  # single-token marker (Kimi case)
        ([1, 2, 3], [5], 0),
        ([], [5], 0),
        ([1, 2, 3], [], 0),  # empty tags never match
        ([9, 8, 1, 9, 8], [9, 8], 2),  # multi-token marker
        ([7, 7, 7, 7], [7, 7], 2),  # no double-counting overlaps
    ],
)
def test_count_token_subsequence(
    content: list[int], special_tags: list[int], expected: int
) -> None:
    assert _count_token_subsequence(content, special_tags) == expected


class _FakeMatcher:
    def __init__(self, accepting: bool, stopped: bool) -> None:
        self._a, self._s = accepting, stopped

    def is_accepting(self) -> bool:
        return self._a

    def is_stopped(self) -> bool:
        return self._s


def _fake_ctx(matcher: object, generated: list[int]) -> MagicMock:
    ctx = MagicMock()
    ctx.matcher = matcher
    ctx.grammar_enforced = True
    ctx.tools_forced = True
    ctx.snapshot_grammar_state.return_value.in_thinking_region = False
    ctx.tokens.generated = generated
    return ctx


def test_rejection_diagnostics_reports_mid_tool_call_state() -> None:
    """Diagnostics surface open-section + non-accepting (mid-tool-call) state.

    Logs raw token IDs only (no decoded text); the desyncing batch is
    reconstructable offline from ``committed_token_ids``.
    """
    helper = StructuredOutputHelper(
        enabled=True,
        tool_call_region_delimiters=StructuredOutputRegionDelimiters(
            start_token_ids=[256], end_token_ids=[257]
        ),
    )
    # One section-begin (256), no section-end -> open_sections == 1.
    ctx = _fake_ctx(_FakeMatcher(accepting=False, stopped=False), [256, 10, 11])
    diag = helper._rejection_diagnostics(
        ctx, committed_tokens=[1, 2, 27], committed_idx=2
    )
    assert "matcher_accepting=False" in diag
    assert "open_sections=1" in diag
    assert "reject_idx=2/3" in diag
    assert "committed_token_ids=[1, 2, 27]" in diag


def test_rejection_diagnostics_never_raises() -> None:
    """A diagnostic failure degrades to a placeholder, never crashes."""
    helper = StructuredOutputHelper(enabled=True)
    bad_matcher = MagicMock()
    bad_matcher.is_accepting.side_effect = RuntimeError("boom")
    ctx = _fake_ctx(bad_matcher, [1, 2, 3])
    diag = helper._rejection_diagnostics(
        ctx, committed_tokens=[5], committed_idx=0
    )
    assert diag.startswith("<diagnostics unavailable")


def test_throws_if_enable_log_probs() -> None:
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    request_id = RequestID()
    ctx = TextContext(
        request_id=request_id,
        max_length=1000,
        tokens=TokenBuffer(np.array([42, 67, 21])),
        log_probabilities=1,
    )
    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[ctx]],
    )
    with pytest.raises(
        ValueError,
        match=r"Log probabilities are not supported with overlap pipeline",
    ):
        pipeline.execute(inputs)


def test_forward_launched_before_sampling_processor_build() -> None:
    """The non-spec decode path launches the forward before building the sampler.

    ``_run_forward`` enqueues the decode kernels asynchronously, so building the
    sampling processor afterwards overlaps its host-side work (SamplerInputs
    gather + small H2D copies) with the GPU forward instead of being exposed
    host time ahead of the launch. This pins that ordering (a regression guard
    for the async-overlap optimization) and verifies the two calls remain
    independent: each runs exactly once and receives the expected argument.
    """
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    pipeline._spec_decode_state = None
    pipeline._prev_batch = None
    pipeline._disable_overlap = False
    pipeline._kv_manager = MagicMock()

    call_order: list[str] = []

    mock_model_outputs = MagicMock(name="model_outputs")
    mock_sampling_processor = MagicMock(name="sampling_processor")
    mock_curr_batch = MagicMock(name="curr_batch")

    def _run_forward(inputs: object) -> object:
        call_order.append("run_forward")
        return mock_model_outputs

    def _create_sampling_processor(
        flat_batch: object,
    ) -> tuple[object, object]:
        call_order.append("create_sampling_processor")
        return mock_sampling_processor, None

    def _sample_logits(
        inputs: object, model_outputs: object, sampling_processor: object
    ) -> object:
        call_order.append("sample_logits")
        return mock_curr_batch

    pipeline._run_forward = MagicMock(side_effect=_run_forward)  # type: ignore[method-assign]
    pipeline._create_sampling_processor = MagicMock(  # type: ignore[method-assign]
        side_effect=_create_sampling_processor
    )
    pipeline._sample_logits = MagicMock(side_effect=_sample_logits)  # type: ignore[method-assign]

    mock_ctx = MagicMock(name="ctx")
    inputs = MagicMock(name="inputs")
    inputs.enable_log_probs = False
    inputs.flat_batch = [mock_ctx]
    inputs.batches = [[mock_ctx]]

    pipeline.execute(cast(TextGenerationInputs[TextContext], inputs))

    # Forward must be enqueued before the sampling processor is built so that
    # the sampler's host-side construction overlaps the GPU forward.
    assert call_order == [
        "run_forward",
        "create_sampling_processor",
        "sample_logits",
    ]
    pipeline._run_forward.assert_called_once_with(inputs)
    pipeline._create_sampling_processor.assert_called_once_with(
        inputs.flat_batch
    )
    # Reorder must not clear the deferred (overlapped) current batch.
    assert pipeline._prev_batch is mock_curr_batch


@pytest.mark.parametrize(
    ("config_max_batch_size", "expected_capture_batch_size"),
    [
        (4096, _MAX_GRAPH_CAPTURE_BATCH_SIZE),
        (32, 32),
    ],
    ids=["capped", "uncapped"],
)
def test_warmup_graph_capture_batch_size(
    config_max_batch_size: int,
    expected_capture_batch_size: int,
) -> None:
    """warmup_graph_capture should cap the batch size at _MAX_GRAPH_CAPTURE_BATCH_SIZE."""
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.max_seq_len = 2048
    pipeline._pipeline_model = mock_model
    pipeline._pipeline_config = MagicMock()
    pipeline._max_batch_size = config_max_batch_size
    pipeline._kv_manager = MagicMock()
    mock_kv_params = MagicMock()
    mock_kv_params.page_size = 128
    mock_kv_params.num_draft_tokens = 0
    pipeline._kv_manager.params = mock_kv_params
    pipeline._kv_manager.cache_params.return_value = mock_kv_params
    pipeline._kv_manager._total_num_pages = 100
    pipeline._kv_manager.effective_max_seq_length = 100 * 128
    pipeline._spec_decode_state = None
    pipeline._kv_manager.num_caches = 1

    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".ServeGraphCaptureRunner"
    ) as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.warmup_pre_ready = MagicMock()

        pipeline.warmup_graph_capture()

        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["model"] is mock_model.model
        assert call_kwargs["kv_params"] is mock_kv_params
        assert callable(call_kwargs["warmup_model_inputs"])
        assert (
            call_kwargs["max_cache_length_upper_bound"]
            == mock_model.max_seq_len
        )
        assert call_kwargs["max_batch_size"] == expected_capture_batch_size
        assert (
            pipeline._max_graph_capture_batch_size
            == expected_capture_batch_size
        )


def _make_effective_cache_length_pipeline(
    *,
    max_seq_len: int,
    num_draft_tokens: int,
    num_draft_tokens_per_step: int,
    total_num_pages: int,
    page_size: int = 128,
) -> OverlapTextGenerationPipeline[TextContext]:
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    mock_model = MagicMock()
    mock_model.max_seq_len = max_seq_len
    pipeline._pipeline_model = mock_model
    pipeline._kv_manager = MagicMock()
    mock_kv_params = MagicMock()
    mock_kv_params.page_size = page_size
    mock_kv_params.num_draft_tokens = num_draft_tokens
    mock_kv_params.num_draft_tokens_per_step = num_draft_tokens_per_step
    pipeline._kv_manager.params = mock_kv_params
    pipeline._kv_manager._total_num_pages = total_num_pages
    pipeline._kv_manager.effective_max_seq_length = total_num_pages * page_size
    return pipeline


@pytest.mark.parametrize(
    ("num_draft_tokens", "num_draft_tokens_per_step", "expected_slack"),
    [
        (0, 0, 0),  # speculative decoding disabled: strict no-op
        (3, 1, 10),  # eagle/mtp autoregressive drafts: 3*3 + 0 + 1
        (4, 4, 14),  # dflash block drafts: 3*4 + 1 + 1
    ],
    ids=["disabled", "eagle", "dflash"],
)
def test_effective_max_cache_length_spec_slack(
    num_draft_tokens: int,
    num_draft_tokens_per_step: int,
    expected_slack: int,
) -> None:
    """The capture bound folds in the worst-case speculative-decode slack."""
    max_seq_len = 2048
    pipeline = _make_effective_cache_length_pipeline(
        max_seq_len=max_seq_len,
        num_draft_tokens=num_draft_tokens,
        num_draft_tokens_per_step=num_draft_tokens_per_step,
        # Pool far larger than the bound so the capacity cap does not engage.
        total_num_pages=10_000,
    )
    assert pipeline._effective_max_cache_length == max_seq_len + expected_slack


@pytest.mark.parametrize(
    ("num_draft_tokens", "num_draft_tokens_per_step"),
    [
        (3, 1),  # eagle/mtp
        (4, 4),  # dflash
    ],
    ids=["eagle", "dflash"],
)
def test_effective_max_cache_length_covers_compute_seq_len(
    num_draft_tokens: int,
    num_draft_tokens_per_step: int,
) -> None:
    """The capture bound must cover the requirement ``runtime_inputs`` enforces.

    ``PagedKVCacheManager.runtime_inputs`` rejects any batch whose
    ``_compute_seq_len`` exceeds the captured ``max_cache_length``. Pin the
    bound to ``_compute_seq_len`` directly so a future change to its
    speculative-slack accounting is caught here rather than crashing
    capture-replay at the context boundary (GEX-3748 / MAX-615).
    """
    max_seq_len = 2048
    pipeline = _make_effective_cache_length_pipeline(
        max_seq_len=max_seq_len,
        num_draft_tokens=num_draft_tokens,
        num_draft_tokens_per_step=num_draft_tokens_per_step,
        total_num_pages=10_000,
    )

    # Worst-case boundary request: committed tokens fill the context window and
    # carry the FUTURE_TOKEN placeholder, with the previous overlap batch's
    # drafts all counted as accepted.
    tokens = TokenBuffer(np.zeros(max_seq_len + 1, dtype=np.int64))
    tokens.skip_processing(max_seq_len)
    boundary_ctx = SimpleNamespace(
        tokens=tokens,
        spec_decoding_state=SimpleNamespace(
            maybe_accepted_draft_tokens=[0] * num_draft_tokens
        ),
    )
    required = _compute_seq_len(
        cast(TextContext, boundary_ctx),
        num_draft_tokens=num_draft_tokens,
        num_draft_tokens_per_step=num_draft_tokens_per_step,
    )
    assert pipeline._effective_max_cache_length >= required


def test_effective_max_cache_length_capped_to_pool_capacity() -> None:
    """The bound never exceeds the pages the pool actually allocated."""
    pipeline = _make_effective_cache_length_pipeline(
        max_seq_len=2048,
        num_draft_tokens=3,
        num_draft_tokens_per_step=1,
        # One page of capacity: the bound clamps to page_size regardless of
        # max_seq_len + slack.
        total_num_pages=1,
        page_size=128,
    )
    assert pipeline._effective_max_cache_length == 128


def _make_pipeline_config_mock(
    *, enable_overlap: bool, pipeline_role: str
) -> MagicMock:
    config = MagicMock()
    config.runtime.enable_overlap_scheduler = enable_overlap
    config.runtime.pipeline_role = pipeline_role
    config.speculative = None
    return config


def test_prefill_only_gets_overlap_pipeline() -> None:
    """Prefill-only DI workers use the overlap pipeline when overlap is enabled."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="prefill_only"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_decode_only_gets_overlap_pipeline() -> None:
    """Decode-only workers use the overlap pipeline when overlap is enabled."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="decode_only"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_prefill_and_decode_gets_overlap_pipeline() -> None:
    """Non-DI workers continue to use the overlap pipeline as before."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="prefill_and_decode"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_async_batch_sync_with_single_step_tokens() -> None:
    """AsyncBatch.sync_and_process_outputs handles single-step token shapes."""

    batch_size = 3

    # Create mock contexts
    contexts = []
    for i in range(batch_size):
        ctx = TextContext(
            request_id=RequestID(f"req_{i}"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        contexts.append(ctx)

    # Create mock inputs
    mock_inputs = MagicMock()
    mock_inputs.flat_batch = contexts

    # Create single-step tokens buffer [batch_size]
    generated_tokens = np.arange(batch_size, dtype=np.int64)

    # Create mock host buffer
    mock_host_buffer = MagicMock()
    mock_host_buffer.to_numpy.return_value = generated_tokens

    # Create mock event
    mock_event = MagicMock()

    # Create AsyncBatch
    async_batch: AsyncBatch[TextContext] = AsyncBatch(
        inputs=mock_inputs,
        generated_tokens_device=MagicMock(),
        generated_tokens_host=mock_host_buffer,
        copy_event=mock_event,
    )

    # Patch update_context_and_prepare_responses to verify it's called correctly
    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".update_context_and_prepare_responses"
    ) as mock_update:
        mock_update.return_value = {}

        async_batch.sync_and_process_outputs()

        # Verify the function was called with correct arguments
        mock_update.assert_called_once()
        call_args = mock_update.call_args

        # Check positional args - tokens should be reshaped to [batch_size, 1]
        tokens_arg = call_args[0][0]
        assert tokens_arg.shape == (batch_size, 1)

        contexts_arg = call_args[0][1]
        assert contexts_arg == contexts

        # Check keyword args (overlap path always uses single-step [batch, 1] tokens)
        assert call_args[1]["overwrite_future"] is True


class TestUpdateWithFutureTokenStructuredOutput:
    """Tests for update_with_future_token behavior with structured output."""

    def test_skips_fsm_when_matcher_present(self) -> None:
        """update_with_future_token should NOT advance FSM when matcher is present.

        For structured output, the FSM cannot accept placeholder tokens (-999).
        The FSM will be advanced later with the real token.
        """

        ctx = TextContext(
            request_id=RequestID("test"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        # Set up a mock matcher
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        initial_length = len(ctx.tokens.all)

        # Call update_with_future_token
        ctx.update_with_future_token()

        # Token buffer should have FUTURE_TOKEN appended
        assert len(ctx.tokens.all) == initial_length + 1
        assert ctx.tokens.all[-1] == FUTURE_TOKEN

        # FSM (matcher.try_consume_tokens) should NOT have been called
        mock_matcher.try_consume_tokens.assert_not_called()

        # On the other hand, update_with_future_token should advance FSM when no matcher present.
        # For non-structured output, the standard update() path is used
        ctx = TextContext(
            request_id=RequestID("test"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        # No matcher set (ctx.matcher is None)
        assert ctx.matcher is None

        initial_length = len(ctx.tokens.all)

        # Call update_with_future_token
        ctx.update_with_future_token()

        # Token buffer should have FUTURE_TOKEN appended
        assert len(ctx.tokens.all) == initial_length + 1
        assert ctx.tokens.all[-1] == FUTURE_TOKEN


class TestSyncAndProcessOutputsStructuredOutput:
    """Tests for sync_and_process_outputs with structured output."""

    def test_advances_fsm_with_real_token(self) -> None:
        """sync_and_process_outputs should advance FSM with real token for structured output.

        When syncing the previous batch, the FSM should be advanced with the
        actual sampled token (not the placeholder).
        """
        batch_size = 2
        real_tokens = np.array([100, 200], dtype=np.int64)

        # Create contexts with mock matchers. ``advance_fsm`` only calls
        # ``try_consume_tokens`` while ``grammar_enforced=True``, so flip
        # it on for this test path.
        contexts = []
        for i in range(batch_size):
            ctx = TextContext(
                request_id=RequestID(f"req_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([42, 67, 21])),
            )
            ctx.grammar_enforced = True
            mock_matcher = MagicMock()
            mock_matcher.try_consume_tokens = MagicMock(return_value=1)
            ctx._matcher = mock_matcher
            # A real committed token bumps generated_length > 0 -- the only
            # state in which the FSM should advance.
            ctx.update_with_future_token()
            assert ctx.tokens.generated_length == 1
            contexts.append(ctx)

        # Create mock inputs
        mock_inputs = MagicMock()
        mock_inputs.flat_batch = contexts

        # Create mock host buffer with real tokens
        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_tokens
        mock_host_buffer.shape = real_tokens.shape

        # Create mock event
        mock_event = MagicMock()

        # Create mock structured output helper (required for FSM advancement)
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True

        # Create AsyncBatch
        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=mock_event,
            structured_output=mock_structured_output,
        )

        # Patch update_context_and_prepare_responses
        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs()

            # Verify FSM was advanced with real tokens for each context
            for i, ctx in enumerate(contexts):
                assert ctx._matcher is not None
                ctx._matcher.try_consume_tokens.assert_called_once_with(
                    [int(real_tokens[i])]
                )

    def test_skips_fsm_for_intermediate_chunk_even_when_not_actively_chunked(
        self,
    ) -> None:
        """Regression: do not advance the FSM for an intermediate chunked-prefill
        batch, even if ``actively_chunked`` reads False at sync time.

        An intermediate chunk-prefill step never commits a real generated token
        (``update_with_future_token`` early-returns via ``advance_chunk()``
        without appending a placeholder), so ``generated_length`` stays 0. By
        the time the previous batch is synced, the scheduler may have rebuilt
        the current batch and toggled ``actively_chunked`` back to False (e.g.
        when the current batch is this request's final, short chunk). The old
        guard read that mutated flag and wrongly fed the previous
        (intermediate-chunk) batch's prefill-artifact sampled token into the
        matcher, advancing the grammar FSM one token too far — which dropped the
        opening ``{`` of a JSON-schema answer and led to a structured-output
        runaway under chunked prefill. The correct guard is
        ``generated_length``.
        """
        real_token = np.array([100], dtype=np.int64)

        ctx = TextContext(
            request_id=RequestID("chunked_req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21, 11, 9])),
        )
        ctx.grammar_enforced = True
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        # Simulate the previous batch having been an intermediate chunked
        # prefill step: no real token committed, so generated_length == 0.
        assert ctx.tokens.generated_length == 0
        # Reproduce the trap: actively_chunked reads False at sync time even
        # though no real token was committed.
        assert not ctx.tokens.actively_chunked

        mock_inputs = MagicMock()
        mock_inputs.flat_batch = [ctx]
        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_token
        mock_host_buffer.shape = real_token.shape
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True

        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=MagicMock(),
            structured_output=mock_structured_output,
        )

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}
            async_batch.sync_and_process_outputs()

            # The FSM must NOT be advanced: no real token was committed.
            mock_matcher.try_consume_tokens.assert_not_called()

    def test_updates_bitmask_for_continuing_requests(self) -> None:
        """sync_and_process_outputs should update bitmask for requests continuing to next batch."""
        real_token = np.array([100], dtype=np.int64)

        # Create context with mock matcher
        ctx = TextContext(
            request_id=RequestID("continuing_req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        # Previous batch inputs
        mock_inputs = MagicMock()
        mock_inputs.flat_batch = [ctx]

        # Create mock host buffer
        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_token
        mock_host_buffer.shape = real_token.shape

        # Create mock StructuredOutputHelper
        mock_structured_output = MagicMock()

        # Create AsyncBatch for previous batch
        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=MagicMock(),
            structured_output=mock_structured_output,
        )

        # Current batch also contains this request (continuing)
        curr_flat_batch = [ctx]
        bitmask = np.zeros((1, 10), dtype=np.int32)
        mock_sampling_processor = MagicMock()

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs(
                curr_flat_batch=curr_flat_batch,
                bitmask=bitmask,
                sampling_processor=mock_sampling_processor,
            )

            # Verify bitmask was filled for continuing request via StructuredOutputHelper
            mock_structured_output.fill_bitmask.assert_called_once_with(
                ctx,
                bitmask,
                0,  # curr_idx = 0
            )

            # Verify bitmask was transferred to device
            mock_sampling_processor.update_bitmask.assert_called_once_with(
                bitmask
            )

    def test_skips_fsm_for_chunked_prefill(self) -> None:
        """sync_and_process_outputs should skip FSM advancement for actively chunked contexts."""
        real_token = np.array([100], dtype=np.int64)

        # Create context with mock matcher that is actively chunked
        ctx = TextContext(
            request_id=RequestID("chunked_req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        ctx.tokens._actively_chunked = True  # Simulate chunked prefill
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        mock_inputs = MagicMock()
        mock_inputs.flat_batch = [ctx]

        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_token
        mock_host_buffer.shape = real_token.shape

        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=MagicMock(),
        )

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs()

            # FSM should NOT be advanced for actively chunked context
            mock_matcher.try_consume_tokens.assert_not_called()


class TestAdvanceFsmAndComputeBitmasks:
    """Tests for StructuredOutputHelper.advance_fsm_and_compute_bitmasks."""

    def _make_helper(self, vocab_size: int = 64) -> StructuredOutputHelper:
        return StructuredOutputHelper(enabled=True, vocab_size=vocab_size)

    def _make_context_with_matcher(
        self, always_accept: bool = True
    ) -> tuple[TextContext, MagicMock]:
        ctx = TextContext(
            request_id=RequestID("req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        # advance_fsm_and_compute_bitmasks only advances the matcher while
        # ``grammar_enforced=True``. Default the test context to enforced so
        # these tests exercise the matcher-advance path.
        ctx.grammar_enforced = True
        # Mark as a continuing (non-initial-prompt) context so Part 2 writes
        # its row; is_initial_prompt=True causes Part 2 to skip the row.
        ctx._is_initial_prompt = False
        mock_matcher = MagicMock(spec=LLMatcher)
        ret = 1 if always_accept else 0
        mock_matcher.try_consume_tokens = MagicMock(return_value=ret)
        # Part 2 speculates on a deep copy of the matcher (never the real one),
        # so the rollback-across-rule-boundary desync cannot occur. Mirror the
        # accept behavior on the copy; tests reach it via
        # ``mock_matcher.deep_copy.return_value``.
        mock_matcher.deep_copy.return_value = MagicMock(spec=LLMatcher)
        mock_matcher.deep_copy.return_value.try_consume_tokens = MagicMock(
            return_value=ret
        )
        ctx._matcher = mock_matcher
        return ctx, mock_matcher

    def test_unconstrained_context_bitmask_stays_all_minus_one(self) -> None:
        """Context with no matcher keeps all bitmask slots at -1 (unconstrained)."""
        helper = self._make_helper()
        ctx = TextContext(
            request_id=RequestID("req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        assert ctx.matcher is None
        # Mark as a continuing context so Part 2 resets the row to -1;
        # is_initial_prompt=True causes Part 2 to skip the row entirely.
        ctx._is_initial_prompt = False

        bitmask_out = np.zeros((1, 3, 2), dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask") as mock_fill:
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 2), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        mock_fill.assert_not_called()
        assert np.all(bitmask_out == -1)

    def test_part1_advances_through_accepted_drafts_and_bonus(self) -> None:
        """Part 1 advances FSM through accepted draft tokens then bonus (no rollback)."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.array([[7, 8]], dtype=np.int64),
                num_accepted=np.array([2], dtype=np.int32),
                bonus_tokens=np.array([9], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # Part 1 consumes one token at a time (accepted drafts then bonus) so
        # the enforcement state machine can flip mid-sequence on special
        # tokens. Followed by Part 2 speculatively consuming next_draft_tokens.
        all_calls = mock_matcher.try_consume_tokens.call_args_list
        assert all_calls[:3] == [call([7]), call([8]), call([9])]

    def test_part1_skips_accepted_drafts_when_zero_accepted(self) -> None:
        """When n_accepted=0, only the bonus token is consumed in Part 1."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        bitmask_out = np.full((1, 2, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.array([[7, 8]], dtype=np.int64),
                num_accepted=np.array([0], dtype=np.int32),
                bonus_tokens=np.array([9], dtype=np.int64),
                next_draft_tokens=np.array([[10]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        all_calls = mock_matcher.try_consume_tokens.call_args_list
        # No call for either accepted draft token; bonus is consumed first.
        assert call([7]) not in all_calls
        assert call([8]) not in all_calls
        assert all_calls[0] == call([9])

    def test_part2_fills_position_0_bitmask_after_part1(self) -> None:
        """Position 0 of bitmask is filled with current FSM state (after Part 1 advance)."""
        helper = self._make_helper(vocab_size=64)
        ctx, _ = self._make_context_with_matcher()
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask") as mock_fill:
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # fill_next_token_bitmask called at least once (position 0)
        assert mock_fill.call_count >= 1
        # First call is for position 0
        first_kwargs = mock_fill.call_args_list[0][1]
        assert first_kwargs["index"] == 0

    def test_part2_speculatively_advances_on_deep_copy_not_real_matcher(
        self,
    ) -> None:
        """Part 2 walks next draft tokens on a deep copy; the real matcher is
        never advanced or rolled back.

        ``LLMatcher.rollback`` is not a perfect inverse across a grammar
        rule/repetition boundary, so the speculative walk must not mutate the
        real matcher. ``try_consume_tokens`` and ``rollback`` run on a deep copy.
        """
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher(always_accept=True)
        scratch = mock_matcher.deep_copy.return_value
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # Real matcher: Part 1 consumes only the bonus token; never rolled back.
        assert mock_matcher.try_consume_tokens.call_args_list == [call([5])]
        mock_matcher.rollback.assert_not_called()
        # Part 2 speculates on the deep copy: both next draft tokens consumed.
        mock_matcher.deep_copy.assert_called_once()
        assert scratch.try_consume_tokens.call_args_list == [
            call([10]),
            call([11]),
        ]
        scratch.rollback.assert_not_called()

    def test_part2_stops_on_deep_copy_when_fsm_rejects_first_draft(
        self,
    ) -> None:
        """When the FSM rejects the first next draft token, the speculative
        walk stops and the real matcher is untouched in Part 2."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        scratch = mock_matcher.deep_copy.return_value
        # Bonus token accepted (Part 1, real matcher); first next_draft
        # rejected (Part 2, deep copy).
        mock_matcher.try_consume_tokens.side_effect = [1]
        scratch.try_consume_tokens.side_effect = [0]
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # Walk broke at the first rejected draft on the copy.
        assert scratch.try_consume_tokens.call_args_list == [call([10])]
        mock_matcher.rollback.assert_not_called()
        # Rollback() should not be used to undo token consumption.
        scratch.rollback.assert_not_called()

    def test_part2_degrades_unattributable_output_row(self) -> None:
        """A row absent from the producing batch is degraded, not raised on."""
        helper = self._make_helper()
        ctx_prod, _ = self._make_context_with_matcher()
        # A consumer-only row whose request is NOT in the producing batch.
        ctx_extra = TextContext(
            request_id=RequestID("extra"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        ctx_extra._is_initial_prompt = False

        # Poison every row: a reached row is reset to -1 at the top of its loop
        # iteration, so a row still holding 9 afterwards was never reached.
        bitmask_out = np.full((2, 3, 2), 9, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx_prod],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
                output_context_batch=[ctx_extra, ctx_prod],
            )

        # No raise; both rows reached (neither retains the sentinel).
        assert (bitmask_out == -1).all()

    def test_part2_degrades_preempted_initial_prompt_row(self) -> None:
        """A row reset (preempted) after enqueue is degraded, not raised on."""
        helper = self._make_helper()
        ctx, _ = self._make_context_with_matcher()
        # Same request present in both batches, but reset to an initial prompt.
        ctx._is_initial_prompt = True

        bitmask_out = np.full((1, 3, 2), 9, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
                output_context_batch=[ctx],
            )

        # No raise; the preempted row is degraded to all-valid (-1).
        assert (bitmask_out == -1).all()

    def test_part2_reordered_rows_use_producer_drafts_by_request(self) -> None:
        """Part 2 writes in CONSUMER row order but reads each row's drafts from
        the PRODUCER slot via ``rid_to_src[request_id]``. With a non-identity
        permutation (producer [a, b] -> consumer [b, a]) each consumer row must
        be filled from its OWN request's ``next_draft_tokens`` -- guards against
        a ``src``/``out_idx`` swap, which identity-order tests cannot catch."""
        helper = self._make_helper()

        def _ctx(rid: str) -> TextContext:
            c = TextContext(
                request_id=RequestID(rid),
                max_length=1000,
                tokens=TokenBuffer(np.array([1, 2, 3])),
            )
            c.grammar_enforced = True
            c._is_initial_prompt = False
            m = MagicMock()
            # Accept committed tokens in Part 1 so the matcher-advance path
            # doesn't trip the rejection branch.
            m.try_consume_tokens = MagicMock(return_value=1)
            c._matcher = m
            return c

        ctx_a, ctx_b = _ctx("a"), _ctx("b")
        # Producer order [a, b]; a's next drafts = [10, 11], b's = [20, 21].
        next_draft_tokens = np.array([[10, 11], [20, 21]], dtype=np.int64)
        bitmask_out = np.full((2, 3, 2), -1, dtype=np.int32)

        # Capture (request_id, drafts, which consumer row) per fill.
        calls: list[tuple[str, list[int], bool]] = []

        def _spy(ctx, drafts, bitmask_window) -> None:  # noqa: ANN001
            calls.append(
                (
                    str(ctx.request_id),
                    list(np.asarray(drafts)),
                    np.shares_memory(bitmask_window, bitmask_out[0]),
                )
            )

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            with patch.object(
                helper, "_speculatively_fill_bitmask_window", side_effect=_spy
            ):
                helper.advance_fsm_and_compute_bitmasks(
                    context_batch=[ctx_a, ctx_b],
                    accepted_draft_tokens=np.zeros((2, 0), dtype=np.int64),
                    num_accepted=np.zeros(2, dtype=np.int32),
                    bonus_tokens=np.array([5, 6], dtype=np.int64),
                    next_draft_tokens=next_draft_tokens,
                    bitmask_out=bitmask_out,
                    # Consumer reorders to [b, a].
                    output_context_batch=[ctx_b, ctx_a],
                )

        # Consumer row 0 is b -> must use b's producer drafts [20, 21];
        # consumer row 1 is a -> must use a's producer drafts [10, 11].
        assert calls == [
            ("b", [20, 21], True),  # row 0 (shares memory with bitmask_out[0])
            ("a", [10, 11], False),  # row 1
        ]


class TestBuildBitmaskCallback:
    """Tests for OverlapTextGenerationPipeline._build_bitmask_callback."""

    def test_callback_calls_advance_fsm_and_compute_bitmasks(self) -> None:
        """The returned callback delegates to advance_fsm_and_compute_bitmasks."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        pipeline._structured_output = mock_so

        ctx = TextContext(
            request_id=RequestID("r"),
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        bonus_np = np.array([5], dtype=np.int64)
        num_acc_np = np.zeros(1, dtype=np.int64)
        draft_np = np.zeros((1, 2), dtype=np.int64)
        next_draft_np = np.zeros((1, 2), dtype=np.int64)
        # Packed int32 bitmask the callback writes straight into; the GPU
        # acceptance sampler unpacks and applies it, so there is no bool target.
        bitmask_np = np.full((1, 3, 2), -1, dtype=np.int32)

        callback = pipeline._build_bitmask_callback(
            context_batch=[ctx],
            output_context_batch=[ctx],
            bonus_tokens_np=bonus_np,
            num_accepted_np=num_acc_np,
            accepted_draft_tokens_np=draft_np,
            next_draft_tokens_np=next_draft_np,
            overlap_pinned_np=bitmask_np,
        )

        callback()

        mock_so.advance_fsm_and_compute_bitmasks.assert_called_once()
        kwargs = mock_so.advance_fsm_and_compute_bitmasks.call_args.kwargs
        assert kwargs["context_batch"] == [ctx]
        assert kwargs["output_context_batch"] == [ctx]
        assert kwargs["accepted_draft_tokens"] is draft_np
        assert kwargs["num_accepted"] is num_acc_np
        assert kwargs["bonus_tokens"] is bonus_np
        assert kwargs["next_draft_tokens"] is next_draft_np
        assert kwargs["bitmask_out"] is bitmask_np
        # The builder captures the snapshots itself, one per producing row.
        assert len(kwargs["committed_span_snapshots"]) == 1

    def test_callback_logs_error_on_exception(self) -> None:
        """Callback catches exceptions and logs them instead of propagating."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        mock_so.advance_fsm_and_compute_bitmasks.side_effect = RuntimeError(
            "boom"
        )
        pipeline._structured_output = mock_so

        callback = pipeline._build_bitmask_callback(
            context_batch=[],
            output_context_batch=[],
            bonus_tokens_np=np.array([], dtype=np.int64),
            num_accepted_np=np.array([], dtype=np.int64),
            accepted_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            next_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            overlap_pinned_np=np.zeros((0, 0, 0), dtype=np.int32),
        )

        # Must not raise — exceptions are caught and logged
        callback()

    def test_callback_exception_resets_whole_rectangle(self) -> None:
        """On exception, the fallback resets the entire owned rectangle to -1.

        The callback is the sole writer of the ``[:curr_batch_size]`` rectangle
        (the synchronous new-admission fill was consolidated into the callback),
        so the blanket reset races no main-thread write. Every row -- continuing
        or otherwise -- falls back to all-valid (-1) so generation still makes
        forward progress and the grammar re-converges next iter.
        """
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        mock_so.advance_fsm_and_compute_bitmasks.side_effect = RuntimeError(
            "boom"
        )
        pipeline._structured_output = mock_so

        ctx_a = TextContext(
            request_id=RequestID("a"),
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        ctx_a._is_initial_prompt = False
        ctx_b = TextContext(
            request_id=RequestID("b"),
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        ctx_b._is_initial_prompt = False

        # Both rows start with a non-(-1) sentinel; the fallback must reset all.
        overlap_pinned_np = np.full((2, 3, 2), 9, dtype=np.int32)

        callback = pipeline._build_bitmask_callback(
            context_batch=[ctx_a, ctx_b],
            output_context_batch=[ctx_a, ctx_b],
            bonus_tokens_np=np.zeros(2, dtype=np.int64),
            num_accepted_np=np.zeros(2, dtype=np.int64),
            accepted_draft_tokens_np=np.zeros((2, 2), dtype=np.int64),
            next_draft_tokens_np=np.zeros((2, 2), dtype=np.int64),
            overlap_pinned_np=overlap_pinned_np,
        )

        callback()

        # The whole rectangle is reset to all-valid (-1).
        assert (overlap_pinned_np == -1).all()


class TestEnqueuePrevBitmaskCallback:
    """Tests for OverlapTextGenerationPipeline._enqueue_prev_bitmask_callback."""

    def _make_pipeline(
        self, structured_output_enabled: bool = True
    ) -> OverlapTextGenerationPipeline[TextContext]:
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        mock_so.enabled = structured_output_enabled
        pipeline._structured_output = mock_so
        mock_config = MagicMock()
        mock_config.needs_bitmask_constraints = structured_output_enabled
        pipeline._pipeline_config = mock_config
        return pipeline

    def _make_prev_batch(
        self,
        prev_contexts: list[TextContext],
        num_draft_to_verify: int,
        next_draft_k: int,
    ) -> MagicMock:
        """Build a mock AsyncBatch with a spec_decode sub-object.

        ``num_draft_tokens_to_verify`` is a property returning
        ``draft_tokens_to_verify_host.shape[1]``, so mock that shape.
        ``next_draft_tokens_host.shape`` must be ``(batch, next_draft_k)``
        for the next_draft_k computation in ``_enqueue_prev_bitmask_callback``.
        """
        mock_batch = MagicMock()
        mock_batch.inputs.flat_batch = prev_contexts
        mock_spec_decode = MagicMock()
        mock_spec_decode.draft_tokens_to_verify_host = MagicMock()
        mock_spec_decode.draft_tokens_to_verify_host.shape = (
            len(prev_contexts),
            num_draft_to_verify,
        )
        # Set the property value directly so ``num_draft_tokens_to_verify > 0``
        # comparisons in _prev_batch_verified_drafts work (MagicMock does not
        # support ``>`` against int by default).
        mock_spec_decode.num_draft_tokens_to_verify = num_draft_to_verify
        mock_spec_decode.next_draft_tokens_host = MagicMock()
        mock_spec_decode.next_draft_tokens_host.shape = (
            len(prev_contexts),
            next_draft_k,
        )
        mock_spec_decode.fsm_advanced_by_callback = False
        mock_batch.spec_decode = mock_spec_decode
        return mock_batch

    def test_returns_false_when_structured_output_disabled(self) -> None:
        """Returns False immediately when structured output is not enabled."""
        pipeline = self._make_pipeline(structured_output_enabled=False)
        pipeline._spec_decode_state = MagicMock()
        pipeline._prev_batch = MagicMock()

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_spec_decode_state_is_none(self) -> None:
        """Returns False when spec decode state is not initialized."""
        pipeline = self._make_pipeline()
        pipeline._spec_decode_state = None
        pipeline._prev_batch = MagicMock()

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_persistent_buffers_missing(self) -> None:
        """Returns False when any persistent pinned buffer is absent."""
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        mock_spec_state.persistent_bonus_tokens_pinned = None
        pipeline._spec_decode_state = mock_spec_state
        pipeline._prev_batch = self._make_prev_batch([], 2, 2)

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_no_prev_batch(self) -> None:
        """Returns False when _prev_batch is None (first iteration)."""
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        for name in (
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "accepted_token_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            setattr(mock_spec_state, name, MagicMock())
        pipeline._spec_decode_state = mock_spec_state
        pipeline._prev_batch = None

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_prev_batch_has_no_spec_decode(self) -> None:
        """Returns False when prev_batch.spec_decode is None (prefill batch)."""
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        for name in (
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "accepted_token_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            setattr(mock_spec_state, name, MagicMock())
        pipeline._spec_decode_state = mock_spec_state
        mock_prev = MagicMock()
        mock_prev.spec_decode = None
        pipeline._prev_batch = mock_prev

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_prev_num_draft_tokens_zero(self) -> None:
        """Returns False when prev batch verified zero draft tokens (prefill)."""
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        for name in (
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "accepted_token_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            setattr(mock_spec_state, name, MagicMock())
        pipeline._spec_decode_state = mock_spec_state
        pipeline._prev_batch = self._make_prev_batch(
            [], num_draft_to_verify=0, next_draft_k=2
        )

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_false_when_curr_ctx_not_verifying(self) -> None:
        """Returns False when any current context has generated_length == 0.

        ``_enqueue_prev_bitmask_callback`` only fires on the steady decode
        path where the callback's bitmask is consumed in place. If the
        current batch contains a fresh prompt (generated_length == 0), it
        is not a pure decode batch; the callback must not be enqueued.
        """
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        for name in (
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "accepted_token_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            setattr(mock_spec_state, name, MagicMock())
        pipeline._spec_decode_state = mock_spec_state
        pipeline._prev_batch = self._make_prev_batch(
            [], num_draft_to_verify=2, next_draft_k=2
        )

        # curr batch has one context with generated_length == 0 (fresh prompt)
        curr_ctx = TextContext(
            request_id=RequestID("fresh"),
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        assert curr_ctx.tokens.generated_length == 0

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[curr_ctx],
            curr_verify_width=0,
        )

        assert result is False

    def test_returns_true_and_enqueues_for_decode_batch(self) -> None:
        """Returns True and dispatches via overlap_state.enqueue_async_callback
        for a decode batch."""
        pipeline = self._make_pipeline()
        pipeline._structured_output.vocab_size = 64

        batch_size = 1
        num_draft = 2
        num_positions = num_draft + 1
        vocab_size = 64

        mock_spec_state = MagicMock()
        bonus_tokens_pinned = MagicMock()
        bonus_tokens_pinned.to_numpy.return_value = np.array(
            [5], dtype=np.int64
        )
        num_accepted_pinned = MagicMock()
        num_accepted_pinned.to_numpy.return_value = np.zeros(1, dtype=np.int64)
        accepted_draft_tokens_pinned = MagicMock()
        accepted_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )
        next_draft_tokens_pinned = MagicMock()
        next_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )

        mock_spec_state.persistent_bonus_tokens_pinned = bonus_tokens_pinned
        mock_spec_state.persistent_num_accepted_pinned = num_accepted_pinned
        mock_spec_state.accepted_token_pinned = accepted_draft_tokens_pinned
        mock_spec_state.persistent_next_draft_tokens_pinned = (
            next_draft_tokens_pinned
        )
        mock_spec_state.has_precomputed_bitmask = False

        overlap_pinned = MagicMock()
        overlap_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_positions, vocab_size), dtype=np.bool_
        )
        mock_overlap_state = MagicMock()
        mock_overlap_state.pinned_bitmask = overlap_pinned
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        pipeline._devices = [MagicMock()]
        pipeline._disable_overlap = False

        rid = RequestID("r")
        prev_ctx = TextContext(
            request_id=rid,
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        # curr_ctx reuses the same request_id so it is a member of the
        # producing batch -- the steady aggregated decode path.
        curr_ctx = TextContext(
            request_id=rid,
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        # Simulate one accepted token so is_initial_prompt=False; the
        # membership guard requires every current row to be a producing-batch
        # member with is_initial_prompt=False.
        curr_ctx.update(new_token=99)

        pipeline._prev_batch = self._make_prev_batch(
            [prev_ctx], num_draft_to_verify=num_draft, next_draft_k=num_draft
        )

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_prev_bitmask_callback(
                curr_context_batch=[curr_ctx],
                curr_verify_width=num_draft,
            )

        assert result is True
        mock_overlap_state.enqueue_async_callback.assert_called_once()
        assert mock_spec_state.has_precomputed_bitmask is True
        # The producing batch's flag is set so its sync skips redundant advance.
        assert pipeline._prev_batch.spec_decode.fsm_advanced_by_callback is True

    def test_prev_batch_row_order_used_for_continuing_identification(
        self,
    ) -> None:
        """The producing batch's row order drives which curr rows are
        identified as continuing in ``_assign_bitmask_inputs``. The rows are
        read from ``_prev_batch.inputs.flat_batch`` — not from a snapshot field
        on spec_state — so the identity is always derived from the live batch."""
        pipeline = self._make_pipeline()
        pipeline._structured_output.vocab_size = 64

        batch_size = 3
        num_draft = 2
        num_positions = num_draft + 1
        vocab_size = 64

        mock_spec_state = MagicMock()
        bonus_tokens_pinned = MagicMock()
        bonus_tokens_pinned.to_numpy.return_value = np.zeros(
            batch_size, dtype=np.int64
        )
        num_accepted_pinned = MagicMock()
        num_accepted_pinned.to_numpy.return_value = np.zeros(
            batch_size, dtype=np.int64
        )
        accepted_draft_tokens_pinned = MagicMock()
        accepted_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )
        next_draft_tokens_pinned = MagicMock()
        next_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )

        mock_spec_state.persistent_bonus_tokens_pinned = bonus_tokens_pinned
        mock_spec_state.persistent_num_accepted_pinned = num_accepted_pinned
        mock_spec_state.accepted_token_pinned = accepted_draft_tokens_pinned
        mock_spec_state.persistent_next_draft_tokens_pinned = (
            next_draft_tokens_pinned
        )
        mock_spec_state.has_precomputed_bitmask = False

        overlap_pinned = MagicMock()
        overlap_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_positions, vocab_size), dtype=np.bool_
        )
        mock_overlap_state = MagicMock()
        mock_overlap_state.pinned_bitmask = overlap_pinned
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        pipeline._devices = [MagicMock()]
        pipeline._disable_overlap = False

        # Deliberately non-sorted previous-batch ordering so a stray sort
        # would corrupt the row-order contract.
        prev_rids = [
            RequestID("z"),
            RequestID("a"),
            RequestID("m"),
        ]
        prev_contexts = [
            TextContext(
                request_id=rid,
                max_length=100,
                tokens=TokenBuffer(np.array([1])),
            )
            for rid in prev_rids
        ]
        pipeline._prev_batch = self._make_prev_batch(
            prev_contexts, num_draft_to_verify=num_draft, next_draft_k=num_draft
        )

        # curr_ctx reuses one of the prev batch's request ids (here "a") so it
        # is a member of the producing batch. The test validates that membership
        # is checked against the live prev_batch.inputs.flat_batch rather than
        # any snapshot field.
        curr_ctx = TextContext(
            request_id=prev_rids[1],  # "a"
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        # Apply one token so is_initial_prompt=False; the membership guard
        # requires every current row to satisfy both conditions.
        curr_ctx.update(new_token=99)

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_prev_bitmask_callback(
                curr_context_batch=[curr_ctx],
                curr_verify_width=num_draft,
            )

        assert result is True
        assert mock_spec_state.has_precomputed_bitmask is True
        # The producing batch's flag is set; no callback_request_ids field.
        assert pipeline._prev_batch.spec_decode.fsm_advanced_by_callback is True

    def _make_pipeline_with_spec_state(
        self,
        prev_contexts: list[TextContext],
        num_draft: int = 2,
    ) -> tuple[
        OverlapTextGenerationPipeline[TextContext], MagicMock, MagicMock
    ]:
        """Return (pipeline, mock_spec_state, mock_overlap_state) wired for
        a verify batch with ``prev_contexts`` as the producing batch."""
        pipeline = self._make_pipeline()
        pipeline._structured_output.vocab_size = 64

        batch_size = len(prev_contexts)
        num_positions = num_draft + 1
        vocab_size = 64

        mock_spec_state = MagicMock()
        bonus_tokens_pinned = MagicMock()
        bonus_tokens_pinned.to_numpy.return_value = np.zeros(
            max(batch_size, 1), dtype=np.int64
        )
        num_accepted_pinned = MagicMock()
        num_accepted_pinned.to_numpy.return_value = np.zeros(
            max(batch_size, 1), dtype=np.int64
        )
        accepted_draft_tokens_pinned = MagicMock()
        accepted_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (max(batch_size, 1), num_draft), dtype=np.int64
        )
        next_draft_tokens_pinned = MagicMock()
        next_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (max(batch_size, 1), num_draft), dtype=np.int64
        )

        mock_spec_state.persistent_bonus_tokens_pinned = bonus_tokens_pinned
        mock_spec_state.persistent_num_accepted_pinned = num_accepted_pinned
        mock_spec_state.accepted_token_pinned = accepted_draft_tokens_pinned
        mock_spec_state.persistent_next_draft_tokens_pinned = (
            next_draft_tokens_pinned
        )
        mock_spec_state.has_precomputed_bitmask = False

        overlap_pinned = MagicMock()
        overlap_pinned.to_numpy.return_value = np.zeros(
            (max(batch_size, 1), num_positions, vocab_size), dtype=np.bool_
        )
        mock_overlap_state = MagicMock()
        mock_overlap_state.pinned_bitmask = overlap_pinned
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        pipeline._devices = [MagicMock()]
        pipeline._disable_overlap = False
        pipeline._prev_batch = self._make_prev_batch(
            prev_contexts,
            num_draft_to_verify=num_draft,
            next_draft_k=num_draft,
        )
        return pipeline, mock_spec_state, mock_overlap_state

    def test_disagg_transferred_row_routes_to_cold_start_preserving_constraints(
        self,
    ) -> None:
        """Disagg repro: a KV-transferred constrained row must reach cold-start
        prime, not the async callback.

        A KV-transferred row arrives with generated_length > 0 and
        is_initial_prompt=False but was never in the decode engine's previous
        producing batch. The old guard (generated_length > 0) would enqueue
        the async callback, which would then assert because the row cannot be
        attributed via rid_to_src. But even if that assert were removed, the
        callback would hit ``if ctx.matcher is None: continue`` (the matcher is
        built lazily inside compute_speculative_bitmasks on the cold-start path)
        and silently leave the bitmask all-valid (-1), dropping the row's
        grammar/schema constraints entirely.

        The membership guard must return False so _assign_bitmask_inputs takes
        the cold-start branch (has_precomputed_bitmask=False), which calls
        compute_speculative_bitmasks + prime. That path builds the matcher and
        fills a constrained bitmask. The assertion below on has_precomputed_bitmask
        is the gate: False forces cold-start; True would have taken the
        steady-state pass-through that relies on the callback having filled the
        buffer.

        The producer row is itself a continuing member with generated_length > 0,
        so the old generated_length > 0 guard would PASS and wrongly enqueue the
        callback -- that is what makes this a real regression test rather than a
        tautology. The transferred row is the sole non-member, so only the
        membership guard distinguishes the two code paths.
        """
        producer = TextContext(
            request_id=RequestID("producer"),
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        # The producer is a genuine continuing row: it was in the prev batch and
        # has generated_length > 0. Without this, the old generated_length > 0
        # guard would also return False (for the wrong reason) and the test
        # could not tell the buggy and fixed paths apart.
        producer.update(new_token=7)
        assert producer.tokens.generated_length > 0
        pipeline, mock_spec_state, mock_overlap_state = (
            self._make_pipeline_with_spec_state([producer])
        )

        # Transferred row: generated_length > 0, is_initial_prompt=False,
        # but its request_id was never in the decode engine's prev batch.
        # A constrained row in this state would have matcher=None until
        # compute_speculative_bitmasks builds it; the callback path would
        # silently skip it (all-valid), dropping constraints.
        transferred = TextContext(
            request_id=RequestID("transferred"),
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        transferred.update(new_token=99)
        assert transferred.tokens.generated_length > 0
        assert not transferred.is_initial_prompt

        result = pipeline._enqueue_prev_bitmask_callback(
            curr_context_batch=[producer, transferred],
            curr_verify_width=2,
        )

        assert result is False
        # has_precomputed_bitmask=False is the gate that forces _assign_bitmask_inputs
        # into the cold-start prime branch, which builds matchers and enforces
        # constraints for the transferred row. True here would route to the
        # steady-state pass-through and silently drop constraints.
        assert mock_spec_state.has_precomputed_bitmask is False
        # No callback enqueued: the async callback would have mis-attributed the
        # transferred row and either crashed (assert) or silently dropped its
        # grammar constraints (matcher=None skip).
        mock_overlap_state.enqueue_async_callback.assert_not_called()

    def test_disagg_all_member_batch_still_enqueues_callback(self) -> None:
        """Steady-state aggregated decode: every current row is a producing-
        batch member -- the callback path is preserved, paying no extra sync."""
        rid_a = RequestID("a")
        rid_b = RequestID("b")
        ctx_a = TextContext(
            request_id=rid_a,
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        ctx_b = TextContext(
            request_id=rid_b,
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        # Both rows continuing (is_initial_prompt=False) from the producing batch.
        ctx_a.update(new_token=10)
        ctx_b.update(new_token=11)

        pipeline, mock_spec_state, mock_overlap_state = (
            self._make_pipeline_with_spec_state([ctx_a, ctx_b])
        )

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_prev_bitmask_callback(
                curr_context_batch=[ctx_a, ctx_b],
                curr_verify_width=2,
            )

        assert result is True
        assert mock_spec_state.has_precomputed_bitmask is True
        mock_overlap_state.enqueue_async_callback.assert_called_once()

    def test_dp_padding_row_preserves_callback_path(self) -> None:
        """A fresh padding row does not turn steady decode into a mixed batch."""
        producer = TextContext(
            request_id=RequestID("producer"),
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        producer.update(new_token=10)
        padding = TextContext(
            request_id=RequestID("ordinary-id"),
            max_length=100,
            tokens=TokenBuffer(np.array([0])),
            _is_padding_ctx=True,
        )
        padding.update(new_token=0)

        pipeline, mock_spec_state, mock_overlap_state = (
            self._make_pipeline_with_spec_state([producer])
        )

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_prev_bitmask_callback(
                curr_context_batch=[producer, padding],
                curr_verify_width=2,
            )

        assert result is True
        assert mock_spec_state.has_precomputed_bitmask is True
        mock_overlap_state.enqueue_async_callback.assert_called_once()


class TestInitializeBitmaskWithGrammar:
    """Tests for initialize_bitmask behavior with grammar field.

    These tests verify that structured output bitmasks are allocated when
    a context has a grammar set (e.g., for tool calls), even if json_schema
    is not set.
    """

    def _create_overlap_pipeline_with_structured_output(
        self, enabled: bool = True
    ) -> OverlapTextGenerationPipeline[TextContext]:
        """Create a mock OverlapTextGenerationPipeline with structured output."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = enabled
        mock_structured_output.vocab_size = 32000
        mock_structured_output.allocate_bitmask.return_value = np.zeros(
            (1, 1000), dtype=np.int32
        )
        pipeline._structured_output = mock_structured_output
        return pipeline

    def _create_text_pipeline_with_structured_output(
        self, enabled: bool = True
    ) -> TextGenerationPipeline[TextContext]:
        """Create a mock TextGenerationPipeline with structured output."""
        pipeline = TextGenerationPipeline.__new__(TextGenerationPipeline)
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = enabled
        mock_structured_output.vocab_size = 32000
        mock_structured_output.allocate_bitmask.return_value = np.zeros(
            (1, 1000), dtype=np.int32
        )
        pipeline._structured_output = mock_structured_output
        return pipeline

    def test_allocates_bitmask_when_grammar_only_overlap_pipeline(self) -> None:
        """initialize_bitmask should allocate when grammar is set but json_schema is None.

        This simulates tool call scenarios where grammar is used for constrained
        decoding without a JSON schema.
        """
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Create context with grammar but no json_schema
        ctx = TextContext(
            request_id=RequestID("grammar_only"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<some llguidance grammar>",
        )
        assert ctx.json_schema is None
        assert ctx.grammar is not None

        result = pipeline.initialize_bitmask([ctx])

        # Should have allocated a bitmask
        assert result is not None

    def test_allocates_bitmask_when_grammar_only_text_pipeline(self) -> None:
        """initialize_bitmask should allocate when grammar is set (TextGenerationPipeline)."""
        pipeline = self._create_text_pipeline_with_structured_output()

        ctx = TextContext(
            request_id=RequestID("grammar_only"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<some llguidance grammar>",
        )

        result = pipeline.initialize_bitmask([ctx])

        assert result is not None

    def test_returns_none_when_both_grammar_and_json_schema_none(self) -> None:
        """initialize_bitmask should return None when neither grammar nor json_schema is set."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Create context with neither grammar nor json_schema
        ctx = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        assert ctx.json_schema is None
        assert ctx.grammar is None

        result = pipeline.initialize_bitmask([ctx])

        # Should NOT allocate a bitmask
        assert result is None

    def test_allocates_bitmask_for_grammar_even_when_response_format_schema_disabled(
        self,
    ) -> None:
        """initialize_bitmask should allocate for grammar even when enable_response_format_schema=False.

        Tool calling (grammar) should work regardless of the --enable-structured-output
        flag. Only user-provided json_schema requires the flag.
        """
        # Create pipeline with enabled=True (constrained decoding available)
        # but enable_response_format_schema=False (json_schema not allowed)
        pipeline = self._create_overlap_pipeline_with_structured_output(
            enabled=True
        )

        # Grammar for tool calls should work even without --enable-structured-output
        ctx = TextContext(
            request_id=RequestID("grammar_for_tool_call"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        result = pipeline.initialize_bitmask([ctx])

        # Should allocate bitmask for tool call grammar
        assert result is not None

    def test_allocates_bitmask_for_heterogeneous_batch_with_grammar(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch with mixed grammar/no-grammar contexts.

        A batch containing at least one context with grammar should trigger
        bitmask allocation, even if other contexts have no constraints.
        """
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with grammar (e.g., tool call)
        ctx_with_grammar = TextContext(
            request_id=RequestID("with_grammar"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        # Context without any constraint
        ctx_no_constraint = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
        )

        batch = [ctx_with_grammar, ctx_no_constraint]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_allocates_bitmask_for_heterogeneous_batch_with_json_schema(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch with mixed json_schema/no-constraint contexts."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with json_schema
        ctx_with_schema = TextContext(
            request_id=RequestID("with_schema"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            json_schema='{"type": "object"}',
        )

        # Context without any constraint
        ctx_no_constraint = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
        )

        batch = [ctx_with_schema, ctx_no_constraint]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_allocates_bitmask_for_batch_with_grammar_and_json_schema(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch mixing grammar and json_schema contexts."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with grammar (tool call)
        ctx_with_grammar = TextContext(
            request_id=RequestID("with_grammar"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        # Context with json_schema (structured response format)
        ctx_with_schema = TextContext(
            request_id=RequestID("with_schema"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
            json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
        )

        # Context with neither
        ctx_freeform = TextContext(
            request_id=RequestID("freeform"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )

        batch = [ctx_with_grammar, ctx_with_schema, ctx_freeform]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_returns_none_for_all_unconstrained_batch(self) -> None:
        """initialize_bitmask should return None when all contexts are unconstrained."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Multiple contexts, none with grammar or json_schema
        contexts = [
            TextContext(
                request_id=RequestID(f"ctx_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([i, i + 1, i + 2])),
            )
            for i in range(3)
        ]

        # Verify all contexts are unconstrained
        for ctx in contexts:
            assert ctx.json_schema is None
            assert ctx.grammar is None

        result = pipeline.initialize_bitmask(contexts)

        # Should NOT allocate a bitmask
        assert result is None


class TestAssignBitmaskInputs:
    """Tests for OverlapTextGenerationPipeline._assign_bitmask_inputs.

    The method populates model_inputs with the packed bitmask triple
    (pinned_view, wait_payload, device_scratch_view) sourced from
    ``StructuredOutputOverlapState.get_input_views``.

    Steady state (a callback ran): the async callback enqueued at the head of
    execute is the sole writer of the bitmask rectangle, so this method does no
    synchronous fill -- it only binds the views. Cold start (no prior callback)
    fills every row via ``StructuredOutputOverlapState.prime``.
    """

    _VOCAB = 64
    _PACKED_VOCAB = ceildiv(_VOCAB, 32)  # packed int32 words (1 bit per token)
    _MAX_BATCH = 4
    _K = 2  # num speculative tokens (matches num_draft_tokens_to_verify)
    _NUM_POS = _K + 1

    @staticmethod
    def _make_constrained_ctx(
        request_id: RequestID, is_initial_prompt: bool = True
    ) -> TextContext:
        """Create a constrained context with ``ctx.matcher`` set."""
        ctx = TextContext(
            request_id=request_id,
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        ctx._matcher = MagicMock()
        ctx._is_initial_prompt = is_initial_prompt
        return ctx

    @classmethod
    def _make_pipeline(
        cls,
        callback_request_ids: list[RequestID],
        has_precomputed_bitmask: bool,
    ) -> tuple[
        OverlapTextGenerationPipeline[TextContext],
        MagicMock,
        MagicMock,
        MagicMock,
        MagicMock,
    ]:
        """Build a pipeline + spec_state + overlap_state wired with mocks.

        Returns ``(pipeline, structured_output, spec_state,
        overlap_state, mock_device)``.

        ``callback_request_ids`` identifies which previous-batch request IDs
        the async callback wrote; the pipeline wires a ``_prev_batch`` mock
        whose ``flat_batch`` has exactly those IDs so ``_assign_bitmask_inputs``
        derives the continuing-row set correctly.

        ``overlap_state.get_input_views(batch_size, num_positions)`` returns
        a stable ``(pinned_view, scratch_view)`` pair so tests can assert
        identity on ``model_inputs.pinned_bitmask`` and
        ``model_inputs.device_bitmask_scratch``.

        ``overlap_state.pinned_bitmask.to_numpy()`` returns a real writeable
        numpy array so direct slot writes by the method under test succeed.
        """
        pipeline: OverlapTextGenerationPipeline[TextContext] = (
            OverlapTextGenerationPipeline.__new__(OverlapTextGenerationPipeline)
        )
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True
        mock_structured_output.compute_speculative_bitmasks.side_effect = (
            lambda context_batch, draft_tokens, num_positions: np.full(
                (len(context_batch), num_positions, cls._PACKED_VOCAB),
                -1,
                dtype=np.int32,
            )
        )
        pipeline._structured_output = mock_structured_output

        # Real writeable array backing the pinned bitmask so slot writes land.
        pinned_backing = np.zeros(
            (cls._MAX_BATCH, cls._NUM_POS, cls._PACKED_VOCAB), dtype=np.int32
        )
        pinned_view = MagicMock(name="pinned_view")
        scratch_view = MagicMock(name="scratch_view")

        mock_overlap_state = MagicMock()
        mock_overlap_state.num_positions = cls._NUM_POS
        mock_overlap_state.vocab_size = cls._VOCAB
        mock_overlap_state.packed_vocab_size = cls._PACKED_VOCAB
        mock_overlap_state.max_batch_size = cls._MAX_BATCH
        mock_overlap_state.pinned_bitmask.to_numpy.return_value = pinned_backing
        mock_overlap_state.wait_payload = MagicMock(name="wait_payload")
        mock_overlap_state.get_input_views.return_value = (
            pinned_view,
            scratch_view,
        )

        mock_spec_state = MagicMock()
        mock_spec_state.has_precomputed_bitmask = has_precomputed_bitmask
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        mock_device = MagicMock()
        pipeline._devices = [mock_device]

        # Wire _prev_batch so _assign_bitmask_inputs can derive prev_rids
        # from self._prev_batch.inputs.flat_batch. Always set a mock (even for
        # an empty producing batch) when has_precomputed_bitmask=True, because
        # the steady-state branch asserts _prev_batch is not None.
        if has_precomputed_bitmask:
            mock_prev_batch = MagicMock()
            prev_contexts = [
                TextContext(
                    request_id=rid,
                    max_length=100,
                    tokens=TokenBuffer(np.array([1])),
                )
                for rid in callback_request_ids
            ]
            mock_prev_batch.inputs.flat_batch = prev_contexts
            pipeline._prev_batch = mock_prev_batch
        else:
            pipeline._prev_batch = None

        return (
            pipeline,
            mock_structured_output,
            mock_spec_state,
            mock_overlap_state,
            mock_device,
        )

    def test_steady_state_does_no_synchronous_fill(self) -> None:
        """Steady state (a callback ran, ``has_precomputed_bitmask=True``): the
        callback is the sole writer of the rectangle, so this method performs no
        synchronous bitmask fill -- neither ``prime`` nor
        ``compute_speculative_bitmasks`` is called -- and only binds the views.

        This holds for every continuing row in the batch (the only rows that can
        appear when a callback ran: the callback is gated on the whole batch
        verifying drafts, and the scheduler routes fresh/resumed requests
        through the cold-start path instead)."""
        rid_a = RequestID("a")
        rid_b = RequestID("b")
        ctx_a = self._make_constrained_ctx(rid_a, is_initial_prompt=False)
        ctx_b = self._make_constrained_ctx(rid_b, is_initial_prompt=False)

        pipeline, structured_output, spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[rid_a, rid_b],
                has_precomputed_bitmask=True,
            )
        )

        model_inputs = MagicMock()
        draft_tokens_np = np.zeros((2, self._K), dtype=np.int64)
        pipeline._assign_bitmask_inputs(
            model_inputs=model_inputs,
            context_batch=[ctx_a, ctx_b],
            draft_tokens_np=draft_tokens_np,
            num_draft_tokens_to_verify=self._K,
        )

        structured_output.compute_speculative_bitmasks.assert_not_called()
        overlap_state.prime.assert_not_called()
        mock_device.default_queue.synchronize.assert_not_called()
        assert spec_state.has_precomputed_bitmask is False
        # Views from get_input_views are wired to model_inputs.
        overlap_state.get_input_views.assert_called_once_with(2, self._NUM_POS)
        pinned_view, scratch_view = overlap_state.get_input_views.return_value
        assert model_inputs.pinned_bitmask is pinned_view
        assert model_inputs.device_bitmask_scratch is scratch_view
        assert model_inputs.wait_payload is overlap_state.wait_payload
        # No row_map attribute (the old gather is gone).
        assert (
            not hasattr(model_inputs, "row_map")
            or model_inputs.row_map != overlap_state
        )

    def test_cold_start_calls_prime_with_full_batch_bitmask(self) -> None:
        """Cold start (prefill -> first decode, ``has_precomputed_bitmask``
        False): compute every row synchronously then call ``prime`` so the
        in-graph flag wait passes immediately."""
        rid_a = RequestID("a")
        ctx_a = self._make_constrained_ctx(rid_a)

        pipeline, structured_output, spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[],
                has_precomputed_bitmask=False,
            )
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_queue.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()
        assert spec_state.has_precomputed_bitmask is False

    def test_get_input_views_always_wired_to_model_inputs(self) -> None:
        """``model_inputs.pinned_bitmask`` and
        ``model_inputs.device_bitmask_scratch`` are the views returned by
        ``get_input_views``, regardless of branch."""
        rid_a = RequestID("a")
        ctx_a = self._make_constrained_ctx(rid_a, is_initial_prompt=False)

        pipeline, _, _, overlap_state, _ = self._make_pipeline(
            callback_request_ids=[rid_a],
            has_precomputed_bitmask=True,
        )

        model_inputs = MagicMock()
        pipeline._assign_bitmask_inputs(
            model_inputs=model_inputs,
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        pinned_view, scratch_view = overlap_state.get_input_views.return_value
        assert model_inputs.pinned_bitmask is pinned_view
        assert model_inputs.device_bitmask_scratch is scratch_view
        assert model_inputs.wait_payload is overlap_state.wait_payload

    def test_sync_fill_uses_realized_drafts_not_magic(self) -> None:
        """Synchronous-fill must build the speculative bitmask from realized drafts.

        On the whole-batch synchronous-fill path (cold start / prefill -> first
        decode, ``has_precomputed_bitmask=False``) the speculative bitmask
        must be built from the real EAGLE drafts that ``realize_future_tokens``
        scattered onto the device buffer -- NOT from the
        ``MAGIC_DRAFT_TOKEN_ID`` placeholders left in ``draft_tokens_np``.

        When the synchronous fill passes ``draft_tokens_np`` (all MAGIC) straight to
        ``compute_speculative_bitmasks``, the speculative FSM walk breaks on
        the grammar-illegal placeholder (``try_consume_tokens`` returns 0) and
        the bonus / tail slots are left unconstrained, so a grammar-illegal
        token can be sampled and committed.

        The realized host drafts are gathered through the same prev->curr
        scatter map ``realize_future_tokens`` uses (a pure row permutation of
        ``prev_batch.next_draft_tokens_host``) and passed as
        ``realized_draft_tokens_host`` to ``_assign_bitmask_inputs``. The
        synchronous fill must feed those to ``compute_speculative_bitmasks``
        instead.

        Contract asserted: the ``draft_tokens`` handed to
        ``compute_speculative_bitmasks`` on the synchronous-fill path equal the
        realized drafts, not the MAGIC placeholders.
        """
        rid_a = RequestID("a")
        ctx_a = self._make_constrained_ctx(rid_a, is_initial_prompt=False)

        pipeline, structured_output, _spec_state, _overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[],  # cold start: no callback rows
                has_precomputed_bitmask=False,  # -> whole-batch synchronous fill
            )
        )

        # ``draft_tokens_np`` holds only MAGIC placeholders -- exactly what
        # ``_execute_spec_decode`` fills it with in overlap mode (the saved
        # ``draft_tokens_to_verify`` was reset to [] at the end of the prior
        # step, so the MAGIC fallback fires).
        magic_drafts = np.full(
            (1, self._K), MAGIC_DRAFT_TOKEN_ID, dtype=np.int64
        )
        # The real drafts ``realize_future_tokens`` scattered onto the device
        # buffer for this row, made host-visible through the shared map.
        realized_drafts = np.array([[7, 9]], dtype=np.int64)
        assert not np.array_equal(realized_drafts, magic_drafts)

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=magic_drafts,
            num_draft_tokens_to_verify=self._K,
            realized_draft_tokens_host=realized_drafts,
        )

        structured_output.compute_speculative_bitmasks.assert_called_once()
        passed = (
            structured_output.compute_speculative_bitmasks.call_args.kwargs[
                "draft_tokens"
            ]
        )
        assert np.array_equal(passed, realized_drafts), (
            "synchronous-fill built the speculative bitmask from MAGIC placeholders "
            f"({np.asarray(passed).tolist()}) instead of the realized device "
            f"drafts ({realized_drafts.tolist()}); the speculative FSM walk "
            "breaks on the placeholder and bonus/tail slots go unconstrained. "
            "The synchronous fill must feed the realized host drafts."
        )


class TestHostMirrorRealizedDrafts:
    """Tests for ``_host_mirror_realized_drafts``.

    It reconstructs, on the host, the post-realize device draft buffer that the
    GPU verifies: the pre-scatter ``draft_tokens_np`` copy, overwritten for rows
    present in the previous batch by the realize scatter via ``prev_to_curr_map``
    (``prev_to_curr_map[p]`` = the current row prev-row ``p`` maps to, or
    ``_OOB_IDX``). These prove that mirror is exact -- mapped rows take prev's
    next drafts, unmapped rows keep their own ``draft_tokens_np`` value.
    """

    _K = 3  # Number of speculative tokens.

    def test_reorder_permutes_rows_by_map(self) -> None:
        """Prev ``[A, B]`` reordered to curr ``[B, A]``: each curr row gets the
        next drafts of the prev row that maps to it."""
        draft_tokens_np = np.full((2, self._K), MAGIC_DRAFT_TOKEN_ID, np.int64)
        prev_next = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int64)
        # prev A (row 0) -> curr row 1; prev B (row 1) -> curr row 0.
        prev_to_curr_map = np.array([1, 0], dtype=np.int64)

        out = _host_mirror_realized_drafts(
            draft_tokens_np, prev_to_curr_map, prev_next
        )

        assert np.array_equal(out, np.array([[20, 21, 22], [10, 11, 12]]))

    def test_unmapped_row_keeps_real_saved_drafts(self) -> None:
        """A curr row absent from the prev batch (a
        preempted/resumed request) keeps its own ``draft_tokens_np`` -- which
        may hold real saved drafts -- instead of being clobbered."""
        # Row 0 = continuing (mapped, MAGIC seed); row 1 = resumed with real
        # saved drafts and NOT in the prev batch.
        draft_tokens_np = np.array(
            [[MAGIC_DRAFT_TOKEN_ID] * self._K, [7, 8, 9]], dtype=np.int64
        )
        prev_next = np.array([[10, 11, 12]], dtype=np.int64)  # prev = [A]
        prev_to_curr_map = np.array([0], dtype=np.int64)  # A -> curr row 0

        out = _host_mirror_realized_drafts(
            draft_tokens_np, prev_to_curr_map, prev_next
        )

        assert np.array_equal(out[0], [10, 11, 12])  # mapped -> prev next
        assert np.array_equal(out[1], [7, 8, 9])  # unmapped -> kept its seed

    def test_unmapped_row_keeps_magic(self) -> None:
        """An unmapped curr row whose seed is MAGIC keeps MAGIC (it genuinely
        has no real drafts to verify)."""
        draft_tokens_np = np.array(
            [[10, 11, 12], [MAGIC_DRAFT_TOKEN_ID] * self._K], dtype=np.int64
        )
        prev_next = np.array([[10, 11, 12]], dtype=np.int64)
        prev_to_curr_map = np.array([0], dtype=np.int64)

        out = _host_mirror_realized_drafts(
            draft_tokens_np, prev_to_curr_map, prev_next
        )

        assert np.array_equal(out[1], [MAGIC_DRAFT_TOKEN_ID] * self._K)

    def test_oob_prev_row_is_skipped(self) -> None:
        """A prev row absent from the current batch (``_OOB_IDX``) writes
        nothing -- its drafts must not leak into any current row."""
        draft_tokens_np = np.full((1, self._K), MAGIC_DRAFT_TOKEN_ID, np.int64)
        # prev = [A, X]; A -> curr 0, X not in curr.
        prev_next = np.array([[10, 11, 12], [99, 99, 99]], dtype=np.int64)
        prev_to_curr_map = np.array([0, _OOB_IDX], dtype=np.int64)

        out = _host_mirror_realized_drafts(
            draft_tokens_np, prev_to_curr_map, prev_next
        )

        assert np.array_equal(out, np.array([[10, 11, 12]]))  # no 99s

    def test_does_not_mutate_input(self) -> None:
        """The seed array is copied, not mutated in place."""
        draft_tokens_np = np.full((1, self._K), MAGIC_DRAFT_TOKEN_ID, np.int64)
        prev_next = np.array([[10, 11, 12]], dtype=np.int64)
        prev_to_curr_map = np.array([0], dtype=np.int64)

        out = _host_mirror_realized_drafts(
            draft_tokens_np, prev_to_curr_map, prev_next
        )

        assert out is not draft_tokens_np
        assert np.array_equal(
            draft_tokens_np, np.full((1, self._K), MAGIC_DRAFT_TOKEN_ID)
        )

    def test_matches_independent_device_scatter_reference(self) -> None:
        """Equals an independent scatter reference: seed then, for each prev row
        with an in-range target, overwrite that current row -- the same thing
        the device graph does."""
        rng_curr = np.array(
            [[1, 2, 3], [4, 5, 6], [MAGIC_DRAFT_TOKEN_ID] * self._K],
            dtype=np.int64,
        )
        prev_next = np.array([[70, 71, 72], [80, 81, 82]], dtype=np.int64)
        # prev row 0 -> curr 2, prev row 1 -> curr 0; curr 1 stays its seed.
        prev_to_curr_map = np.array([2, 0], dtype=np.int64)

        out = _host_mirror_realized_drafts(
            rng_curr, prev_to_curr_map, prev_next
        )

        reference = rng_curr.copy()
        for p_i, c_i in enumerate(prev_to_curr_map):
            if 0 <= c_i < reference.shape[0]:
                reference[c_i] = prev_next[p_i]
        assert np.array_equal(out, reference)
        assert np.array_equal(
            out, np.array([[80, 81, 82], [4, 5, 6], [70, 71, 72]])
        )
