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

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.config import ConfigFileModel
from max.driver import (
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    DeviceSpec,
    Usage,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    SymbolicDim,
    TensorType,
    ops,
)
from max.nn import kernels
from max.nn.kv_cache import KVCacheInputsInterface, KVCacheParams
from max.pipelines.context import (
    EOSTracker,
    SamplingParams,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.context import FUTURE_TOKEN
from max.pipelines.lib import (
    MemoryPlan,
    ModelInputs,
    ModelOutputs,
    OverlapTextGenerationPipeline,
    PipelineConfig,
    PipelineModel,
    PipelineModelWithKVCache,
    SupportedEncoding,
)
from max.pipelines.lib.pipeline_variants import overlap_text_generation
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
)

GPU_SECONDS = 0.5
CPU_SECONDS = 0.2
FAKE_VOCAB_SIZE = 500


def draw_span_rows(
    rows: dict[str, list[tuple[float, float]]],
    xmax: float = 4.5,
    width: int = 80,
) -> None:
    if not rows:
        return

    if xmax <= 0:
        raise ValueError("xmax must be > 0")

    # Normalize by global minimum
    original_min = min(start for spans in rows.values() for start, _ in spans)

    def scale(x_norm: float) -> int:
        x_norm = max(0.0, min(x_norm, xmax))
        return min(width - 1, int(x_norm / xmax * (width - 1)))

    label_width = max(len(name) for name in rows)

    # ---- DRAW ROWS ----
    for name, spans in rows.items():
        line = [" "] * width

        for start, end in sorted(spans):
            start = start - original_min
            end = end - original_min

            if end <= 0 or start >= xmax:
                continue

            start = scale(start)
            end = scale(end)

            start = max(0, min(start, width - 1))
            end = max(0, min(end, width))

            # Enforce minimum width
            if (end - start) < 2:
                start = max(0, start - 1)
                end = min(width, end + 1)

            length = end - start

            # ---- Render ----
            line[start] = "["
            if length > 2:
                for pos in range(start + 1, end - 1):
                    if 0 <= pos < width:
                        line[pos] = "█"
            line[end - 1] = "]"

        print(f"{name:>{label_width}} | " + "".join(line))

    # ---- AXIS ----
    print(" " * (label_width + 1) + "+" + "-" * width)

    max_tick = math.floor(xmax)

    tick_line = [" "] * width
    label_line = [" "] * width

    for val in range(max_tick + 1):
        pos = scale(float(val))
        tick_line[pos] = "|"

        label = str(val)
        start_pos = max(0, min(width - len(label), pos - len(label) // 2))
        for i, ch in enumerate(label):
            label_line[start_pos + i] = ch

    padding = " " * (label_width + 3)
    print(padding + "".join(tick_line))
    print(padding + "".join(label_line))


class FakeSamplingConfig(ConfigFileModel):
    enable_penalties: bool = False
    enable_variable_logits: bool = False
    in_dtype: DType = DType.float32
    out_dtype: DType = DType.float32
    enable_structured_output: bool = False
    structured_output_backend: str | None = None
    sample_on_host: bool = False
    enable_min_tokens: bool = False


class FakeModelConfig(ConfigFileModel):
    model_path: str
    huggingface_config: Any
    device_specs: list[DeviceSpec]
    kv_cache: Any
    quantization_encoding: SupportedEncoding = "float32"
    enable_echo: bool = False
    data_parallel_degree: int = 1

    def resolved_weight_paths(self) -> list[Path]:
        return []


class FakeRuntimeConfig(ConfigFileModel):
    execute_empty_batches: bool = False
    enable_overlap_scheduler: bool = False
    device_graph_capture: bool = False
    max_batch_size: int = 999
    pipeline_role: str = "prefill_and_decode"
    reasoning_parser: str | None = None
    tool_parser: str | None = None


class FakeSpeculativeConfig(ConfigFileModel):
    num_speculative_tokens: int = 0


class FakePipelineConfig(ConfigFileModel):
    model: FakeModelConfig
    sampling: FakeSamplingConfig
    runtime: FakeRuntimeConfig = FakeRuntimeConfig()
    enable_echo: bool = False
    debug_verify_replay: bool = False
    speculative: FakeSpeculativeConfig | None = None

    def configure_session(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def needs_bitmask_constraints(self) -> bool:
        return (
            self.sampling.enable_structured_output
            or self.runtime.tool_parser is not None
        )


@dataclass
class FakeModelInputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer
    sleep_duration: Buffer
    arange: Buffer


def build_graph(device_ref: DeviceRef) -> Model:
    """Builds a graph that mimics the behavior of a LLM that performs X -> X+1.

    Given:
      - tokens: [0, 44, 45, 46, 47, 1, 2]
      - input_row_offsets: [0, 3, 7, 10]
      - sleep_duration: [3.14]

    The graph will:
      - take 3.14 seconds to execute
      - produce the following logits:
        [
          [ -INF,  INF, -INF, -INF, ... ] # INF @ idx=1
          [ -INF, -INF, -INF, -INF, ... ] # INF @ idx=48
          [ -INF, -INF, -INF,  INF, ... ] # INF @ idx=3
        ]

    Then when we sample the logits we will produce next_tokens=[1, 48, 3].
    """
    with Graph(
        "my_lil_llm",
        input_types=[
            # tokens
            TensorType(
                DType.int64, [SymbolicDim("total_seq_len")], device=device_ref
            ),
            # input row offsets
            TensorType(
                DType.uint32,
                [SymbolicDim("input_row_offsets_len")],
                device=device_ref,
            ),
            # sleep duration
            BufferType(DType.float64, [1], device=DeviceRef.CPU()),
            # arange
            TensorType(
                DType.int64, [SymbolicDim("arange_len")], device=device_ref
            ),
        ],
    ) as graph:
        tokens_input, input_row_offsets_input, sleep_duration_input, arange = (
            graph.inputs
        )
        tokens = tokens_input.tensor
        input_row_offsets = input_row_offsets_input.tensor
        sleep_duration = sleep_duration_input.buffer
        arange = arange.tensor
        batch_size = input_row_offsets.shape[0] - 1

        gather_indices = input_row_offsets[1:] - 1
        last_tokens = ops.gather(
            input=tokens.tensor, indices=gather_indices, axis=0
        )
        next_tokens = last_tokens + 1
        scatter_indices = ops.stack([arange[:batch_size], next_tokens], axis=1)
        neg_inf = ops.constant(-12345.0, DType.float32, device=device_ref)
        pos_inf = ops.constant(12345.0, DType.float32, device=device_ref)
        logits = ops.broadcast_to(neg_inf, [batch_size, Dim(FAKE_VOCAB_SIZE)])
        updates = ops.broadcast_to(pos_inf, [batch_size])
        logits = ops.scatter_nd(
            input=logits,
            updates=updates,
            indices=scatter_indices,
        )
        kernels.sleep(sleep_duration.buffer, device_ref=device_ref)
        graph.output(logits)
    device = device_ref.to_device()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    return model


class FakePipelineModel(PipelineModelWithKVCache[TextContext]):
    def __init__(
        self, pipeline_config: FakePipelineConfig, *args: Any, **kwargs: Any
    ) -> None:
        self.kv_params = MagicMock(spec=KVCacheParams)
        self.enable_overlap_scheduler = (
            pipeline_config.runtime.enable_overlap_scheduler
        )
        self.device = Accelerator()
        self.kv_cache_config = MagicMock()
        # max_seq_len is a read-only view of the plan on the base class.
        self.memory_plan = MemoryPlan(
            planned_max_batch_size=1, footprint=0, planned_max_length=9999
        )
        print(f"Building graph for device {self.device}")
        t0 = time.time()
        self.model = build_graph(device_ref=DeviceRef.from_device(self.device))
        t1 = time.time()
        print(f"Graph built in {t1 - t0} seconds")

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        del kv_cache_inputs, return_n_logits  # Unused args

        assert len(replica_batches) == 1, "DP must be 1"

        batch = replica_batches[0]
        batch_size = len(batch)
        sleep_duration = Buffer.from_numpy(
            np.array([GPU_SECONDS], dtype=np.float64)
        )
        active_lengths = [ctx.tokens.active_length for ctx in batch]
        total_seq_len = sum(active_lengths)
        tokens = DevicePinnedBuffer(
            shape=[total_seq_len],
            dtype=DType.int64,
            device=self.device,
        )
        np.concatenate(
            [ctx.tokens.active for ctx in batch], out=tokens.to_numpy()
        )
        input_row_offsets = DevicePinnedBuffer(
            shape=[len(batch) + 1],
            dtype=DType.uint32,
            device=self.device,
        )
        np.cumsum(
            [0] + active_lengths,
            dtype=np.int64,
            out=input_row_offsets.to_numpy(),
        )
        arange = DevicePinnedBuffer(
            dtype=DType.int64,
            shape=[batch_size],
            device=self.device,
        )
        arange.to_numpy()[:] = np.arange(
            start=0, stop=batch_size, dtype=np.int64
        )

        return FakeModelInputs(
            tokens=tokens.to(self.device),
            input_row_offsets=input_row_offsets.to(self.device),
            sleep_duration=sleep_duration,
            arange=arange.to(self.device),
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, FakeModelInputs)
        if not self.enable_overlap_scheduler:
            Accelerator().synchronize()

        (logits,) = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.sleep_duration,
            model_inputs.arange,
        )

        if not self.enable_overlap_scheduler:
            Accelerator().synchronize()

        return ModelOutputs(logits=logits)


# Delete all abstract methods so python doesn't complain about unimplemented
# abstract methods (this is extremely cursed)
FakePipelineModel.__abstractmethods__ = frozenset()


def create_context(
    isl: int = 64,
    osl: int = 64,
    offset: int = 0,
    temperature: float = 1.0,
    eos_token_ids: set[int] | None = None,
) -> TextContext:
    kwargs: dict[str, Any] = {}
    if eos_token_ids is not None:
        kwargs["eos_tracker"] = EOSTracker(eos_token_ids=eos_token_ids)
    return TextContext(
        request_id=RequestID(),
        max_length=isl + osl,
        tokens=TokenBuffer(np.arange(isl) + offset),
        sampling_params=SamplingParams(temperature=temperature),
        **kwargs,
    )


def monkeypatch_weight_and_kvcache_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for func in [
        "load_weights",
        "weights_format",
        "load_kv_manager",
    ]:
        monkeypatch.setattr(overlap_text_generation, func, MagicMock())


def create_overlap_pipeline(
    enable_overlap_scheduler: bool,
    disable_overlap: bool = False,
    pipeline_role: str = "prefill_and_decode",
    speculative: FakeSpeculativeConfig | None = None,
) -> OverlapTextGenerationPipeline[Any]:
    sampling_config = FakeSamplingConfig(enable_penalties=False)
    model_config = FakeModelConfig(
        model_path="test_model",
        huggingface_config=MagicMock(),
        device_specs=[DeviceSpec(id=0, device_type="gpu")],
        kv_cache=MagicMock(),
    )
    runtime = FakeRuntimeConfig(
        enable_overlap_scheduler=enable_overlap_scheduler,
        pipeline_role=pipeline_role,
    )
    pipeline_config = FakePipelineConfig(
        model=model_config,
        sampling=sampling_config,
        runtime=runtime,
        speculative=speculative,
    )
    pipeline = OverlapTextGenerationPipeline(
        pipeline_config=cast(PipelineConfig, pipeline_config),
        pipeline_model=cast(type[PipelineModel[Any]], FakePipelineModel),
        weight_adapters=MagicMock(),
        tokenizer=MagicMock(spec=[]),
        memory_plan=MemoryPlan(
            planned_max_batch_size=runtime.max_batch_size or 1,
            footprint=0,
            planned_max_length=None,
            device_specs=tuple(model_config.device_specs),
        ),
        disable_overlap=disable_overlap,
    )
    return pipeline


def fake_cpu_pre_or_post_processing() -> None:
    time.sleep(CPU_SECONDS)


def prime_host_buffer_cache() -> None:
    t = Buffer(
        shape=[1024 * 1024],
        dtype=DType.int8,
        device=Accelerator(),
        usage=Usage.STAGING,
    )
    del t


"""
In the below test, we record some spans and plot them in an ascii chart.
Note that each span corresponds to CPU execution time. Due to lack of CUDA Events
with timing, we cannot get GPU timing spans.

Overlap=True:
 Preprocess | [█][██]        [██]     [██]     [█]
    Execute |   []   [███]       []      [█]      []
Postprocess |             [█]      [█]     [██]     [██]
            +--------------------------------------------------------------------------------
              |                |                 |                |                 |
              0                1                 2                3                 4
Actual: 2.40s, Expected: 2.40s, Error: 0.00s

Overlap=False:
 Preprocess | [█]         [█]             [█]             [█]            [██]
    Execute |    [███████]   [███████]       [███████]       [███████]       [███████]
Postprocess |                         [██]            [██]            [█]             [█]
            +--------------------------------------------------------------------------------
              |                |                 |                |                 |
              0                1                 2                3                 4
Actual: 4.31s, Expected: 4.30s, Error: 0.01s
"""


@pytest.mark.parametrize(
    "enable_overlap_scheduler,expected_duration", [(True, 2.4), (False, 4.3)]
)
def test_overlap_execution(
    monkeypatch: pytest.MonkeyPatch,
    enable_overlap_scheduler: bool,
    expected_duration: float,
) -> None:
    monkeypatch_weight_and_kvcache_loading(monkeypatch)
    prime_host_buffer_cache()

    pipeline = create_overlap_pipeline(
        enable_overlap_scheduler=enable_overlap_scheduler
    )

    num_trials = 3
    for _trial in range(num_trials):
        _ = pipeline.execute(TextGenerationInputs(batches=[[]]))

        req_a = create_context(isl=17, osl=1, offset=100)
        req_b = create_context(isl=42, osl=4, offset=200)
        req_c = create_context(isl=77, osl=2, offset=300)
        active_requests = {
            req_a.request_id: req_a,
            req_b.request_id: req_b,
            req_c.request_id: req_c,
        }
        generated_tokens: dict[RequestID, list[int]] = {
            req_a.request_id: [],
            req_b.request_id: [],
            req_c.request_id: [],
        }
        preprocess_spans: list[tuple[float, float]] = []
        execute_spans: list[tuple[float, float]] = []
        postprocess_spans: list[tuple[float, float]] = []
        start_time = time.time()
        iters = 0
        while active_requests:
            print()
            print("-" * 80)
            print(f"Running iteration {iters + 1}")

            span_start = time.time()
            fake_cpu_pre_or_post_processing()
            span_end = time.time()
            preprocess_spans.append((span_start, span_end))

            span_start = time.time()
            inputs = TextGenerationInputs(
                batches=[list(active_requests.values())]
            )
            outputs = pipeline.execute(inputs)
            span_end = time.time()
            execute_spans.append((span_start, span_end))

            if outputs:
                span_start = time.time()
                fake_cpu_pre_or_post_processing()
                span_end = time.time()
                postprocess_spans.append((span_start, span_end))

            # Filter out outputs for requests that are not active anymore.
            outputs = {
                req_id: output
                for req_id, output in outputs.items()
                if req_id in active_requests
            }

            for req_id, output in outputs.items():
                generated_tokens[req_id].extend(output.tokens)
                if output.is_done and req_id in active_requests:
                    del active_requests[req_id]

            iters += 1
        end_time = time.time()
        elapsed = end_time - start_time

        # We should run 5 iterations because the largest osl is 4 (req_b)
        # Recall that overlap scheduler may run for one more iteration than needed
        assert iters == 5

        # Check that the generated tokens are what we expect.
        # We exclude that last token since it is undefined.
        assert generated_tokens[req_a.request_id] == [117]
        assert generated_tokens[req_b.request_id] == [242, 243, 244, 245]
        assert generated_tokens[req_c.request_id] == [377, 378]

        error = abs(elapsed - expected_duration)

        draw_span_rows(
            {
                "Preprocess": preprocess_spans,
                "Execute": execute_spans,
                "Postprocess": postprocess_spans,
            }
        )
        print(
            f"Actual: {elapsed:.2f}s, Expected: {expected_duration:.2f}s, Error: {error:.2f}s"
        )

        # Disable check since this is unreliable in CI
        #
        # For the last trial, ensure that the error is less than 1 second.
        # Don't check this for the other trials since we need to warmup the kernels.
        # if trial == num_trials - 1:
        #     assert error < 1.0


def test_overlap_execution_with_preemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch_weight_and_kvcache_loading(monkeypatch)
    prime_host_buffer_cache()

    pipeline = create_overlap_pipeline(enable_overlap_scheduler=True)
    context = create_context(isl=17, offset=100)
    req_id = context.request_id

    def create_inputs(
        context: TextContext,
    ) -> TextGenerationInputs[TextContext]:
        return TextGenerationInputs(batches=[[context]])

    out = pipeline.execute(create_inputs(context))
    assert req_id not in out
    out = pipeline.execute(create_inputs(context))
    assert out[req_id].tokens == [117]
    out = pipeline.execute(create_inputs(context))
    assert out[req_id].tokens == [118]

    context.reset()  # simulate a preemption

    out = pipeline.execute(create_inputs(context))
    assert req_id not in out
    out = pipeline.execute(create_inputs(context))
    assert out[req_id].tokens == [119]

    # Expected tokens are [100, 101, ..., 118, 119, FUTURE_TOKEN]
    assert context.tokens.all.tolist() == (
        list(range(100, 120)) + [FUTURE_TOKEN]
    )


def test_disable_overlap_returns_outputs_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify disable_overlap=True returns current-batch outputs in the same
    execute() call, never deferring them to the next iteration."""
    monkeypatch_weight_and_kvcache_loading(monkeypatch)
    prime_host_buffer_cache()

    pipeline = create_overlap_pipeline(
        enable_overlap_scheduler=True,
        disable_overlap=True,
    )

    def create_inputs(
        contexts: list[TextContext],
    ) -> TextGenerationInputs[TextContext]:
        return TextGenerationInputs(batches=[contexts])

    # --- Single request, multiple generation steps ---
    req_a = create_context(isl=17, osl=3, offset=100)
    req_a_id = req_a.request_id

    # Every execute() must return outputs immediately and never hold a
    # deferred _prev_batch.
    out = pipeline.execute(create_inputs([req_a]))
    assert not pipeline.has_pending_outputs()
    assert req_a_id in out
    assert out[req_a_id].tokens == [117]

    out = pipeline.execute(create_inputs([req_a]))
    assert not pipeline.has_pending_outputs()
    assert out[req_a_id].tokens == [118]

    out = pipeline.execute(create_inputs([req_a]))
    assert not pipeline.has_pending_outputs()
    assert out[req_a_id].tokens == [119]

    # --- Empty batch returns empty outputs and no pending state ---
    out = pipeline.execute(create_inputs([]))
    assert not pipeline.has_pending_outputs()
    assert len(out) == 0


def test_overlap_reuses_generated_tokens_pinned_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-step generated-token D2H pinned buffer is allocated once and
    reused across overlap decode steps (no page-locking allocation per step),
    while producing the same tokens as the deterministic fake model.

    Guards the ``_sample_logits`` optimization: a single persistent buffer is
    safe only because the previous batch's copy is read (in
    ``sync_and_process_outputs``) strictly before the next batch's copy is
    written. If that ordering broke, the token sequences below would corrupt.
    """
    monkeypatch_weight_and_kvcache_loading(monkeypatch)
    prime_host_buffer_cache()

    pipeline = create_overlap_pipeline(enable_overlap_scheduler=True)

    # Two requests of different output lengths so the decode batch shrinks
    # (2 -> 1) mid-run, exercising the ``[:batch_size]`` slice of the reused
    # buffer without reallocation.
    req_a = create_context(isl=17, osl=1, offset=100)
    req_b = create_context(isl=42, osl=3, offset=200)
    active = {req_a.request_id: req_a, req_b.request_id: req_b}
    generated: dict[RequestID, list[int]] = {
        req_a.request_id: [],
        req_b.request_id: [],
    }

    seen_buffer_ids: set[int] = set()
    while active:
        inputs = TextGenerationInputs(batches=[list(active.values())])
        outputs = pipeline.execute(inputs)

        # After a decode step samples, the persistent buffer must exist and be
        # sized for the max batch size, not the current (possibly smaller) one.
        buf = pipeline._pinned_generated_tokens_host
        assert buf is not None
        assert int(buf.shape[0]) == pipeline.max_batch_size
        seen_buffer_ids.add(id(buf))

        for req_id, output in outputs.items():
            if req_id in active:
                generated[req_id].extend(output.tokens)
                if output.is_done:
                    del active[req_id]

    # Deterministic fake model emits last_token + 1 each step.
    assert generated[req_a.request_id] == [117]
    assert generated[req_b.request_id] == [242, 243, 244]

    # The buffer object was reused across every decode step (allocated once).
    assert len(seen_buffer_ids) == 1
