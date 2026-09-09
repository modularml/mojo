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

"""Layer-agnostic test runner: benchmark, IR dump, correctness."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Generic, TypeAlias

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from benchmark_utils import (
    BENCHMARK_NVTX_RANGE,
    BenchmarkStats,
    compute_stats,
    measure_gpu_latency,
    measure_gpu_latency_cuda_graph,
)
from max._core.profiler import Trace
from max.driver import CPU, Accelerator, Buffer, DLPackArray
from max.engine import InferenceSession, Model
from max.profiler import set_gpu_profiling_state

from testbed.correctness import CorrectnessResult, compare_outputs
from testbed.harness import (
    CompiledLayerBundle,
    ContextT,
    DynamicParamsT,
    LayerTestHarness,
    StaticParamsT,
)
from testbed.ir_dump import dump_mo_ir

ModelInitializer: TypeAlias = Callable[
    [InferenceSession, Mapping[str, DLPackArray]], Model
]
"""Binds weights onto a model that was compiled elsewhere.

Given the session and the harness's weights registry, returns a model ready to
execute. Lets a caller substitute a precompiled artifact for the compile the
harness would otherwise run (see ``docs/internal/CompileOnCpuRunOnGpu.md``).
"""


def create_session(
    num_devices: int = 1,
) -> tuple[InferenceSession, Accelerator]:
    """Create an inference session with the requested number of GPUs."""
    devices = [Accelerator(id=i) for i in range(num_devices)]
    all_devices = [CPU(0), *devices] if num_devices > 1 else devices
    session = InferenceSession(devices=all_devices)
    return session, devices[0]


class LayerTestRunner(Generic[StaticParamsT, DynamicParamsT, ContextT]):
    """Wraps a LayerTestHarness and provides benchmark/IR-dump/correctness.

    Usage::

        harness = AttentionWithRopeHarness(static_params, session, device)
        runner = LayerTestRunner(harness)
        results = runner.benchmark(shapes, iterations=50, warmup=5)

    Args:
        harness: The layer harness to drive.
        init_model: Initializes a model compiled elsewhere, in place of the
            harness's own compile. Pass this to run a graph that was
            precompiled to a MEF on CPU; ``None`` compiles here.
    """

    def __init__(
        self,
        harness: LayerTestHarness[StaticParamsT, DynamicParamsT, ContextT],
        init_model: ModelInitializer | None = None,
    ) -> None:
        self.harness = harness
        self._init_model = init_model
        self._bundle: CompiledLayerBundle | None = None

    def _ensure_compiled(self) -> CompiledLayerBundle:
        """Compile the layer if not already compiled."""
        if self._bundle is not None:
            return self._bundle

        if self._init_model is None:
            self._bundle = self.harness.build_and_compile()
        else:
            # The graph itself is already compiled; it's rebuilt only for the
            # weights registry (and the reference weights a harness stashes
            # while building), then dropped.
            _, weights_registry = self.harness.build_graph()
            self._bundle = CompiledLayerBundle(
                compiled_model=self._init_model(
                    self.harness.session, weights_registry
                ),
                device=self.harness.device,
                session=self.harness.session,
            )
        return self._bundle

    @property
    def bundle(self) -> CompiledLayerBundle:
        return self._ensure_compiled()

    # ------------------------------------------------------------------ #
    # Benchmark mode
    # ------------------------------------------------------------------ #

    def benchmark(
        self,
        shapes: Sequence[DynamicParamsT],
        iterations: int = 50,
        warmup: int = 5,
        profile: bool = False,
    ) -> list[tuple[str, BenchmarkStats]]:
        """Run benchmark across all shapes.

        When ``profile=True``, enables GPU profiling state so that NVTX
        ranges (``testbed/benchmark`` outer, ``iteration/<n>`` inner)
        are emitted. nsys can then capture only the timed region via
        ``--capture-range=nvtx --nvtx-capture='testbed/benchmark'``.

        Args:
            shapes: List of dynamic_params objects for the harness.
            iterations: Number of timed iterations per shape.
            warmup: Number of warmup iterations per shape.
            profile: If True, enable GPU profiling state for NVTX traces.

        Returns:
            List of (label, BenchmarkStats) tuples.
        """
        bundle = self._ensure_compiled()
        if profile:
            set_gpu_profiling_state("detailed")

        results: list[tuple[str, BenchmarkStats]] = []

        for dynamic_params in shapes:
            execute_args, context = self.harness.prepare_inputs(
                bundle, dynamic_params
            )
            bundle.device.synchronize()

            use_cuda_graph = self.harness.cuda_graph_eligible(dynamic_params)
            with Trace(BENCHMARK_NVTX_RANGE):
                if use_cuda_graph:
                    graph_key = _dynamic_params_hash(dynamic_params)
                    latencies = measure_gpu_latency_cuda_graph(
                        bundle.compiled_model,
                        execute_args,
                        bundle.device,
                        graph_key=graph_key,
                        num_iterations=iterations,
                        num_warmup=warmup,
                    )
                else:

                    def _run(
                        args: list[Buffer] = execute_args,
                    ) -> object:
                        return bundle.compiled_model.execute(*args)

                    latencies = measure_gpu_latency(
                        _run,
                        bundle.device,
                        num_iterations=iterations,
                        num_warmup=warmup,
                    )
            self.harness.cleanup_inputs(bundle, context)

            stats = compute_stats(latencies)
            label = _make_label(dynamic_params)
            results.append((label, stats))

        return results

    # ------------------------------------------------------------------ #
    # IR dump mode
    # ------------------------------------------------------------------ #

    def dump_ir(self, mo_path: str | Path) -> Path:
        """Dump MO-level MLIR IR for the layer graph.

        Does not require a GPU — only builds the graph, does not compile.

        Args:
            mo_path: Path to write the .mo.mlir file.

        Returns:
            The output path.
        """
        graph, _ = self.harness.build_graph()
        return dump_mo_ir(graph, Path(mo_path))

    # ------------------------------------------------------------------ #
    # Correctness mode
    # ------------------------------------------------------------------ #

    def correctness(
        self,
        shapes: Sequence[DynamicParamsT],
        atol: float = 1e-2,
        rtol: float = 1e-2,
        cos_threshold: float = 0.001,
    ) -> list[CorrectnessResult]:
        """Compare MAX output against torch reference for each shape.

        Args:
            shapes: List of dynamic_params objects.
            atol: Absolute tolerance for element-wise comparison.
            rtol: Relative tolerance for element-wise comparison.
            cos_threshold: Maximum cosine distance threshold.

        Returns:
            List of CorrectnessResult per shape.
        """
        # Compile first so harness can store weights for the torch reference.
        bundle = self._ensure_compiled()

        torch_module = self.harness.torch_reference_layer()
        results = []

        for dynamic_params in shapes:
            # MAX forward pass.
            execute_args, context = self.harness.prepare_inputs(
                bundle, dynamic_params
            )
            bundle.device.synchronize()
            max_output = bundle.compiled_model.execute(*execute_args)
            bundle.device.synchronize()

            # Convert MAX output to numpy. BF16 buffers can't go directly
            # to numpy (unsupported dtype), so route through torch.
            max_np = torch.from_dlpack(max_output[0]).float().cpu().numpy()

            # Torch forward pass — must happen before cleanup_inputs so
            # execute_args buffers remain valid.
            torch_inputs = self.harness.prepare_torch_inputs(
                execute_args, dynamic_params
            )
            self.harness.cleanup_inputs(bundle, context)
            with torch.no_grad():
                raw_output = torch_module(*torch_inputs)
            torch_output = self.harness.postprocess_torch_output(raw_output)
            torch_np = torch_output.float().cpu().numpy()

            result = compare_outputs(
                max_np,
                torch_np,
                atol=atol,
                rtol=rtol,
                cos_threshold=cos_threshold,
            )
            result.label = _make_label(dynamic_params)
            results.append(result)

        return results


_LABEL_ABBREVS: Mapping[str, str] = {
    "batch_size": "b",
    "seq_len": "q",
    "ctx_len": "ctx",
    "context_len": "ctx",
}


def _dynamic_params_hash(dynamic_params: object) -> int:
    """Compute a hash suitable for CUDA graph keys."""
    if hasattr(dynamic_params, "__dict__"):
        items = sorted(vars(dynamic_params).items())
    elif isinstance(dynamic_params, dict):
        items = sorted(dynamic_params.items())
    else:
        items = [(str(dynamic_params), 0)]
    return hash(tuple(items)) & 0x7FFFFFFF


def _make_label(dynamic_params: object) -> str:
    """Create a concise label from dynamic params."""
    if hasattr(dynamic_params, "__dict__"):
        items = vars(dynamic_params).items()
    elif isinstance(dynamic_params, dict):
        items = dynamic_params.items()
    else:
        return str(dynamic_params)

    parts = []
    for k, v in items:
        prefix = _LABEL_ABBREVS.get(k, k)
        parts.append(f"{prefix}{v}")
    return "_".join(parts)
