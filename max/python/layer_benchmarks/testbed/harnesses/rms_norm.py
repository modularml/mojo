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

"""RMSNorm harness for the model test bed.

Simple example harness showing that the framework generalizes beyond
attention layers.
"""

from __future__ import annotations

from dataclasses import dataclass

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.driver import Accelerator, Buffer, DLPackArray
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import RMSNorm

from testbed.dtypes import DTYPE_MAP
from testbed.harness import CompiledLayerBundle, LayerTestHarness
from testbed.registry import register_harness


@dataclass
class RMSNormStaticParams:
    """Static parameters for the RMSNorm harness."""

    dim: int
    dtype: str = "bfloat16"
    eps: float = 1e-6


@dataclass
class RMSNormDynamicParams:
    """Per-shape parameters for the RMSNorm harness."""

    batch_size: int
    seq_len: int


@register_harness("rms_norm")
class RMSNormHarness(
    LayerTestHarness[RMSNormStaticParams, RMSNormDynamicParams, None]
):
    """Harness for benchmarking RMSNorm layers."""

    @staticmethod
    def static_params_type() -> type:
        return RMSNormStaticParams

    @staticmethod
    def dynamic_params_type() -> type:
        return RMSNormDynamicParams

    def __init__(
        self,
        static_params: RMSNormStaticParams,
        session: InferenceSession,
        device: Accelerator,
    ) -> None:
        super().__init__(static_params, session, device)
        _, torch_dtype = DTYPE_MAP[static_params.dtype]
        self._weight = torch.randn(static_params.dim, dtype=torch_dtype)

    @property
    def name(self) -> str:
        return "rms_norm"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        max_dtype, _ = DTYPE_MAP[p.dtype]

        layer = RMSNorm(dim=p.dim, dtype=max_dtype, eps=p.eps)
        layer.load_state_dict({"weight": self._weight})

        device_ref = DeviceRef.GPU()
        input_type = TensorType(
            dtype=max_dtype,
            shape=["total_tokens", p.dim],
            device=device_ref,
        )

        with Graph("RMSNorm", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            graph.output(layer(x.tensor))

        return graph, layer.state_dict()

    def build_and_compile(self) -> CompiledLayerBundle:
        graph, weights_registry = self.build_graph()
        compiled = self.session.load(graph, weights_registry=weights_registry)

        return CompiledLayerBundle(
            compiled_model=compiled,
            device=self.device,
            session=self.session,
        )

    def prepare_inputs(
        self,
        bundle: CompiledLayerBundle,
        dynamic_params: RMSNormDynamicParams,
    ) -> tuple[list[Buffer], None]:
        p = self.static_params
        _, torch_dtype = DTYPE_MAP[p.dtype]

        total_tokens = dynamic_params.batch_size * dynamic_params.seq_len
        torch_input = torch.randn(total_tokens, p.dim, dtype=torch_dtype)
        input_tensor = Buffer.from_dlpack(torch_input).to(bundle.device)

        return [input_tensor], None

    def cleanup_inputs(
        self, bundle: CompiledLayerBundle, context: None
    ) -> None:
        pass  # No resources to clean up for RMSNorm.

    def cuda_graph_eligible(self, dynamic_params: RMSNormDynamicParams) -> bool:
        # RMSNorm is always eligible for CUDA graphs.
        return True

    def torch_reference_layer(self, device: str = "cuda") -> torch.nn.Module:
        """Return a torch RMSNorm-equivalent for correctness comparison."""
        p = self.static_params
        _, torch_dtype = DTYPE_MAP[p.dtype]

        torch_layer = torch.nn.RMSNorm(p.dim, eps=p.eps).to(
            device=device, dtype=torch_dtype
        )
        torch_layer.weight = torch.nn.Parameter(self._weight.to(device=device))
        return torch_layer

    def prepare_torch_inputs(
        self,
        execute_args: list[Buffer],
        dynamic_params: RMSNormDynamicParams,
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        return [torch.from_dlpack(execute_args[0]).to(device=device)]
