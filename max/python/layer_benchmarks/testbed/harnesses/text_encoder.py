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

"""UMT5 TextEncoder harness for the model test bed.

Tests the full UMT5 encoder graph including the on-device post-processing
(slice to embed_seq_len, mask padding positions) used by the WAN pipeline.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.driver import Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.layer import Module
from max.pipelines.architectures.umt5.model_config import UMT5ConfigBase
from max.pipelines.architectures.umt5.umt5 import UMT5EncoderModel
from transformers import UMT5Config as HFUmt5Config
from transformers import UMT5EncoderModel as HFUmt5EncoderModel

from testbed.dtypes import DTYPE_MAP
from testbed.harness import CompiledLayerBundle, LayerTestHarness
from testbed.registry import register_harness


@dataclass
class TextEncoderStaticParams:
    """Static parameters for the TextEncoder harness."""

    vocab_size: int = 256384
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "gated-gelu"
    embed_seq_len: int = 226
    dtype: str = "bfloat16"


@dataclass
class TextEncoderDynamicParams:
    """Per-shape parameters for the TextEncoder harness."""

    batch_size: int
    seq_len: int
    # Number of real (non-padding) tokens. If None, all tokens are real.
    num_real_tokens: int | None = None


def _make_umt5_config(p: TextEncoderStaticParams) -> UMT5ConfigBase:
    """Create a UMT5ConfigBase from harness static params."""
    return UMT5ConfigBase(
        vocab_size=p.vocab_size,
        d_model=p.d_model,
        d_kv=p.d_kv,
        d_ff=p.d_ff,
        num_layers=p.num_layers,
        num_heads=p.num_heads,
        relative_attention_num_buckets=p.relative_attention_num_buckets,
        relative_attention_max_distance=p.relative_attention_max_distance,
        layer_norm_epsilon=p.layer_norm_epsilon,
        feed_forward_proj=p.feed_forward_proj,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
        is_decoder=False,
        use_cache=False,
        is_encoder_decoder=False,
    )


def _make_hf_config(p: TextEncoderStaticParams) -> HFUmt5Config:
    """Create an HF UMT5Config from harness static params."""
    return HFUmt5Config(
        vocab_size=p.vocab_size,
        d_model=p.d_model,
        d_kv=p.d_kv,
        d_ff=p.d_ff,
        num_layers=p.num_layers,
        num_heads=p.num_heads,
        relative_attention_num_buckets=p.relative_attention_num_buckets,
        relative_attention_max_distance=p.relative_attention_max_distance,
        layer_norm_epsilon=p.layer_norm_epsilon,
        feed_forward_proj=p.feed_forward_proj,
        is_decoder=False,
        is_encoder_decoder=False,
    )


def _discover_weight_shapes(module: Module) -> dict[str, list[int]]:
    """Extract weight name -> shape from a module without touching SSA values."""
    shapes: dict[str, list[int]] = {}
    for name, param in module.state_dict().items():
        assert isinstance(param, Buffer)
        shapes[name] = list(param.shape)
    return shapes


_WEIGHT_SEED = 0xD1FF


def _generate_random_weights(
    shapes: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    """Generate deterministic random torch CPU weights from a name->shape map.

    Uses a fixed seed so the MAX graph and HF torch reference see identical
    weights without needing to cache them between calls.
    """
    gen = torch.Generator(device="cpu").manual_seed(_WEIGHT_SEED)
    return {
        name: (
            torch.randn(
                shape, dtype=torch.bfloat16, generator=gen, device="cpu"
            )
            * 0.02
        ).detach()
        for name, shape in shapes.items()
    }


def _torch_to_hf_state_dict(
    torch_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap MAX/our weight keys to HuggingFace UMT5EncoderModel keys.

    Our module uses:
      shared.weight, encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
      encoder.block.{i}.layer.0.layer_norm.weight
      encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
      encoder.block.{i}.layer.1.layer_norm.weight
      encoder.final_layer_norm.weight

    HF UMT5EncoderModel uses the same structure but wraps everything under
    the 'encoder' prefix, and the embedding is at
    'encoder.embed_tokens.weight'.
    """
    hf_dict: dict[str, torch.Tensor] = {}
    for key, value in torch_weights.items():
        if key == "shared.weight":
            # HF ties shared.weight and encoder.embed_tokens.weight.
            # Must provide both for strict loading.
            hf_dict["shared.weight"] = value
            hf_dict["encoder.embed_tokens.weight"] = value
        elif key.startswith("encoder."):
            hf_dict[key] = value
        else:
            hf_dict[key] = value
    return hf_dict


@register_harness("text_encoder")
class TextEncoderHarness(
    LayerTestHarness[TextEncoderStaticParams, TextEncoderDynamicParams, None]
):
    """Harness for benchmarking and testing the WAN TextEncoder graph.

    Tests the full UMT5 encoder with on-device post-processing
    (slice to embed_seq_len + mask padding).
    """

    @staticmethod
    def static_params_type() -> type:
        return TextEncoderStaticParams

    @staticmethod
    def dynamic_params_type() -> type:
        return TextEncoderDynamicParams

    def __init__(
        self,
        static_params: TextEncoderStaticParams,
        session: InferenceSession,
        device: Accelerator,
    ) -> None:
        super().__init__(static_params, session, device)
        self._umt5_config = _make_umt5_config(static_params)

        # Parse feed_forward config (same as UMT5EncoderModel.__init__).
        act_info = self._umt5_config.feed_forward_proj.split("-")
        self._umt5_config.dense_act_fn = act_info[-1]
        self._umt5_config.is_gated_act = act_info[0] == "gated"
        if self._umt5_config.feed_forward_proj == "gated-gelu":
            self._umt5_config.dense_act_fn = "gelu_new"

        max_dtype, _ = DTYPE_MAP[static_params.dtype]
        dev_ref = DeviceRef.GPU()

        # Build a throwaway module just to discover weight shapes, then
        # discard it. A fresh module is created inside build_graph() so
        # Weight objects are bound to the correct Graph context.
        tmp_module = UMT5EncoderModel(
            self._umt5_config, dtype=max_dtype, device=dev_ref
        )
        self._weight_shapes = _discover_weight_shapes(tmp_module)
        del tmp_module
        gc.collect()

    @property
    def name(self) -> str:
        return "text_encoder"

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        max_dtype, _ = DTYPE_MAP[p.dtype]
        dev_ref = DeviceRef.GPU()
        embed_len = p.embed_seq_len

        # Create a fresh module so Weight objects are bound to this Graph.
        module = UMT5EncoderModel(
            self._umt5_config, dtype=max_dtype, device=dev_ref
        )
        # Generate weights on-demand; MAX's compile step copies them into
        # its own GPU allocation, so we don't need to cache them.
        torch_weights = _generate_random_weights(self._weight_shapes)
        module.load_state_dict(torch_weights, weight_alignment=1, strict=True)

        input_types = [
            TensorType(DType.int64, ["batch", "seq_len"], device=dev_ref),
            TensorType(DType.int64, ["batch", "seq_len"], device=dev_ref),
        ]

        with Graph("umt5_encoder", input_types=input_types) as graph:
            input_ids = graph.inputs[0].tensor
            attention_mask = graph.inputs[1].tensor
            hidden_states = module(input_ids, attention_mask)

            # On-device post-processing: slice + mask (mirrors TextEncoder).
            sliced = hidden_states[:, :embed_len, :]
            mask_sliced = ops.cast(
                attention_mask[:, :embed_len], hidden_states.dtype
            )
            result = sliced * ops.unsqueeze(mask_sliced, -1)
            graph.output(result)

        return graph, module.state_dict()

    def build_and_compile(self) -> CompiledLayerBundle:
        graph, weights_registry = self.build_graph()
        compiled = self.session.load(graph, weights_registry=weights_registry)
        # Drop local refs and run GC so the source torch tensors and any
        # transient GPU staging buffers are released before benchmarking.
        del graph
        del weights_registry
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CompiledLayerBundle(
            compiled_model=compiled,
            device=self.device,
            session=self.session,
        )

    def prepare_inputs(
        self,
        bundle: CompiledLayerBundle,
        dynamic_params: TextEncoderDynamicParams,
    ) -> tuple[list[Buffer], None]:
        p = self.static_params
        batch = dynamic_params.batch_size
        seq_len = dynamic_params.seq_len
        num_real = dynamic_params.num_real_tokens
        if num_real is None:
            num_real = seq_len

        # Generate random token IDs (1 to vocab_size-1 for real, 0 for pad).
        token_ids = np.zeros((batch, seq_len), dtype=np.int64)
        token_ids[:, :num_real] = np.random.randint(
            1, p.vocab_size, size=(batch, num_real)
        )

        # Attention mask: 1 for real tokens, 0 for padding.
        mask = np.zeros((batch, seq_len), dtype=np.int64)
        mask[:, :num_real] = 1

        token_buf = Buffer.from_numpy(token_ids).to(bundle.device)
        mask_buf = Buffer.from_numpy(mask).to(bundle.device)

        return [token_buf, mask_buf], None

    def cleanup_inputs(
        self, bundle: CompiledLayerBundle, context: None
    ) -> None:
        pass  # No resources to clean up.

    def cuda_graph_eligible(
        self, dynamic_params: TextEncoderDynamicParams
    ) -> bool:
        # The encoder has symbolic seq_len, so static shape is not guaranteed.
        # Disable CUDA graph capture for now.
        return False

    def torch_reference_layer(self, device: str = "cuda") -> torch.nn.Module:
        """Return an HF UMT5EncoderModel wrapped with post-processing."""
        p = self.static_params
        _, torch_dtype = DTYPE_MAP[p.dtype]

        hf_config = _make_hf_config(p)
        hf_model = HFUmt5EncoderModel(hf_config)

        # Generate matching weights on-demand (same seed as build_graph).
        torch_weights = _generate_random_weights(self._weight_shapes)
        hf_state_dict = _torch_to_hf_state_dict(torch_weights)
        hf_model.load_state_dict(hf_state_dict, strict=True)
        del torch_weights, hf_state_dict
        gc.collect()

        hf_model = hf_model.to(device=device, dtype=torch_dtype)
        hf_model.eval()

        return _UMT5WithPostProcess(hf_model, embed_seq_len=p.embed_seq_len)

    def prepare_torch_inputs(
        self,
        execute_args: list[Buffer],
        dynamic_params: TextEncoderDynamicParams,
        device: str = "cuda",
    ) -> list[torch.Tensor]:
        token_ids = torch.from_dlpack(execute_args[0]).to(device=device)
        attention_mask = torch.from_dlpack(execute_args[1]).to(device=device)
        return [token_ids, attention_mask]


class _UMT5WithPostProcess(torch.nn.Module):
    """Wraps HF UMT5EncoderModel with the same post-processing as our graph.

    Slices encoder output to embed_seq_len and masks padding positions.
    """

    def __init__(self, encoder: HFUmt5EncoderModel, embed_seq_len: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed_seq_len = embed_seq_len

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Post-processing: slice + mask (same as MAX graph).
        embed_len = self.embed_seq_len
        sliced = out[:, :embed_len, :]
        mask_sliced = attention_mask[:, :embed_len].to(sliced.dtype)
        return sliced * mask_sliced.unsqueeze(-1)
