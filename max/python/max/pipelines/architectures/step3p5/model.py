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

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from max._core.engine import Model
from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.comm.ep.ep_manager import EPBatchManager
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .batch_processor import Step3p5BatchProcessor
from .model_config import Step3p5Config
from .step3p5 import ParallelismMode, Step3p5

logger = logging.getLogger("max.pipelines")


@dataclass
class Step3p5Inputs(Llama3Inputs):
    """Inputs for Step-3.5 in TP+EP and DP+EP modes.

    Extends ``Llama3Inputs`` with optional ``host_input_row_offsets`` /
    ``data_parallel_splits`` (DP+EP only) and the EP communication buffers
    (TP+EP and DP+EP).
    """

    host_input_row_offsets: Buffer | None = None
    ep_inputs: tuple[Buffer, ...] = field(default_factory=tuple)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        base = [self.tokens, self.input_row_offsets, self.return_n_logits]
        if (
            self.host_input_row_offsets is not None
            and self.data_parallel_splits is not None
        ):
            # The DP_EP graph input for splits lives on CPU
            # (see Step3p5.input_types). prepare_initial_token_inputs is
            # the sole producer and always builds a CPU Buffer; reject
            # any other shape so we fail loudly instead of silently
            # putting splits on the wrong device.
            assert isinstance(self.data_parallel_splits, Buffer), (
                "Step3p5Inputs requires data_parallel_splits to be a CPU Buffer"
            )
            base.extend(
                [self.host_input_row_offsets, self.data_parallel_splits]
            )
        return (
            *base,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.ep_inputs,
        )


class Step3p5Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Step-3.5-Flash pipeline model.

    Supports single-GPU, multi-GPU TP, TP-attention + EP-MoE, and
    DP-attention + EP-MoE.
    """

    model_config_cls: ClassVar[type[Any]] = Step3p5Config
    batch_processor_cls: ClassVar[type[Step3p5BatchProcessor]] = (
        Step3p5BatchProcessor
    )

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    def _create_ep_config(
        self,
        state_dict: dict[str, Any] | None = None,
    ) -> EPConfig | None:
        """Create an :class:`EPConfig` from the pipeline settings.

        Returns ``None`` when EP is not requested (``ep_size <= 1``).
        """
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return None

        n_devices = len(self.devices)
        if ep_size % n_devices != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must be divisible by the number of "
                f"GPUs ({n_devices})."
            )

        config = self.huggingface_config
        n_nodes = ep_size // n_devices
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree

        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=data_parallel_degree,
        )

        encoding = _select_quantization_encoding(
            self.pipeline_config.model, Step3p5Config.DEFAULT_ENCODING
        )
        dispatch_dtype = supported_encoding_dtype(encoding)

        dispatch_quant_config = None
        if dispatch_dtype != DType.bfloat16 and state_dict is not None:
            dispatch_quant_config = parse_quant_config(
                config, state_dict, dispatch_dtype
            )

        # Read directly from the HuggingFace config so a missing field
        # fails loudly rather than silently defaulting to wrong shapes.
        n_experts = config.moe_num_experts
        top_k = config.moe_top_k
        hidden_size = config.hidden_size

        return EPConfig(
            dispatch_dtype=dispatch_dtype,
            combine_dtype=DType.bfloat16,
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=n_experts,
            max_tokens_per_rank=ep_max_rank_send_tokens,
            n_gpus_per_node=n_devices,
            n_nodes=n_nodes,
            dispatch_quant_config=dispatch_quant_config,
        )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Step3p5Config:
        model_config = Step3p5Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        self._ep_config = self._create_ep_config(state_dict)
        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Step3p5Config,
    ) -> None:
        del model_config
        # Set up EP comm infrastructure.
        self.ep_comm_initializer = None
        ep_config = self._ep_config
        if ep_config is None or is_virtual_device_mode():
            return
        self.ep_comm_initializer = EPCommInitializer(ep_config)
        self.ep_comm_initializer.ep_init(session)
        ep_config.node_id = self.ep_comm_initializer.config.node_id

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Step3p5Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        ep_config = self._ep_config
        ep_manager: EPBatchManager | None = None
        if ep_config is not None:
            ep_manager = EPBatchManager(ep_config)

        nn_model = Step3p5(model_config, ep_manager=ep_manager)
        # Cache the mode for input-prep (DP_EP adds host offsets +
        # splits; TP_EP and DP_EP append EP comm buffers at the tail).
        self._mode = nn_model.mode

        # DP_EP's logits postprocess only emits last-token logits and does
        # not produce hidden states. Reject the unsupported combinations
        # at compile time rather than letting execute() fail to unpack
        # the model outputs at runtime.
        if self._mode == ParallelismMode.DP_EP:
            if self.return_logits != ReturnLogits.LAST_TOKEN:
                raise ValueError(
                    "Step-3.5 DP+EP only supports return_logits=LAST_TOKEN; "
                    f"got {self.return_logits}."
                )
            if self.return_hidden_states != ReturnHiddenStates.NONE:
                raise ValueError(
                    "Step-3.5 DP+EP does not support returning hidden "
                    f"states; got return_hidden_states={self.return_hidden_states}."
                )

        logger.info(
            "Step-3.5: parallelism mode=%s, data_parallel_degree=%d, "
            "ep_size=%s.",
            self._mode.name,
            model_config.data_parallel_degree,
            self.pipeline_config.runtime.ep_size,
        )

        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        weights_registry = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph("step3p5", input_types=graph_inputs) as graph:
            inputs_iter = iter(graph.inputs)
            tokens = next(inputs_iter)
            input_row_offsets = next(inputs_iter)
            return_n_logits = next(inputs_iter)

            host_input_row_offsets = None
            data_parallel_splits = None
            if self._mode == ParallelismMode.DP_EP:
                host_input_row_offsets = next(inputs_iter)
                data_parallel_splits = next(inputs_iter)

            signal_buffers = [
                next(inputs_iter).buffer for _ in range(num_devices)
            ]

            kv_input_count = len(self.kv_params.flattened_kv_inputs())
            kv_cache_inputs = [next(inputs_iter) for _ in range(kv_input_count)]
            sliding_kv, global_kv = self.kv_params.unflatten_basic_kv_tree(
                iter(kv_cache_inputs)
            )

            # Tail of the input list is the EP comm buffers, present for
            # both TP_EP and DP_EP. Empty in TP_TP.
            ep_model_inputs = (
                list(inputs_iter) if ep_manager is not None else None
            )

            outputs = nn_model(
                tokens.tensor,
                sliding_kv,
                global_kv,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                host_input_row_offsets=(
                    host_input_row_offsets.tensor
                    if host_input_row_offsets is not None
                    else None
                ),
                data_parallel_splits=(
                    data_parallel_splits.tensor
                    if data_parallel_splits is not None
                    else None
                ),
                ep_inputs=ep_model_inputs,
            )

            graph.output(*outputs)
            return graph, weights_registry

    @override
    def _wire_batch_processor(
        self,
        model: Any = None,
        model_config: Any = None,
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        batch_processor = self.batch_processor
        if batch_processor is None:
            return
        bind_mode = getattr(batch_processor, "bind_parallelism_mode", None)
        if bind_mode is not None:
            bind_mode(self._mode)
