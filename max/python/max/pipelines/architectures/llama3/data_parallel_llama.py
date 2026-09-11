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

import copy
from collections.abc import Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.data_parallelism import split_batch
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.layer import LayerList, Module

from .llama3 import Llama3
from .model_config import Llama3Config


# TODO: This could be into a helper that wraps any top-level Module into a
# data-parallel model.
class DataParallelLlama(Module):
    def __init__(self, config: Llama3Config):
        self.config = config
        self.devices = config.devices
        models = []
        for device in config.devices:
            # TODO: dataclasses.replace doesn't work for model config objects:
            # "... has no attribute 'draft_pipeline_parallel_degree'"
            new_config = copy.deepcopy(config)
            new_config.devices = [device]
            models.append(Llama3(new_config))

        # A LayerList, not a plain list, so weight tracking sees every
        # replica and each rank's weights land on that rank. Sharing rank
        # 0's weights instead makes every `weight.to(x.device)` a
        # peer-to-peer copy of the whole model, on every execution.
        # Don't also alias a replica to an attribute: every `Module`
        # attribute is registered, so rank 0 would get two sets of weights.
        self.models = LayerList(models)

    def __call__(
        self, all_model_args: Sequence[Sequence[Any]]
    ) -> tuple[TensorValue, ...]:
        all_outputs: list[list[TensorValue]] = [[] for _ in range(3)]
        for args, model in zip(all_model_args, self.models, strict=True):
            outputs = model(*args)
            for i, output in enumerate(outputs):
                all_outputs[i].append(output.to(self.devices[0]))
        if all_outputs[1] and all_outputs[2]:
            return tuple(ops.concat(output, axis=0) for output in all_outputs)
        else:
            return (ops.concat(all_outputs[0], axis=0),)

    # Graph helpers.
    def input_types(
        self,
        kv_params: KVCacheParamInterface,
    ) -> tuple[TensorType | BufferType, ...]:
        """Creates input tensor types used for building the graph.

        The input types defined in this function differ from the input types
        expected by `__call__`.

        A single device model expects the inputs:
        - tokens: Buffer of shape [total_seq_len]
        - input_row_offsets: Buffer of shape [batch_size + 1]
        - return_n_logits: Buffer of shape [1]
        - kv_cache_inputs: list of KV cache inputs.

        This class's `__call__` method expects the inputs above for each device.

        The input types defined here, however, are the same as the input types
        as the single device model except for:
        - an additional input for the data_parallel_splits tensor
        - the kv_cache_inputs replicated for each device.

        In `_call_flat`, the data_parallel_splits tensor is used to split the
        tokens and input_row_offsets into data parallel splits.
        """
        inputs = []
        first_model = self.models[0]
        # `LayerList.__getitem__` is typed as returning `Layer`.
        assert isinstance(first_model, Llama3)
        single_model_inputs = first_model.input_types(kv_params)
        (
            token_type,
            input_row_offsets_type,
            return_n_logits_type,
            *single_model_kv_cache_inputs,
        ) = single_model_inputs
        del single_model_kv_cache_inputs

        flat_kv_cache_inputs = kv_params.flattened_kv_inputs()

        data_parallel_split_type = TensorType(
            DType.int64,
            shape=[len(self.config.devices) + 1],
            device=DeviceRef.CPU(),
        )

        inputs = [
            token_type,
            input_row_offsets_type,
            return_n_logits_type,
            data_parallel_split_type,
            *flat_kv_cache_inputs,
        ]
        return tuple(inputs)

    def _call_flat(
        self, kv_params: KVCacheParamInterface, *args: Value[Any]
    ) -> tuple[TensorValue, ...]:
        (
            tokens,
            input_row_offsets,
            return_n_logits,
            data_parallel_splits,
            *all_kv_cache_inputs,
        ) = args

        split_tokens, split_offsets = split_batch(
            self.config.devices,
            tokens.tensor,
            input_row_offsets.tensor,
            data_parallel_splits.tensor,
        )

        all_model_args = []

        symbolic_inputs = kv_params.unflatten_kv_inputs(
            iter(all_kv_cache_inputs)
        )
        assert isinstance(symbolic_inputs, KVCacheInputs)
        kv_collections = symbolic_inputs.inputs

        for i in range(len(self.config.devices)):
            all_model_args.append(
                (
                    split_tokens[i].tensor,
                    kv_collections[i],
                    return_n_logits.tensor,
                    split_offsets[i].tensor,
                )
            )

        return self(all_model_args)


def create_graph(
    config: Llama3Config,
    kv_params: KVCacheParamInterface,
    state_dict: dict[str, Any],
) -> tuple[Graph, dict[str, Any]]:
    model = DataParallelLlama(config)

    state_dict.pop("rope_freqs.weight", None)
    # One prefix per replica. The copy is shallow (the weight buffer stays
    # shared) and is needed because `load_state_dict` rewrites the value's
    # dtype and shape in place.
    new_state_dict = {}
    for replica_idx in range(len(config.devices)):
        # Must match the attribute the replicas are registered under.
        prefix = f"models.{replica_idx}."
        for key, value in state_dict.items():
            new_state_dict[prefix + key] = copy.copy(value)
    model.load_state_dict(
        new_state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=True,
    )
    inputs = model.input_types(kv_params)
    with Graph("llama3", input_types=inputs) as graph:
        outputs = model._call_flat(kv_params, *graph.inputs)
        graph.output(*outputs)
        return graph, model.state_dict()
