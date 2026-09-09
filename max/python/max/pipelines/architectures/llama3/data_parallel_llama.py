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
from max.nn.layer import Module

from .llama3 import Llama3
from .model_config import Llama3Config


# TODO: This could be into a helper that wraps any top-level Module into a
# data-parallel model.
class DataParallelLlama(Module):
    def __init__(self, config: Llama3Config):
        self.config = config
        self.devices = config.devices
        self.models = []
        for device in config.devices:
            # TODO: dataclasses.replace doesn't work for model config objects:
            # "... has no attribute 'draft_pipeline_parallel_degree'"
            new_config = copy.deepcopy(config)
            new_config.devices = [device]
            self.models.append(Llama3(new_config))

        # Sets up weight tracking for the first model.
        self.model = self.models[0]

        # Replace all weights from the other distributed models with weights
        # from the first model.
        model_weights = self.model.raw_state_dict()
        for model in self.models[1:]:
            for key, value in model_weights.items():
                _assign_weight(model, key, value)

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
        single_model_inputs = self.model.input_types(kv_params)
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


def _assign_weight(module: Module, key: str, value: Any) -> None:
    path = key.split(".")
    for attr in path[:-1]:
        if attr.isnumeric():
            module = module[int(attr)]  # type: ignore
        else:
            module = _resolve_attr(module, attr)
    # The leaf segment is set on whichever container actually owns it,
    # which may be a (possibly nested) name-omitting descendant of
    # ``module``.
    leaf = path[-1]
    setattr(_owner_of(module, leaf), leaf, value)


def _resolve_attr(module: Module, attr: str) -> Module:
    """Resolve ``attr`` on ``module``, descending through name-omitting children.

    State-dict FQNs flatten across modules with
    ``_omit_module_attr_name`` (notably :class:`~max.nn.StackedLinear` in
    unfused mode), so a key segment may live one or more levels deeper
    than the attribute hierarchy suggests. Direct ``getattr`` is tried
    first; on miss we recurse into each name-omitting child until one
    exposes ``attr``.
    """
    if hasattr(module, attr):
        return getattr(module, attr)
    for child in module.sublayers.values():
        if not child._omit_module_attr_name:
            continue
        try:
            return _resolve_attr(child, attr)
        except AttributeError:
            continue
    raise AttributeError(
        f"{type(module).__name__!s} has no attribute {attr!r} (also "
        "checked name-omitting descendants)."
    )


def _owner_of(module: Module, attr: str) -> Module:
    """Return the (possibly nested name-omitting) submodule that owns ``attr``.

    Mirrors :func:`_resolve_attr` but returns the *owning module* rather
    than the resolved attribute itself. Used to set a flattened leaf
    weight on the actual child whose namespace was elided. If no
    descendant already owns ``attr``, falls back to ``module`` so the
    caller's ``setattr`` preserves the original (attribute-creating)
    behavior for new weights.
    """
    if hasattr(module, attr):
        return module
    for child in module.sublayers.values():
        if not child._omit_module_attr_name:
            continue
        owner = _owner_of(child, attr)
        if hasattr(owner, attr):
            return owner
    return module


def create_graph(
    config: Llama3Config,
    kv_params: KVCacheParamInterface,
    state_dict: dict[str, Any],
) -> tuple[Graph, dict[str, Any]]:
    model = DataParallelLlama(config)

    new_state_dict = {}
    state_dict.pop("rope_freqs.weight", None)
    for key, value in state_dict.items():
        new_key = "model." + key
        new_state_dict[new_key] = value
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
