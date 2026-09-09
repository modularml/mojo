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
"""Serving for adapters-as-inputs LoRA on ModuleV3 models.

A model whose projections were wrapped with :class:`~max.experimental.nn.LoRA`
(see :meth:`LoRAManagerV3.wrap`) takes the adapters and routing as extra graph inputs
rather than mutable weights. :class:`LoRAManagerV3` loads the adapters
unfused (per-projection q/k/v), produces the batch routing, declares the extra
compile inputs (:meth:`~LoRAManagerV3.symbolic_inputs`), and distributes them
during tracing (:meth:`~LoRAManagerV3.bind_inputs`), so intermediate layers keep
their plain ``forward(x)`` signature.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

_CtxT = TypeVar("_CtxT")

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import LoRA, lora_layers, lora_parameters
from max.experimental.nn.module import Module
from max.experimental.tensor import Tensor, TensorType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsAdapter, WeightsFormat
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
)

from .config import LoRAConfig
from .lora import ADAPTER_CONFIG_FILE, LoRAModel, _LoRALRUCache
from .lora_types import LoRAStatus

# ModuleV3 LoRA routing is three tensors, matching
# ``max.experimental.nn.LoRA.set_lora_batch_info``.
NUM_ROUTING_INPUTS = 3


class _UnfusedLoRAModel(LoRAModel):
    """A :class:`LoRAModel` that keeps its per-projection q/k/v adapter keys.

    ModuleV3 LoRA applies a separate adapter per q/k/v projection, so the V2
    fused-QKV combine is skipped and the unfused keys survive for the buffers.
    """

    def _combine_qkv_weights(self) -> None:
        return


def _download_adapter_repo(repo_id: str) -> str:
    """Downloads an online LoRA adapter repo to a local snapshot and returns its dir.

    Online adapters are materialized on local disk before loading because
    :class:`_UnfusedLoRAModel` reads ``adapter_config.json`` and the safetensors
    weights directly off disk. Only those files are fetched, reusing MAX's
    auth/gating/retry-aware Hugging Face helpers.

    Raises:
        ValueError: If ``repo_id`` is not an accessible Hugging Face repo.
        FileNotFoundError: If ``HF_HUB_OFFLINE`` is set and the repo is uncached.
    """
    repo = HuggingFaceRepo(repo_id=repo_id)  # validates access for online repos
    if repo.repo_type == "local":
        # Either a user-provided directory or a fully cached snapshot dir
        # resolved under HF_HUB_OFFLINE.
        return repo.local_path
    safetensors = repo.weight_files.get(WeightsFormat.safetensors, [])
    files = download_weight_files(
        huggingface_model_id=repo_id,
        filenames=[ADAPTER_CONFIG_FILE, *safetensors],
        revision=repo.revision,
    )
    # adapter_config.json sits at the snapshot root, so its parent is the dir the
    # loader reads the config + weights from.
    return str(files[0].parent)


class LoRAManagerV3:
    """Loads and routes ModuleV3 LoRA adapters (adapters-as-inputs).

    It owns an LRU slot cache and loads adapters unfused (per-projection q/k/v),
    then supplies the batch routing and slot->adapter map that the serving path
    feeds as graph inputs. Adapters ride in as inputs, so there is no
    alias-buffer hot-swap.
    """

    # -1 marks a base-model request; downstream kernels exit early on it.
    _NO_ACTIVE_LORA = -1

    def __init__(
        self,
        config: LoRAConfig,
        base_model_path: str,
        base_dtype: DType,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_lora_seq_len: int,
        targets: Sequence[LoRATargetModule] = (),
    ) -> None:
        self.base_model_path = base_model_path
        self.base_dtype = base_dtype
        self.max_num_loras = config.max_num_loras
        self.max_lora_rank = config.max_lora_rank
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_lora_seq_len = max_lora_seq_len
        # The arch's LoRA targets and the wrapped model are owned here so the
        # serving path (buffers, symbolic inputs) needs no external state.
        self._targets = tuple(targets)
        self._model: Module[..., Any] | None = None
        self._loras: dict[str, LoRAModel] = {}
        self._active_loras = _LoRALRUCache(max_size=self.max_num_loras)
        # Device-resident per-slot adapter buffers, rebuilt only when the
        # slot->adapter assignment changes (keyed by that signature + device).
        self._buffer_cache: (
            tuple[tuple[tuple[int, str], ...], Device, list[Buffer]] | None
        ) = None
        if config.lora_paths:
            self._load_adapters(config.lora_paths)

    @property
    def loras(self) -> list[str]:
        """The loaded adapter names."""
        return list(self._loras.keys())

    def is_lora(self, name: str) -> bool:
        """Whether ``name`` is a loaded adapter."""
        return name in self._loras

    def is_active_lora(self, name: str) -> bool:
        """Whether ``name`` is an active (LRU-slotted) adapter.

        The serving scheduler calls this to decide whether a request can reuse
        an already-active slot or needs to claim a new one.
        """
        return name in self._active_loras

    def unload_adapter(self, name: str) -> LoRAStatus:
        """Unloads ``name`` from the registry and frees its LRU slot.

        The serving request processor calls this to release a loaded adapter.
        """
        if name not in self._loras:
            return LoRAStatus.UNLOAD_NAME_NONEXISTENT
        del self._loras[name]
        self._active_loras.remove(name)
        return LoRAStatus.SUCCESS

    def load_adapter(self, path: str) -> LoRAStatus:
        """Loads one adapter (``name=path`` or ``path``), keeping it unfused.

        A ``path`` that is not a local directory is treated as a Hugging Face
        repo id and downloaded to a local snapshot before loading.
        """
        name, _, explicit = path.partition("=")
        if explicit:
            path = explicit
        else:
            name = path
        if name in self._loras:
            return (
                LoRAStatus.SUCCESS
                if self._loras[name].path == path
                else LoRAStatus.LOAD_NAME_EXISTS
            )
        local_path = self._resolve_adapter_path(path)
        if local_path is None:
            return LoRAStatus.LOAD_INVALID_PATH
        try:
            self._loras[name] = _UnfusedLoRAModel(
                name,
                local_path,
                self.base_dtype,
                self.max_lora_rank,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
            )
        except ValueError:
            return LoRAStatus.LOAD_INVALID_ADAPTER
        return LoRAStatus.SUCCESS

    def _resolve_adapter_path(self, path: str) -> str | None:
        """Resolves an adapter spec to a local directory, or ``None``.

        A local directory passes through unchanged; otherwise ``path`` is treated
        as a Hugging Face repo id and its adapter files are downloaded to a local
        snapshot (see :func:`_download_adapter_repo`). Returns ``None`` when
        ``path`` is neither a local directory nor a resolvable Hugging Face repo.
        """
        if os.path.isdir(path):
            return path
        try:
            return _download_adapter_repo(path)
        except (ValueError, FileNotFoundError):
            return None

    def _load_adapters(self, lora_paths: Sequence[str]) -> None:
        for path in lora_paths:
            status = self.load_adapter(path)
            if status != LoRAStatus.SUCCESS:
                raise RuntimeError(
                    f"Failed to load LoRA adapter '{path}': {status}"
                )

    def activate_adapter(self, name: str) -> None:
        """Marks an adapter active, assigning it an LRU slot."""
        if name not in self._loras:
            raise KeyError(f"LoRA adapter '{name}' not found in registry")
        self._active_loras.put(name, self._loras[name])

    def _model_name_to_id(self, name: str | None) -> int:
        if not name or name == self.base_model_path:
            return self._NO_ACTIVE_LORA
        slot = (
            self._active_loras.get_slot(name) if name in self._loras else None
        )
        return slot if slot is not None else self._NO_ACTIVE_LORA

    def sort_lora_batch(self, context_batch: list[_CtxT]) -> list[_CtxT]:
        """Sorts a batch so same-adapter requests are adjacent, base last.

        :meth:`get_lora_graph_inputs` groups consecutive same-adapter requests
        and truncates at the first base group, so the batch must be ordered by
        adapter id (base id ``-1`` sorts last) for the ragged routing to cover
        every LoRA request.
        """
        return sorted(
            context_batch,
            key=lambda item: self._model_name_to_id(
                getattr(item, "model_name", None)
            ),
            reverse=True,
        )

    def active_adapters_by_slot(
        self, context_batch: Sequence[Any]
    ) -> dict[int, LoRAModel]:
        """Maps each batch adapter to its active LRU slot id.

        Base requests and inactive adapters are skipped.
        """
        by_slot: dict[int, LoRAModel] = {}
        for ctx in context_batch:
            name = getattr(ctx, "model_name", None)
            if (
                not name
                or name == self.base_model_path
                or name not in self._loras
            ):
                continue
            slot = self._active_loras.get_slot(name)
            if slot is not None:
                by_slot[slot] = self._loras[name]
        return by_slot

    def get_lora_graph_inputs(
        self,
        context_batch: Sequence[Any],
        input_row_offsets: npt.NDArray[np.integer[Any]],
        device: Device,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """The batch routing triple ``(lora_ids, grouped_offsets, end)``.

        Groups consecutive same-adapter requests and truncates at the first
        base-model group. ``end`` carries the LoRA-routed token count that the
        boundary-aware expand kernel reads (see ``LoRA.set_lora_batch_info``);
        ``lora_ids``/``grouped_offsets`` live on ``device`` and ``end`` on CPU,
        matching :meth:`symbolic_inputs`.
        """
        ids = []
        for ctx in context_batch:
            name = getattr(ctx, "model_name", None)
            if (
                name
                and name != self.base_model_path
                and name not in self._loras
            ):
                raise RuntimeError(
                    f"Request references unknown LoRA {name!r}; loaded: "
                    f"{list(self._loras)}."
                )
            ids.append(self._model_name_to_id(name))

        grouped_ids: list[int] = []
        grouped_offsets: list[Any] = []
        prev_id: int | None = None
        for i, id_ in enumerate(ids):
            if id_ != prev_id:
                grouped_ids.append(id_)
                grouped_offsets.append(input_row_offsets[i])
                prev_id = id_
        grouped_offsets.append(input_row_offsets[-1])
        last_lora = (
            grouped_ids.index(-1) if -1 in grouped_ids else len(grouped_ids)
        )
        grouped_offsets = grouped_offsets[: last_lora + 1]
        grouped_ids = grouped_ids[:last_lora]

        end_idx = int(grouped_offsets[-1]) if grouped_offsets else 0
        end = np.zeros(end_idx, dtype=np.int64)
        if end_idx:
            end[0] = end_idx

        lora_ids = Buffer.from_numpy(np.array(grouped_ids, dtype=np.int32)).to(
            device
        )
        offsets = Buffer.from_numpy(
            np.array(grouped_offsets, dtype=np.uint32)
        ).to(device)
        return lora_ids, offsets, Buffer.from_numpy(end)

    def input_buffers(
        self,
        context_batch: Sequence[Any],
        input_row_offsets: npt.NDArray[Any],
        device: Device,
    ) -> tuple[Buffer, ...]:
        """Builds the per-call LoRA input tuple.

        The batch routing triple then the per-slot adapter buffers, in
        :meth:`symbolic_inputs` order.
        """
        adapters = self.active_adapters_by_slot(context_batch)
        adapter_buffers = self._adapter_buffers(adapters, device=device)
        if adapters:
            routing = self.get_lora_graph_inputs(
                context_batch, input_row_offsets, device
            )
        else:
            # All-base batch: get_lora_graph_inputs yields empty routing (it
            # strips the base group), which the SGMV kernels can't process. The
            # adapter buffers are all-zero, so route every token through slot 0
            # as a no-op (delta == 0).
            routing = self._base_only_routing(
                int(input_row_offsets[-1]), device
            )
        return (*routing, *adapter_buffers)

    def _adapter_buffers(
        self,
        adapters: Mapping[int, LoRAModel],
        *,
        device: Device,
    ) -> list[Buffer]:
        """Per-slot ``[max_num_loras, ...]`` adapter buffers, cached device-resident.

        Packing stacks and host-to-device-copies the whole adapter pool -- eager
        work that must not run per decode step -- so the result is cached and
        reused until the slot->adapter assignment changes.
        """
        signature = self._adapter_signature(adapters)
        cache = self._buffer_cache
        if cache is not None and cache[0] == signature and cache[1] is device:
            return cache[2]
        buffers = self._pack_adapter_buffers(adapters, device=device)
        self._buffer_cache = (signature, device, buffers)
        return buffers

    @staticmethod
    def _adapter_signature(
        adapters: Mapping[int, LoRAModel],
    ) -> tuple[tuple[int, str], ...]:
        """Hashable key for the slot->adapter assignment (its only dependency)."""
        return tuple(sorted((sid, m.name) for sid, m in adapters.items()))

    def _pack_adapter_buffers(
        self,
        adapters: Mapping[int, LoRAModel],
        *,
        device: Device,
    ) -> list[Buffer]:
        assert self._model is not None, (
            "wrap() must run before packing buffers."
        )
        buffers: list[Buffer] = []
        for name, slot in lora_parameters(self._model):
            max_loras = int(slot.type.shape[0])
            keys = _slot_peft_keys(name, self._targets)
            # A fused qkv slot is the axis-0 concatenation of its per-projection
            # PEFT weights (``part_shapes``); a single slot is one weight.
            part_shapes = slot.part_shapes or [
                [int(d) for d in slot.type.shape[1:]]
            ]
            rows: list[Tensor] = []
            for lora_id in range(max_loras):
                lora = adapters.get(lora_id)
                parts: list[Tensor] = []
                for key, shape in zip(keys, part_shapes, strict=True):
                    weight = lora.get(key) if lora is not None else None
                    if weight is not None:
                        parts.append(Tensor.from_dlpack(weight.data))
                    else:
                        parts.append(
                            Tensor.zeros(
                                shape, dtype=slot.type.dtype, device=CPU()
                            )
                        )
                rows.append(
                    parts[0] if len(parts) == 1 else F.concat(parts, axis=0)
                )
            stacked = F.stack(rows, axis=0)  # [max_num_loras, *inner]
            buffers.append(stacked.to(device).driver_tensor)
        return buffers

    @staticmethod
    def _base_only_routing(
        total_tokens: int, device: Device
    ) -> tuple[Buffer, Buffer, Buffer]:
        """No-op routing (all tokens -> slot 0) for a batch with no adapters.

        Matches the get_lora_graph_inputs layout for a single group.
        """
        ids = Buffer.from_numpy(np.array([0], dtype=np.int32)).to(device)
        offsets = Buffer.from_numpy(
            np.array([0, total_tokens], dtype=np.uint32)
        ).to(device)
        end = np.zeros(total_tokens, dtype=np.int64)
        if total_tokens:
            end[0] = total_tokens
        return ids, offsets, Buffer.from_numpy(end)

    def symbolic_inputs(self, device_ref: DeviceRef) -> list[TensorType]:
        """Returns the extra compile-input types for a ModuleV3 LoRA model.

        The routing triple first, then one adapter-stack type per slot in
        :func:`~max.experimental.nn.lora_parameters` order -- the same order fed
        at runtime and consumed by :meth:`bind_inputs` during tracing.
        """
        assert self._model is not None, (
            "wrap() must run before symbolic_inputs."
        )
        routing = [
            TensorType(DType.int32, shape=["lora_ids"], device=device_ref),
            TensorType(
                DType.uint32, shape=["lora_grouped_offsets"], device=device_ref
            ),
            TensorType(
                DType.int64, shape=["lora_end_idx"], device=DeviceRef.CPU()
            ),
        ]
        adapters = [slot.type for _, slot in lora_parameters(self._model)]
        return routing + adapters

    def bind_inputs(
        self, model: Module[..., Any], lora_inputs: Sequence[Tensor]
    ) -> None:
        """Distributes the LoRA graph inputs into the model's :class:`LoRA` layers.

        Called inside the model's ``forward`` during tracing: the batch-wide
        routing goes to every layer and each adapter stack to its slot, in the
        order :meth:`symbolic_inputs` produced.

        Raises:
            ValueError: If the input count does not match the routing triple
                plus the discovered adapter slots.
        """
        slots = list(lora_parameters(model))
        expected = NUM_ROUTING_INPUTS + len(slots)
        if len(lora_inputs) != expected:
            raise ValueError(
                f"Expected {expected} LoRA inputs (routing + {len(slots)} "
                f"adapter slots), got {len(lora_inputs)}."
            )
        lora_ids, grouped_offsets, end_idx = lora_inputs[:NUM_ROUTING_INPUTS]
        adapters = lora_inputs[NUM_ROUTING_INPUTS:]
        for _, layer in lora_layers(model):
            layer.set_lora_batch_info(lora_ids, grouped_offsets, end_idx)
        for value, (_, slot) in zip(adapters, slots, strict=True):
            slot.set(value)

    def wrap(self, model: Module[..., Any]) -> Module[..., Any]:
        """Makes ``model`` ModuleV3 LoRA ready and returns it.

        Wraps each of this manager's target projections in place with
        :class:`~max.experimental.nn.LoRA`, then returns (and remembers) a
        top-level module that fans the per-call LoRA inputs into those layers, so
        the inner model's ``forward`` stays LoRA-oblivious. The returned wrapper
        holds ``model`` at attribute ``model`` (adding a ``model.`` prefix that
        :meth:`lora_weight_adapter` compensates for).
        """
        self._apply_lora(model)
        self._model = _LoRAFanoutModel(model, self)
        return self._model

    def lora_weight_adapter(
        self, base_adapter: WeightsAdapter, *, wrapper_prefix: str = "model"
    ) -> WeightsAdapter:
        """Wraps a format weight adapter to emit the wrapped LoRA layout.

        Runs ``base_adapter`` (the format's normal conversion), rewrites this
        manager's target projections via :func:`_fuse_projections_for_lora`
        (concatenating base q/k/v only for ``stacked`` targets; name-transparent
        targets keep their native per-projection keys), then prefixes each key
        with ``wrapper_prefix`` -- the attribute under which the fanout wrapper
        nests the inner model (``"model"`` by convention). This lets the base
        ``_load_state_dict`` produce the LoRA-ready state dict, so an arch needs
        no ``_load_state_dict`` override.
        """

        def adapter(state_dict: Any, **kwargs: Any) -> dict[str, WeightData]:
            fused = _fuse_projections_for_lora(
                base_adapter(state_dict, **kwargs), self._targets
            )
            return {f"{wrapper_prefix}.{k}": v for k, v in fused.items()}

        return adapter

    def _apply_lora(self, model: Module[..., Any]) -> int:
        """Wraps each target's leaf projection with :class:`~max.experimental.nn.LoRA` in place.

        Locates each target's container by its within-layer parent path and
        wraps the leaf on every matching container, using this manager's slot
        count / rank / seq-len. Returns the number of containers wrapped.

        Raises:
            ValueError: If a matched container lacks a target leaf, or if no
                container matched any target.
        """
        leaves_by_parent: dict[str, list[str]] = {}
        for target in self._targets:
            parent, _, leaf = target.path.rpartition(".")
            leaves_by_parent.setdefault(parent, []).append(leaf)

        def wrap_leaf(module: Module[..., Any]) -> LoRA:
            return LoRA(
                module,
                max_num_loras=self.max_num_loras,
                max_lora_rank=self.max_lora_rank,
                max_lora_seq_len=self.max_lora_seq_len,
            )

        wrapped = 0
        for name, module in [("", model), *model.descendants]:
            for parent, leaves in leaves_by_parent.items():
                if name == parent or name.endswith(f".{parent}"):
                    for leaf in leaves:
                        if not hasattr(module, leaf):
                            raise ValueError(
                                f"LoRA target container {name!r} has no "
                                f"callable {leaf!r} to wrap; is the projection "
                                "built as a sub-module?"
                            )
                        setattr(module, leaf, wrap_leaf(getattr(module, leaf)))
                    wrapped += 1
        if wrapped == 0:
            raise ValueError(
                "wrap matched no container for targets "
                f"{sorted(leaves_by_parent)}."
            )
        return wrapped


class _LoRAFanoutModel(Module[..., Tensor]):
    """Top-level wrapper that fans LoRA graph inputs into the wrapped model.

    Its ``forward`` peels the trailing LoRA inputs off the model's variadic
    inputs, binds them into the ``LoRA`` layers, then delegates to the inner
    model's plain ``forward`` -- so no arch ``forward`` needs LoRA plumbing. The
    inner model sits at attribute ``model``.
    """

    def __init__(self, model: Module[..., Any], manager: LoRAManagerV3) -> None:
        self.model = model
        self._manager = manager

    def forward(self, *args: Tensor) -> Tensor:
        num_lora = NUM_ROUTING_INPUTS + len(list(lora_parameters(self.model)))
        inner_args, lora_inputs = args[:-num_lora], args[-num_lora:]
        self._manager.bind_inputs(self.model, lora_inputs)
        return self.model(*inner_args)


@dataclass(frozen=True)
class LoRATargetModule:
    """One wrappable projection module and the PEFT projections it fuses.

    ``path`` is the module's location within a decoder layer (e.g.
    ``self_attn.qkv``); ``projections`` are the PEFT projection names it covers,
    in fused order -- ``("q_proj", "k_proj", "v_proj")`` for a fused qkv, or a
    single name like ``("o_proj",)``. A per-arch tuple of these is the single
    source of truth that drives adapter wrapping, base-weight fusion, and the
    slot->PEFT-key map, so a new model declares its targets rather than
    hand-writing three matching transforms.

    ``stacked`` marks a wrapped module that stores one pre-fused base weight (a
    ``stacked=True``
    :class:`~max.experimental.nn.common_layers.linear.QKVLinear`), so its base
    q/k/v weights are concatenated into that single weight. The default
    (``False``) is a name-transparent module that keeps separate per-projection
    children, whose base weights stay under their native checkpoint names.
    Either way the *adapter* is fused into one slot; only the *base* layout
    differs.
    """

    path: str
    projections: tuple[str, ...]
    stacked: bool = False


# A adapter slot name ends in
# ``<block-namespace>.<n>.<within-layer path>.lora_<a|b>``; the namespace + index
# locate the layer, the within-layer path selects the target, ``a|b`` the
# shrink/expand matrix. The namespace is arch-specific (``layers`` for Llama3,
# ``transformer_blocks`` / ``single_transformer_blocks`` for FLUX.2), so the
# pattern is built from the arch's ``layer_prefixes`` rather than hardcoded.
DEFAULT_LAYER_PREFIXES: tuple[str, ...] = ("layers",)


@functools.cache
def _slot_re(layer_prefixes: tuple[str, ...]) -> re.Pattern[str]:
    r"""Compiles the slot-name pattern for a set of block namespaces.

    Longest namespace first so ``single_transformer_blocks`` wins over
    ``transformer_blocks``; the leading ``(?:^|\\.)`` anchors the namespace at a
    path boundary so the shorter name cannot match inside the longer one.
    """
    alt = "|".join(
        re.escape(p) for p in sorted(layer_prefixes, key=len, reverse=True)
    )
    return re.compile(rf"(?:^|\.)({alt})\.(\d+)\.(.+)\.lora_(a|b)$")


def _slot_peft_keys(
    slot_name: str,
    targets: Sequence[LoRATargetModule],
    *,
    layer_prefixes: Sequence[str] = DEFAULT_LAYER_PREFIXES,
) -> list[str]:
    """Maps a adapter slot to its constituent PEFT keys.

    Driven by ``targets``: the slot's within-layer path selects the target, and
    the slot resolves to one normalized PEFT key per projection the target
    covers -- one for a single-projection target, three (q, k, v) for a fused
    qkv target, in the fused A/B stacking order. With the Llama targets,
    ``...layers.3.self_attn.qkv.lora_a`` ->
    ``[layers.3.self_attn.{q,k,v}_proj.lora_A.weight]`` and
    ``...layers.3.self_attn.o_proj.lora_b`` ->
    ``[layers.3.self_attn.o_proj.lora_B.weight]``. The emitted key reuses the
    namespace matched in the slot (e.g. ``single_transformer_blocks.5`` for
    FLUX.2), so ``layer_prefixes`` selects the arch's block namespaces.

    Args:
        slot_name: The qualified slot name from :func:`lora_parameters`.
        targets: The arch's LoRA target modules.
        layer_prefixes: The arch's block namespaces (``("layers",)`` for Llama3).

    Raises:
        ValueError: If the slot name is unrecognized or matches no target.
    """
    match = _slot_re(tuple(layer_prefixes)).search(slot_name)
    if match is None:
        raise ValueError(f"Unrecognized ModuleV3 LoRA slot: {slot_name!r}")
    prefix, layer, within, ab = match.groups()
    kind = "lora_A" if ab == "a" else "lora_B"
    for target in targets:
        if within == target.path:
            if len(target.projections) == 1:
                # Single-projection target: the wrapped module's own path is the
                # PEFT projection path. Using the path directly (rather than a
                # parent + leaf reconstruction) handles multi-component leaves
                # like ``attn.to_out.0`` (a one-element ModuleList element).
                return [f"{prefix}.{layer}.{target.path}.{kind}.weight"]
            parent = (
                target.path.rsplit(".", 1)[0]
                if "." in target.path
                else target.path
            )
            return [
                f"{prefix}.{layer}.{parent}.{proj}.{kind}.weight"
                for proj in target.projections
            ]
    raise ValueError(
        f"ModuleV3 LoRA slot {slot_name!r} (within-layer path {within!r}) "
        f"matches no LoRA target."
    )


def _fuse_projections_for_lora(
    state_dict: Mapping[str, WeightData],
    targets: Sequence[LoRATargetModule],
) -> dict[str, WeightData]:
    """Rewrites unfused projection weights into the LoRA-wrapped layout.

    For a ``stacked`` target, the checkpoint's per-projection weights (siblings
    under the target's parent path) concatenate on the output axis into the
    target's single pre-fused weight. A name-transparent target (the default,
    e.g. a ``stacked=False``
    :class:`~max.experimental.nn.common_layers.linear.QKVLinear`) keeps its
    separate per-projection children, so its base weights pass through unchanged
    under their native names -- concatenating them would drop the ``q_proj`` /
    ``k_proj`` / ``v_proj`` keys the module still expects.
    :class:`~max.experimental.nn.LoRA` is name-transparent, so a rewritten base
    weight keeps the target's native path (no ``.module`` infix). A
    single-projection target is renamed, not concatenated. Bias tensors are
    handled the same way; all other keys pass through unchanged.

    Args:
        state_dict: The adapted (already MAX-named) base checkpoint.
        targets: The arch's LoRA target modules.

    Returns:
        A new state dict keyed for the LoRA-wrapped model.
    """
    specs = []
    for target in targets:
        # A name-transparent multi-projection target keeps its per-projection
        # base weights separate under native names, so it needs no rewrite;
        # only a stacked (pre-fused) module concatenates them. A
        # single-projection target may still be renamed (e.g. FLUX.2's
        # ``attn.to_out.0``), so it always gets a spec.
        if len(target.projections) > 1 and not target.stacked:
            continue
        parent = (
            target.path.rsplit(".", 1)[0] if "." in target.path else target.path
        )
        trigger = f".{parent}.{target.projections[0]}."
        consumed = [f".{parent}.{p}." for p in target.projections[1:]]
        specs.append((target, parent, trigger, consumed))

    out: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        handled = False
        for target, parent, trigger, consumed in specs:
            if trigger in key:
                base, suffix = key.split(trigger, 1)  # suffix in {weight, bias}
                fused_key = f"{base}.{target.path}.{suffix}"
                if len(target.projections) == 1:
                    out[fused_key] = value
                else:
                    parts = [
                        Tensor.from_dlpack(
                            state_dict[f"{base}.{parent}.{p}.{suffix}"]
                        )
                        for p in target.projections
                    ]
                    fused = F.concat(parts, axis=0)
                    out[fused_key] = WeightData(
                        fused, fused_key, fused.dtype, fused.shape
                    )
                handled = True
                break
            if any(marker in key for marker in consumed):
                handled = True  # folded into the fused key above
                break
        if not handled:
            out[key] = value
    return out
