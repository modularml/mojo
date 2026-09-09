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
"""Tests for the standalone :class:`LoRAManagerV3` routing + slotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.nn import (
    Linear,
    LoRA,
    Module,
    ModuleList,
    TransparentModule,
    lora_layers,
)
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.tensor import Tensor, default_dtype
from max.graph.weights import WeightData
from max.pipelines.lora import (
    LoRAConfig,
    LoRAManagerV3,
    LoRATargetModule,
    modulev3,
)
from max.pipelines.lora.lora import LoRAModel
from max.pipelines.lora.lora_types import LoRAStatus

BASE = "base-model"

# bf16 has no CPU cast in the eager interpreter, so it would cold-compile a
# graph per weight; these tests only read parameter names.
_WEIGHT_DTYPE = DType.float32


class _StubLoRA(LoRAModel):
    """A LoRAModel that skips safetensors loading, for manager-only tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = name
        self.rank = 8


class _RecordingModel(LoRAModel):
    """A LoRAModel stub that records its resolved path without touching disk."""

    def __init__(self, name: str, path: str, *args: object) -> None:
        self.name = name
        self.path = path
        self.rank = 8


class _Ctx:
    def __init__(self, model_name: str | None) -> None:
        self.model_name = model_name


def _manager(
    max_num_loras: int = 2,
    targets: Sequence[LoRATargetModule] = (),
) -> LoRAManagerV3:
    config = LoRAConfig(max_num_loras=max_num_loras, max_lora_rank=8)
    mgr = LoRAManagerV3(
        config, BASE, DType.bfloat16, 4, 2, 16, 128, targets=targets
    )
    for name in ("adapterA", "adapterB"):
        mgr._loras[name] = _StubLoRA(name)
        mgr.activate_adapter(name)
    return mgr


def _np(buf: Buffer) -> np.ndarray:
    return np.from_dlpack(buf.to(CPU()))


def test_activate_assigns_distinct_slots() -> None:
    mgr = _manager()
    assert mgr._model_name_to_id("adapterA") == 0
    assert mgr._model_name_to_id("adapterB") == 1
    assert mgr._model_name_to_id(BASE) == -1
    assert mgr._model_name_to_id(None) == -1


def test_active_adapters_by_slot_skips_base_and_unknown() -> None:
    mgr = _manager()
    batch = [_Ctx("adapterB"), _Ctx(BASE), _Ctx("adapterA"), _Ctx(None)]
    by_slot = mgr.active_adapters_by_slot(batch)
    assert set(by_slot) == {0, 1}
    assert by_slot[0].name == "adapterA"
    assert by_slot[1].name == "adapterB"


def test_is_active_lora_tracks_activated_adapters() -> None:
    # The serving scheduler gates requests on is_active_lora; base and unknown
    # names are never active.
    mgr = _manager()
    assert mgr.is_active_lora("adapterA")
    assert mgr.is_active_lora("adapterB")
    assert not mgr.is_active_lora(BASE)
    assert not mgr.is_active_lora("unknown")


def test_unload_adapter_removes_from_registry_and_cache() -> None:
    mgr = _manager()
    assert mgr.is_lora("adapterA")
    assert mgr.is_active_lora("adapterA")
    mgr.unload_adapter("adapterA")
    assert not mgr.is_lora("adapterA")
    assert not mgr.is_active_lora("adapterA")
    mgr.unload_adapter("nope")
    assert not mgr.is_lora("nope")


def test_routing_groups_consecutive_adapters_and_truncates_base() -> None:
    mgr = _manager()
    batch = [_Ctx("adapterA"), _Ctx("adapterA"), _Ctx("adapterB"), _Ctx(BASE)]
    offsets = np.array([0, 2, 4, 6, 7], dtype=np.uint32)
    lora_ids, grouped_offsets, end = mgr.get_lora_graph_inputs(
        batch, offsets, CPU()
    )
    # Runs of the same adapter collapse; the trailing base group is dropped, so
    # only the two LoRA groups (slots 0, 1) and their boundaries remain.
    assert list(_np(lora_ids)) == [0, 1]
    assert list(_np(grouped_offsets)) == [0, 4, 6]
    end_np = _np(end)
    assert end_np.shape == (6,)  # LoRA-routed token count carried in end[0].
    assert int(end_np[0]) == 6


def test_unknown_adapter_raises() -> None:
    mgr = _manager()
    batch = [_Ctx("nope")]
    offsets = np.array([0, 3], dtype=np.uint32)
    with pytest.raises(RuntimeError, match="unknown LoRA"):
        mgr.get_lora_graph_inputs(batch, offsets, CPU())


def test_load_adapter_downloads_remote_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-local spec is resolved as an HF repo, downloaded, then loaded."""
    mgr = _manager()
    snapshot = "/snapshots/org__adapter"
    monkeypatch.setattr(
        modulev3, "_download_adapter_repo", lambda repo_id: snapshot
    )
    monkeypatch.setattr(modulev3, "_UnfusedLoRAModel", _RecordingModel)

    status = mgr.load_adapter("remote=org/adapter")

    assert status == LoRAStatus.SUCCESS
    # The adapter loads from the downloaded local snapshot, not the repo id.
    assert mgr._loras["remote"].path == snapshot


def test_load_adapter_unresolvable_repo_is_invalid_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec that is neither a local dir nor a resolvable repo is invalid."""
    mgr = _manager()

    def _raise(repo_id: str) -> str:
        raise ValueError("no such repo")

    monkeypatch.setattr(modulev3, "_download_adapter_repo", _raise)

    assert (
        mgr.load_adapter("ghost=org/does-not-exist")
        == LoRAStatus.LOAD_INVALID_PATH
    )


def test_load_adapter_local_dir_is_not_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local directories load in place, without any HF download."""
    mgr = _manager()
    downloads: list[str] = []

    def _record(repo_id: str) -> str:
        downloads.append(repo_id)
        return repo_id

    monkeypatch.setattr(modulev3, "_download_adapter_repo", _record)
    monkeypatch.setattr(modulev3, "_UnfusedLoRAModel", _RecordingModel)

    status = mgr.load_adapter(f"local={tmp_path}")

    assert status == LoRAStatus.SUCCESS
    assert mgr._loras["local"].path == str(tmp_path)
    assert downloads == []


class _Attn(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.o_proj = Linear(8, 8)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(x)


class _Model(Module[..., Tensor]):
    def __init__(self) -> None:
        self.self_attn = _Attn()

    def forward(self, x: Tensor) -> Tensor:
        return self.self_attn(x)


def test_wrap_applies_lora_and_wraps_model() -> None:
    target = LoRATargetModule(path="self_attn.o_proj", projections=("o_proj",))
    mgr = _manager(targets=(target,))
    with default_dtype(_WEIGHT_DTYPE):
        model = _Model()
    wrapped = mgr.wrap(model)

    # The target projection is now a LoRA layer discoverable from the wrapper.
    assert len(list(lora_layers(wrapped))) == 1
    # The fanout wrapper holds the inner model under ``model``, so its parameter
    # paths gain a ``model.`` prefix (which lora_weight_adapter compensates for).
    names = {name for name, _ in wrapped.parameters}
    assert names and all(name.startswith("model.") for name in names)


_HIDDEN = 8

# The Llama3 ModuleV3 LoRA targets: a name-transparent qkv (one adapter over
# q/k/v) and o_proj. Inlined to keep this test hermetic (no arch import).
_QKV_TARGETS = (
    LoRATargetModule("self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")),
    LoRATargetModule("self_attn.o_proj", ("o_proj",)),
)


class _AttnQKV(Module[[Tensor], Tensor]):
    """Attention mirroring the arch: name-transparent QKVLinear + o_proj."""

    def __init__(self) -> None:
        self.qkv_proj = QKVLinear(in_dim=_HIDDEN, q_dim=_HIDDEN, kv_dim=_HIDDEN)
        self.o_proj = Linear(_HIDDEN, _HIDDEN, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(self.qkv_proj(x))


class _Layer(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        self.self_attn = _AttnQKV()

    def forward(self, x: Tensor) -> Tensor:
        return self.self_attn(x)


class _LangModel(Module[..., Tensor]):
    def __init__(self) -> None:
        self.layers = ModuleList([_Layer()])

    def forward(self, x: Tensor) -> Tensor:
        return self.layers[0](x)


class _NestedModel(Module[..., Tensor]):
    def __init__(self) -> None:
        self.language_model = _LangModel()

    def forward(self, x: Tensor) -> Tensor:
        return self.language_model(x)


def _base_qkv_state_dict() -> dict[str, WeightData]:
    keys = [
        f"language_model.layers.0.self_attn.{p}.weight"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    return {
        k: WeightData.from_numpy(
            np.zeros((_HIDDEN, _HIDDEN), dtype=np.float32), k
        )
        for k in keys
    }


class _OpaqueLeaf(Module[[Tensor], Tensor]):
    """Stands in for a plain (opaque) ``Linear`` leaf, minus the weights."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class _TransparentLeaf(TransparentModule[[Tensor], Tensor]):
    """Stand-in for a name-transparent ``QKVLinear`` leaf (no weights)."""

    def forward(self, x: Tensor) -> Tensor:
        return x


def test_lora_qualify_name_is_wrapper_invisible() -> None:
    """``LoRA`` names its wrapped module's params as if the wrapper were absent.

    The ModuleV3 LoRA weight adapter relies on this: an opaque leaf keeps its
    native leaf name (``o_proj.weight``, not a leaked ``module.weight``), and a
    name-transparent leaf still passes its native child paths straight through.
    Constructed via ``object.__new__`` to exercise the naming contract without
    building real (compile-triggering) weights.
    """
    lora = object.__new__(LoRA)
    lora.module = _OpaqueLeaf()
    assert lora._qualify_name("o_proj", "module.weight") == "o_proj.weight"
    assert lora._qualify_name("o_proj", "module.bias") == "o_proj.bias"
    lora.module = _TransparentLeaf()
    assert lora._qualify_name("qkv_proj", "q_proj.weight") == "q_proj.weight"


def test_fuse_projections_keeps_transparent_qkv_separate() -> None:
    """A name-transparent (default) qkv target leaves its base q/k/v unfused.

    Pure key-level check with no model build or eager ops, so it guards the
    regression even where the compile backend is unavailable: concatenating
    q/k/v into a single ``qkv_proj.weight`` would drop the unfused keys the
    stacked=False ``QKVLinear`` still expects.
    """
    raw = _base_qkv_state_dict()
    out = modulev3._fuse_projections_for_lora(raw, _QKV_TARGETS)
    assert set(out) == set(raw)
    assert not any("qkv_proj" in key for key in out)


def test_lora_adapter_matches_wrapped_model_base_param_names() -> None:
    """Adapter-produced base keys exactly match the LoRA-wrapped model params.

    Locks the whole-block invariant for a LoRA-wrapped attention block:

    - the name-transparent ``QKVLinear`` keeps separate ``q_proj``/``k_proj``/
      ``v_proj`` (the adapter must not fuse them into ``qkv_proj.weight``), and
    - the ``LoRA`` wrapper stays invisible to naming, so the opaque ``o_proj``
      leaf keeps its native ``o_proj.weight`` (not a leaked ``module.weight``).

    Either divergence leaves a base weight missing from the weights mapping at
    compile time (the FinGPT serve KeyError).
    """
    mgr = _manager(targets=_QKV_TARGETS)
    with default_dtype(_WEIGHT_DTYPE):
        model = _NestedModel()
    wrapped = mgr.wrap(model)
    expected = {name for name, _ in wrapped.parameters}

    # The base adapter emits the checkpoint-native unfused q/k/v + o_proj (the
    # working non-LoRA path), under the arch's ``language_model`` prefix.
    raw = _base_qkv_state_dict()
    produced = set(mgr.lora_weight_adapter(lambda sd, **_: sd)(raw))

    assert produced == expected
    assert not any("qkv_proj" in key for key in produced)
    assert not any(".module" in key for key in produced)
