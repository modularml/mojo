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
"""Eager-vs-graph parity test for the Gemma4 vision tower.

The graph tower is already validated against the HF reference
(max/tests/integration/architectures/gemma4/test_vision_*.py), so matching
it transitively validates the eager (ModuleV3) port.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.gemma4.vision_model.pooling import (
    compute_pool_gather_index,
)
from max.pipelines.architectures.gemma4.vision_model.vision_model import (
    Gemma4VisionModel as GraphGemma4VisionModel,
)
from max.pipelines.architectures.gemma4_modulev3.vision_model.vision_model import (
    Gemma4VisionModel as EagerGemma4VisionModel,
)

# Small but structurally faithful: 2 encoder layers, 2 images with
# DIFFERENT patch grids (8x8 and 4x4) so raggedness is exercised.
HIDDEN = 64
HEAD_DIM = 16  # 4 heads; head_dim//(2*ndim) integral for 2-D rope
NUM_HEADS = 4
INTERMEDIATE = 128
NUM_LAYERS = 2
PATCH_SIZE = 4  # patch dim = 3*16 = 48
POS_EMB_SIZE = 16
POOL_K = 2
TEXT_HIDDEN = 96
RMS_EPS = 1e-6
ROPE_THETA = 10_000.0

GRIDS = [(8, 8), (4, 4)]  # (width, height) per image

PATCH_DIM = 3 * PATCH_SIZE**2


def _make_config(devices: list[DeviceRef]) -> Any:
    vision_cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        patch_size=PATCH_SIZE,
        position_embedding_size=POS_EMB_SIZE,
        pooling_kernel_size=POOL_K,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        attention_bias=False,
        # Both MLP implementations take the post-``_HIDDEN_ACTIVATION_MAP``
        # name that ``Gemma4VisionConfig`` produces from the HF config.
        hidden_activation="gelu_tanh",
        standardize=True,
        image_size=None,
    )
    return SimpleNamespace(
        devices=devices,
        unquantized_dtype=DType.bfloat16,
        vision_config=vision_cfg,
        text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN),
    )


def _random_weights(seed: int = 42) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    def r(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, dtype=torch.float32) * 0.02).to(
            torch.bfloat16
        )

    w: dict[str, torch.Tensor] = {}
    w["patch_embedder.input_proj.weight"] = r(HIDDEN, PATCH_DIM)
    w["patch_embedder.position_embedding_table"] = r(2, POS_EMB_SIZE, HIDDEN)
    for i in range(NUM_LAYERS):
        p = f"encoder.layers.{i}."
        w[p + "self_attn.q_proj.weight"] = r(NUM_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = r(NUM_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = r(NUM_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = r(HIDDEN, NUM_HEADS * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = 1.0 + r(HEAD_DIM)
        w[p + "self_attn.k_norm.weight"] = 1.0 + r(HEAD_DIM)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            w[p + norm + ".weight"] = 1.0 + r(HIDDEN)
        w[p + "mlp.gate_proj.weight"] = r(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up_proj.weight"] = r(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down_proj.weight"] = r(HIDDEN, INTERMEDIATE)
    w["embed_vision.embedding_projection.weight"] = r(TEXT_HIDDEN, HIDDEN)
    w["std_bias"] = r(HIDDEN)
    w["std_scale"] = 1.0 + r(HIDDEN)
    return w


def _make_inputs() -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
]:
    rng = np.random.default_rng(7)
    pos_ids_per_image: list[npt.NDArray[np.integer[Any]]] = []
    for w_grid, h_grid in GRIDS:
        xs, ys = np.meshgrid(np.arange(w_grid), np.arange(h_grid))
        pos_ids_per_image.append(
            np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int32)
        )
    patch_counts = [p.shape[0] for p in pos_ids_per_image]
    total = sum(patch_counts)

    patches = rng.random((total, PATCH_DIM), dtype=np.float32)
    position_ids = np.concatenate(pos_ids_per_image, axis=0)
    cu_seqlens = np.cumsum([0] + patch_counts).astype(np.uint32)
    output_lengths = [(w // POOL_K) * (h // POOL_K) for (w, h) in GRIDS]
    pool_gather_index = compute_pool_gather_index(
        pos_ids_per_image, output_lengths, POOL_K
    )
    max_seq_len = np.array(max(patch_counts), dtype=np.uint32)
    return patches, position_ids, cu_seqlens, pool_gather_index, max_seq_len


def _input_buffers(
    inputs: tuple[npt.NDArray[Any], ...], device: Accelerator
) -> list[Buffer]:
    patches, pos_ids, cu, pgi, msl = inputs
    return [
        Buffer.from_dlpack(torch.from_numpy(patches).to(torch.bfloat16)).to(
            device
        ),
        Buffer.from_numpy(pos_ids).to(device),
        Buffer.from_numpy(cu).to(device),
        Buffer.from_numpy(pgi).to(device),
        Buffer.from_numpy(msl),
    ]


def _run_graph_tower(
    weights: dict[str, torch.Tensor], inputs: tuple[npt.NDArray[Any], ...]
) -> npt.NDArray[np.float32]:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    config = _make_config([DeviceRef.GPU()])
    tower = GraphGemma4VisionModel(config, DeviceRef.GPU())
    tower.load_state_dict(weights)

    with Graph(
        "gemma4_vision_graph", input_types=list(tower.input_types())
    ) as graph:
        values = [v.tensor for v in graph.inputs]
        patches, pos_ids, cu, pgi, msl = values
        # The graph tower takes one tensor per device for each ragged input.
        outputs = tower([patches], [pos_ids], [cu], [pgi], msl)
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=tower.state_dict())
    result = compiled.execute(*_input_buffers(inputs, device))[0]
    assert isinstance(result, Buffer)
    return torch.from_dlpack(result).to(torch.float32).cpu().numpy()


def _run_eager_tower(
    weights: dict[str, torch.Tensor], inputs: tuple[npt.NDArray[Any], ...]
) -> npt.NDArray[np.float32]:
    device = Accelerator()
    device_ref = DeviceRef.GPU()
    config = _make_config([device_ref])
    with F.lazy():
        tower = EagerGemma4VisionModel(config)
        tower.to(device)

    input_types = (
        TensorType(
            DType.bfloat16,
            shape=["total_patches", PATCH_DIM],
            device=device_ref,
        ),
        TensorType(DType.int32, shape=["total_patches", 2], device=device_ref),
        TensorType(
            DType.uint32, shape=["num_images_plus_1"], device=device_ref
        ),
        TensorType(
            DType.int32,
            shape=["num_pooled_tokens", "max_pool_patches"],
            device=device_ref,
        ),
        TensorType(DType.uint32, shape=[], device=DeviceRef.CPU()),
    )
    compiled = tower.compile(*input_types, weights=weights)
    result = compiled.execute_raw(*_input_buffers(inputs, device))[0]
    return torch.from_dlpack(result).to(torch.float32).cpu().numpy()


def test_gemma4_vision_tower_parity() -> None:
    weights = _random_weights()
    inputs = _make_inputs()
    graph_out = _run_graph_tower(weights, inputs)
    eager_out = _run_eager_tower(weights, inputs)

    num_pooled = sum((w // POOL_K) * (h // POOL_K) for (w, h) in GRIDS)
    assert graph_out.shape == (num_pooled, TEXT_HIDDEN)
    assert eager_out.shape == graph_out.shape
    # Guard against a vacuous match on all-zero or NaN output.
    assert np.isfinite(graph_out).all()
    assert np.abs(graph_out).max() > 0.0

    # Both towers run the same kernels, so this is currently bit-exact; the
    # tolerance is bf16 flash-vs-flash headroom, not an observed error.
    np.testing.assert_allclose(eager_out, graph_out, rtol=1e-2, atol=2e-3)
