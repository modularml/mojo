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
"""Dataset evaluation test for Kimi K2.7 vision transformer.

Loads real HuggingFace model weights and real ImageNet-1k images, then
compares the full 27-layer MAX vision transformer output against a
hand-written PyTorch reference model.

Before running this test, export ``HF_TOKEN`` for an account that has access
to the gated ImageNet dataset: https://huggingface.co/datasets/ILSVRC/imagenet-1k
If the token is missing or unauthorized, this test is skipped.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from conftest import TorchEncoder, TorchPatchEmbed, TorchPatchMergerMLP
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.comm import Signals
from max.pipelines.architectures.kimik2_5.layers.vision.data_processing import (
    compute_position_ids,
)
from max.pipelines.architectures.kimik2_5.layers.vision.transformer import (
    Transformer,
)
from max.pipelines.architectures.kimik2_5.model_config import VisionConfig
from max.pipelines.architectures.kimik2_5.vision_processor import (
    KimiK2_5VisionProcessor,
)
from max.pipelines.architectures.kimik2_5.weight_adapters import (
    _ATTN_RENAME_PATTERNS,
    KIMIK2_5_VISION_MAPPING,
)
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16
RTOL = 2e-2
ATOL = 4 * torch.finfo(TORCH_DTYPE).eps

HF_REPO_ID = "nvidia/Kimi-K2.7-Code-NVFP4"

NUM_EVAL_IMAGES = 5

# Not present in HF config.json — hardcoded defaults in modeling_kimi_k25.py:
#   IN_CHANNELS: default `in_dim=3` in MoonVision3dPatchEmbed.__init__
#   ROPE_MAX_*:  literal 512, 512 in MoonViT3dEncoder.__init__
#   ROPE_THETA:  default `theta_base=10000` in Rope2DPosEmbRepeated.__init__
IN_CHANNELS = 3
ROPE_MAX_HEIGHT = 512
ROPE_MAX_WIDTH = 512
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class _VisionConfig:
    """Vision transformer architecture config extracted from HF config.json."""

    num_layers: int
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    patch_size: int
    init_pos_emb_height: int
    init_pos_emb_width: int
    init_pos_emb_time: int
    merge_kernel_size: tuple[int, int]
    projector_ln_eps: float
    text_hidden_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


def _remap_hf_to_max(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remaps HuggingFace checkpoint keys to MAX Transformer keys.

    Applies the canonical ``KIMIK2_5_VISION_MAPPING`` (which produces
    ``vision_encoder.*`` prefixed names) then strips the
    ``vision_encoder.`` prefix so the keys match the Transformer module
    directly (e.g. ``encoder.blocks.0.*``, ``patch_merger.linear1.*``).
    """
    remapped: dict[str, torch.Tensor] = {}
    for k, v in hf_state_dict.items():
        new_k = k
        for before, after in KIMIK2_5_VISION_MAPPING.items():
            new_k = new_k.replace(before, after)
        for pattern, replacement in _ATTN_RENAME_PATTERNS:
            new_k = pattern.sub(replacement, new_k)

        new_k = new_k.removeprefix("vision_encoder.")
        remapped[new_k] = v
    return remapped


def _remap_hf_to_torch(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remaps HuggingFace checkpoint keys to TorchTransformer keys."""
    remapped: dict[str, torch.Tensor] = {}
    for k, v in hf_state_dict.items():
        new_k = k
        for before, after in KIMIK2_5_VISION_MAPPING.items():
            new_k = new_k.replace(before, after)
        new_k = new_k.removeprefix("vision_encoder.")
        remapped[new_k] = v
    return remapped


def _torch_tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int],
) -> list[torch.Tensor]:
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kH, kW = merge_kernel_size
        new_h, new_w = h // kH, w // kW
        reshaped = seq.view(t, new_h, kH, new_w, kW, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.view(new_h * new_w * kH * kW, -1))
        pre_sum += t * h * w
    return outputs


class TorchTransformer(nn.Module):
    def __init__(self, cfg: _VisionConfig) -> None:
        super().__init__()
        self.patch_embed = TorchPatchEmbed(
            out_dim=cfg.hidden_dim,
            in_dim=IN_CHANNELS,
            patch_size=cfg.patch_size,
            pos_emb_height=cfg.init_pos_emb_height,
            pos_emb_width=cfg.init_pos_emb_width,
            pos_emb_time=cfg.init_pos_emb_time,
        )
        self.encoder = TorchEncoder(
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            mlp_dim=cfg.mlp_dim,
            num_layers=cfg.num_layers,
            rope_max_height=ROPE_MAX_HEIGHT,
            rope_max_width=ROPE_MAX_WIDTH,
            rope_theta=ROPE_THETA,
        )
        self.merge_kernel_size = cfg.merge_kernel_size
        self.patch_merger = TorchPatchMergerMLP(
            mm_hidden_size=cfg.hidden_dim,
            decoder_hidden_size=cfg.text_hidden_size,
            merge_kernel_size=cfg.merge_kernel_size,
            eps=cfg.projector_ln_eps,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.patch_embed(pixel_values, grid_thws)
        input_row_offsets_list = [0]
        for t, h, w in grid_thws.tolist():
            input_row_offsets_list.append(
                input_row_offsets_list[-1] + t * h * w
            )
        input_row_offsets = torch.tensor(
            input_row_offsets_list, dtype=torch.int32
        )
        hidden = self.encoder(hidden, input_row_offsets, grid_thws)
        merged = _torch_tpool_patch_merger(
            hidden, grid_thws, self.merge_kernel_size
        )
        merged = torch.cat(merged, dim=0)
        kH, kW = self.merge_kernel_size
        merged = merged.reshape(-1, kH * kW, hidden.shape[-1])
        return self.patch_merger(merged)


@pytest.fixture(scope="session")
def vision_config() -> _VisionConfig:
    """Loads vision transformer config from the HuggingFace config.json."""
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    config_path = hf_hub_download(HF_REPO_ID, "config.json")
    with open(config_path) as f:
        vc = json.load(f)["vision_config"]
    return _VisionConfig(
        num_layers=vc["vt_num_hidden_layers"],
        hidden_dim=vc["vt_hidden_size"],
        num_heads=vc["vt_num_attention_heads"],
        mlp_dim=vc["vt_intermediate_size"],
        patch_size=vc["patch_size"],
        init_pos_emb_height=vc["init_pos_emb_height"],
        init_pos_emb_width=vc["init_pos_emb_width"],
        init_pos_emb_time=vc["init_pos_emb_time"],
        merge_kernel_size=tuple(vc["merge_kernel_size"]),
        projector_ln_eps=vc["projector_ln_eps"],
        text_hidden_size=vc["text_hidden_size"],
    )


_VISION_PREFIXES = ("vision_tower.", "mm_projector.", "multi_modal_projector.")


@pytest.fixture(scope="session")
def vision_tower_hf_weights() -> dict[str, torch.Tensor]:
    """Downloads safetensors shards containing vision tower and projector weights.

    Parses ``model.safetensors.index.json`` to identify which shard files
    contain ``vision_tower.*``, ``mm_projector.*``, or
    ``multi_modal_projector.*`` keys, then downloads only those shards.
    """
    index_path = hf_hub_download(
        HF_REPO_ID,
        "model.safetensors.index.json",
    )
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    vt_shards: set[str] = set()
    for key, shard in weight_map.items():
        if any(key.startswith(p) for p in _VISION_PREFIXES):
            vt_shards.add(shard)

    all_weights: dict[str, torch.Tensor] = {}
    for shard_name in sorted(vt_shards):
        shard_path = hf_hub_download(
            HF_REPO_ID,
            shard_name,
        )
        shard_weights = load_file(shard_path)
        for key, tensor in shard_weights.items():
            if any(key.startswith(p) for p in _VISION_PREFIXES):
                all_weights[key] = tensor

    logger.info(
        "Loaded %d vision weights from %d shard(s)",
        len(all_weights),
        len(vt_shards),
    )
    return all_weights


@pytest.fixture(scope="session")
def imagenet_images() -> list:  # type: ignore[type-arg]
    """Loads images from the ImageNet-1k validation split via streaming.

    Returns a list of PIL Images. Skips the test if the dataset is
    unavailable (e.g. no network or no HF_TOKEN for gated datasets).
    """
    if not os.environ.get("HF_TOKEN"):
        pytest.skip(
            "HF_TOKEN is required for gated ImageNet-1k dataset access. "
            "Set HF_TOKEN with a token that has access to "
            "https://huggingface.co/datasets/ILSVRC/imagenet-1k"
        )
    try:
        ds = load_dataset(
            "imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        images = []
        for i, sample in enumerate(ds):
            if i >= NUM_EVAL_IMAGES:
                break
            images.append(sample["image"].convert("RGB"))
        if len(images) < NUM_EVAL_IMAGES:
            pytest.skip(
                f"Only {len(images)} images available, need {NUM_EVAL_IMAGES}"
            )
    except Exception as e:
        pytest.skip(f"Could not load ImageNet-1k dataset: {e}")

    return images


def _build_and_run_max_transformer(
    cfg: _VisionConfig,
    state_dict: dict[str, torch.Tensor],
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    position_ids: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with the real-weight Transformer and execute."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    vc = VisionConfig(
        dtype=MAX_DTYPE,
        devices=[device_ref],
        init_pos_emb_height=cfg.init_pos_emb_height,
        init_pos_emb_time=cfg.init_pos_emb_time,
        init_pos_emb_width=cfg.init_pos_emb_width,
        merge_kernel_size=list(cfg.merge_kernel_size),
        mm_hidden_size=cfg.hidden_dim,
        patch_size=cfg.patch_size,
        projector_ln_eps=cfg.projector_ln_eps,
        text_hidden_size=cfg.text_hidden_size,
        vt_hidden_size=cfg.hidden_dim,
        vt_intermediate_size=cfg.mlp_dim,
        vt_num_attention_heads=cfg.num_heads,
        vt_num_hidden_layers=cfg.num_layers,
        in_channels=IN_CHANNELS,
        rope_max_height=ROPE_MAX_HEIGHT,
        rope_max_width=ROPE_MAX_WIDTH,
        rope_theta=ROPE_THETA,
    )
    vt = Transformer(vc)
    vt.load_state_dict(state_dict)

    session = InferenceSession(devices=[device])
    signals = Signals(devices=[device_ref])

    n_patches = int(pixel_values.shape[0])
    n_videos = int(grid_thws.shape[0])

    with Graph(
        "kimik2_5_vt_eval",
        input_types=[
            TensorType(
                MAX_DTYPE,
                [n_patches, IN_CHANNELS, cfg.patch_size, cfg.patch_size],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.int64,
                [n_videos, 3],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32,
                ["num_seqs"],
                device=DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            TensorType(
                DType.int64,
                [n_patches],
                device=DeviceRef.GPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        (
            pixel_values_in,
            grid_thws_in,
            input_row_offsets_in,
            max_seq_len_in,
            position_ids_in,
            *signal_inputs,
        ) = graph.inputs
        # Asserts for mypy
        assert isinstance(pixel_values_in, TensorValue)
        assert isinstance(grid_thws_in, TensorValue)
        assert isinstance(input_row_offsets_in, TensorValue)
        assert isinstance(max_seq_len_in, TensorValue)
        assert isinstance(position_ids_in, TensorValue)
        signal_buffers = [v.buffer for v in signal_inputs]
        outs = vt(
            [pixel_values_in],
            [grid_thws_in],
            [input_row_offsets_in],
            [max_seq_len_in],
            [position_ids_in],
            signal_buffers,
        )
        graph.output(outs[0])

    compiled = session.load(graph, weights_registry=vt.state_dict())
    (result,) = compiled.execute(
        Buffer.from_dlpack(pixel_values.cuda()),
        Buffer.from_dlpack(grid_thws.cuda()),
        Buffer.from_dlpack(input_row_offsets).to(device),
        max_seq_len,
        Buffer.from_dlpack(position_ids).to(device),
        *signals.buffers(),
    )
    assert isinstance(result, Buffer)
    return result


def _run_max_transformer(
    cfg: _VisionConfig,
    vision_tower_hf_weights: dict[str, torch.Tensor],
    pixel_values: torch.Tensor,
    grid_thws_tensor: torch.Tensor,
    input_row_offsets: torch.Tensor,
    max_seq_len: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Builds and runs the MAX vision transformer, returns a CUDA tensor."""
    max_state_dict = _remap_hf_to_max(vision_tower_hf_weights)
    max_output_buf = _build_and_run_max_transformer(
        cfg,
        max_state_dict,
        pixel_values,
        grid_thws_tensor,
        input_row_offsets,
        max_seq_len,
        position_ids,
    )
    result = torch.from_dlpack(max_output_buf).clone()

    del max_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return result


@pytest.fixture(scope="session")
def torch_reference_model(
    vision_config: _VisionConfig,
    vision_tower_hf_weights: dict[str, torch.Tensor],
) -> TorchTransformer:
    """Initializes and loads weights into the PyTorch reference model once."""
    torch_ref = TorchTransformer(vision_config)
    torch_ref_weights = _remap_hf_to_torch(vision_tower_hf_weights)
    torch_ref.load_state_dict(torch_ref_weights, strict=True)
    return torch_ref.to(dtype=TORCH_DTYPE, device="cuda").eval()


@pytest.mark.parametrize("image_index", list(range(NUM_EVAL_IMAGES)))
def test_vision_transformer_eval_torch_ref(
    image_index: int,
    vision_config: _VisionConfig,
    vision_tower_hf_weights: dict[str, torch.Tensor],
    imagenet_images: list,  # type: ignore[type-arg]
    torch_reference_model: TorchTransformer,
) -> None:
    """Test 27-layer vision transformer against hand-written torch reference."""
    cfg = vision_config
    image = imagenet_images[image_index]

    processor = KimiK2_5VisionProcessor()
    result = processor.preprocess([{"type": "image", "image": image}])
    pixel_values_np = result["pixel_values"]  # (n_patches, 3, 14, 14)
    grid_thws_np = result["grid_thws"]  # (1, 3)

    pixel_values = torch.from_numpy(pixel_values_np).to(TORCH_DTYPE)
    grid_thws_list = [tuple(row) for row in grid_thws_np.tolist()]

    seq_lens = [t * h * w for t, h, w in grid_thws_list]
    input_row_offsets = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.uint32
    )
    max_seq_len = torch.tensor([max(seq_lens)], dtype=torch.uint32)
    position_ids = torch.from_numpy(
        compute_position_ids(grid_thws_list, ROPE_MAX_WIDTH)
    )
    grid_thws_tensor = torch.tensor(grid_thws_list, dtype=torch.int64)

    max_result = _run_max_transformer(
        cfg,
        vision_tower_hf_weights,
        pixel_values,
        grid_thws_tensor,
        input_row_offsets,
        max_seq_len,
        position_ids,
    )

    with torch.no_grad():
        torch_output = torch_reference_model(
            pixel_values.cuda(), grid_thws_tensor.cuda()
        )

    assert max_result.shape == torch_output.shape, (
        f"Shape mismatch: MAX {max_result.shape} vs Torch {torch_output.shape}"
    )

    # Compute stats and move to CPU to free GPU memory between images.
    diff = (max_result.float() - torch_output.float()).abs().cpu()
    max_cpu = max_result.cpu()
    torch_cpu = torch_output.cpu()

    del max_result, torch_output
    gc.collect()
    torch.cuda.empty_cache()

    rmse = diff.pow(2).mean().sqrt().item()
    cos_sim = F.cosine_similarity(
        max_cpu.float().flatten().unsqueeze(0),
        torch_cpu.float().flatten().unsqueeze(0),
    ).item()
    pct_above_01 = (diff > 0.1).float().mean().item() * 100
    p50, p90, p99 = torch.quantile(diff, torch.tensor([0.5, 0.9, 0.99]))
    print(
        f"  image {image_index}: "
        f"rmse={rmse:.4f}  cos_sim={cos_sim:.6f}  "
        f"p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}  "
        f"max={diff.max():.4f}  "
        f">{0.1:.1f}: {pct_above_01:.1f}%"
    )

    torch.testing.assert_close(
        max_cpu,
        torch_cpu,
        rtol=RTOL,
        atol=ATOL,
    )
