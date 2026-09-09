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
"""Shared helpers for the sparse MLA GPU integration tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


def random_weights(
    attn: LatentAttentionWithRopeFp8 | LatentAttentionWithRope,
) -> dict[str, WeightData]:
    """Small random :class:`WeightData` for ``attn`` (the e2e weights registry)."""
    registry: dict[str, WeightData] = {}
    for name, w in attn.raw_state_dict().items():
        shape = tuple(int(s) for s in w.shape)
        dtype = w.dtype
        if dtype == DType.float8_e4m3fn:
            t = (torch.randn(shape, dtype=torch.float32) * 0.02).to(
                torch.float8_e4m3fn
            )
            buf = Buffer.from_dlpack(t.view(torch.uint8)).view(
                DType.float8_e4m3fn
            )
        elif dtype == DType.bfloat16:
            t = (torch.randn(shape, dtype=torch.float32) * 0.02).to(
                torch.bfloat16
            )
            buf = Buffer.from_dlpack(t)
        elif dtype == DType.float32:
            t = torch.randn(shape, dtype=torch.float32) * 0.02
            buf = Buffer.from_dlpack(t)
        else:
            raise AssertionError(f"unsupported weight dtype {dtype} for {name}")
        registry[name] = WeightData(buf, name, dtype, Shape(shape))
    return registry


def paged_kv_from_flat_graph_inputs(
    kv_params: KVCacheParams,
    flat_kv_inputs: Sequence[object],
) -> PagedCacheValues:
    """Flattened graph inputs → :class:`PagedCacheValues` for one device."""
    return (
        kv_params.get_symbolic_inputs()
        .unflatten(iter(flat_kv_inputs))
        .inputs[0]
    )
