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
"""Inkling vision tower: a hierarchical MLP ("hmlp") over image patches.

Ported from ``vllm/models/inkling/common/towers.py`` at vLLM v0.26.0
``568afb3a13806beb53bb2e6bd518269357b237c0``.
"""

from __future__ import annotations

import itertools
import math

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from ..model_config import InklingVisionConfig

# The reference rounds every intermediate width up to a multiple of this.
_WIDTH_GRANULARITY = 64

# The vision config carries no epsilon of its own, and the reference never
# reads the text config's.
_NORM_EPS = 1e-6


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor:
            divisor += 1
        else:
            factors.append(divisor)
            value //= divisor
    if value > 1:
        factors.append(value)
    return factors


def plan_folds(config: InklingVisionConfig) -> list[tuple[int, int, int]]:
    """Plans the tower's ``n_layers`` steps as ``(t_fold, hw_fold, out_dim)``.

    A patch is folded into its channels one prime factor at a time, spatially
    first and then temporally; the tower's linears interpolate that chain by
    taking the ``n_layers + 1`` stops closest to evenly spaced in log scale.
    """

    def round_up(width: int) -> int:
        return -(-width // _WIDTH_GRANULARITY) * _WIDTH_GRANULARITY

    # Reachable folds as (temporal, spatial, channels), coarsening.
    stops = [(1, 1, config.n_channels)]
    spatial = temporal = 1
    for factor in reversed(_prime_factors(config.patch_size)):
        spatial *= factor
        stops.append((1, spatial, round_up(spatial**2 * config.n_channels)))
    for factor in reversed(_prime_factors(config.temporal_patch_size)):
        temporal *= factor
        stops.append(
            (
                temporal,
                spatial,
                round_up(temporal * spatial**2 * config.n_channels),
            )
        )

    # The reference spaces its targets over the full element count but measures
    # the stops without their channels, so the two scales do not line up.
    reached = [math.log(t * s * s) for t, s, _ in stops]
    full = math.log(
        config.temporal_patch_size * config.patch_size**2 * config.n_channels
    )
    chosen = [
        min(
            range(len(stops)),
            key=lambda stop: abs(full * step / config.n_layers - reached[stop]),
        )
        for step in range(config.n_layers + 1)
    ]
    # The chain must start unfolded and end fully folded whatever the spacing.
    chosen[0] = 0
    chosen[-1] = len(stops) - 1

    folds = []
    for step, (before, after) in enumerate(itertools.pairwise(chosen)):
        t_fold = stops[after][0] // stops[before][0]
        hw_fold = stops[after][1] // stops[before][1]
        # The last linear projects into the decoder instead of the next stop.
        out_dim = (
            config.decoder_dmodel
            if step == config.n_layers - 1
            else stops[after][2]
        )
        folds.append((t_fold, hw_fold, out_dim))
    return folds


def fold_timespace_to_depth(
    x: TensorValue, t_fold: int, hw_fold: int
) -> TensorValue:
    """Merges a ``t_fold x hw_fold x hw_fold`` neighborhood of
    ``[patches, t, h, w, c]`` into the channels, ordered frame, row, column,
    channel."""
    if t_fold == 1 and hw_fold == 1:
        return x
    patches = x.shape[0]
    t, h, w, c = (int(dim) for dim in x.shape[1:])
    if t % t_fold or h % hw_fold or w % hw_fold:
        raise ValueError(
            f"cannot fold {(t, h, w)} by {(t_fold, hw_fold, hw_fold)}"
        )
    split = x.reshape(
        [
            patches,
            t // t_fold,
            t_fold,
            h // hw_fold,
            hw_fold,
            w // hw_fold,
            hw_fold,
            c,
        ]
    )
    return ops.permute(split, [0, 1, 3, 5, 2, 4, 6, 7]).reshape(
        [
            patches,
            t // t_fold,
            h // hw_fold,
            w // hw_fold,
            t_fold * hw_fold * hw_fold * c,
        ]
    )


class InklingVisionModel(Module):
    """Encodes image patches into one decoder-width token per patch."""

    def __init__(
        self,
        config: InklingVisionConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.folds = plan_folds(config)

        linears = []
        norms = []
        in_dim = config.n_channels
        for index, (t_fold, hw_fold, out_dim) in enumerate(self.folds):
            linears.append(
                Linear(
                    in_dim=in_dim * t_fold * hw_fold * hw_fold,
                    out_dim=out_dim,
                    dtype=dtype,
                    device=device,
                    has_bias=False,
                )
            )
            if index < len(self.folds) - 1:
                norms.append(
                    RMSNorm(
                        out_dim,
                        dtype,
                        eps=_NORM_EPS,
                        multiply_before_cast=False,
                    )
                )
            in_dim = out_dim
        self.linears = LayerList(linears)
        self.norms = LayerList(norms)

        self.final_norm = (
            RMSNorm(
                config.decoder_dmodel,
                dtype,
                eps=_NORM_EPS,
                multiply_before_cast=False,
            )
            if config.use_vision_norm
            else None
        )

    def __call__(self, patches: TensorValue) -> TensorValue:
        x = patches.cast(self.dtype)
        for index, (t_fold, hw_fold, _) in enumerate(self.folds):
            x = fold_timespace_to_depth(x, t_fold, hw_fold)
            x = self.linears[index](x)
            if index < len(self.norms):
                x = ops.gelu(self.norms[index](x))
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x.reshape([x.shape[0], self.config.decoder_dmodel])

    def input_types(self) -> tuple[TensorType, ...]:
        """One block of raw patches, as the image processor emits them."""
        return (
            TensorType(
                DType.float32,
                shape=[
                    "total_patches",
                    self.config.temporal_patch_size,
                    self.config.patch_size,
                    self.config.patch_size,
                    self.config.n_channels,
                ],
                device=self.device,
            ),
        )
