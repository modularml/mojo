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
"""Shared MX expert-weight preshuffle for the AMD preb grouped-matmul kernel.

CPU byte-permutation of per-expert MXFP4 or MXFP8 ``B`` weights (and their E8M0
``B``-scales) into the 5D / 4D-cell layouts the AMD
``block_scaled_grouped_matmul_amd_preb`` kernel reads, so it can issue coalesced
DRAM->VGPR loads. Model weight adapters call
:func:`preshuffle_block_scaled_b_experts` + :func:`preshuffle_block_scaled_b_scales` in
lockstep and flip ``QuantConfig.block_scaled_preshuffled_b`` so ``MoEQuantized``
dispatches to the preb path.

Matches expert weights named ``...layers.N.mlp.experts.IDX.{gate,up,down}_proj``.
NOTE the caller flips ``block_scaled_preshuffled_b`` for the whole config regardless,
so any weight skipped from a matched group would be read as if it had been
permuted; the helpers below raise on that case rather than skip silently. Used
by the Kimi K2.5 and MiniMax-M3 adapters.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from max.driver import is_virtual_device_mode
from max.dtype import DType
from max.graph.weights import WeightData

logger = logging.getLogger("max.pipelines")

# The layer prefix is optional: Kimi keeps a `language_model.` prefix, while
# MiniMax-M3 strips it, leaving keys like `layers.N.mlp.experts.J...`.
#
# `decoder_layer` is the MiniMax-M3 MTP draft block, a single unindexed block
# that `minimax_m3_mtp/weight_adapters.py` attaches under that prefix. Without
# it the draft's experts match nothing, the helpers no-op silently, and a
# caller that flips `block_scaled_preshuffled_b` anyway gets row-major weights read as
# preshuffled.
_LAYER = r"(?:layers\.\d+|decoder_layer)"

_EXPERT_WEIGHT_RE = re.compile(
    rf"^(?P<prefix>(?:.+\.)?{_LAYER}\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_EXPERT_SCALE_RE = re.compile(
    rf"^(?P<prefix>(?:.+\.)?{_LAYER}\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight_scale$"
)

_SHARED_EXPERT_WEIGHT_RE = re.compile(
    rf"^(?P<prefix>(?:.+\.)?{_LAYER}\.mlp\.shared_experts)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_SHARED_EXPERT_SCALE_RE = re.compile(
    rf"^(?P<prefix>(?:.+\.)?{_LAYER}\.mlp\.shared_experts)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight_scale$"
)

# What an expert projection weight/scale looks like regardless of how the layer
# prefix is spelled. Used only to tell "this checkpoint has no experts" from
# "the strict patterns above no longer match this checkpoint's naming"; never
# to select what gets permuted.
_LOOSE_EXPERT_WEIGHT_RE = re.compile(r"(?:^|\.)mlp\.experts\..*_proj\.weight$")

_LOOSE_EXPERT_SCALE_RE = re.compile(
    r"(?:^|\.)mlp\.experts\..*_proj\.weight_scale$"
)


# MFMA geometry, mirroring the constants of the same name in
# `max/kernels/src/linalg/matmul/gpu/amd/mxfp4_preshuffle_layouts.mojo`.
_MFMA_MN_LANES = 16
_MFMA_K_LANES = 4
_MFMA_LANE_BYTES = 16

MXFP6_LANE_BYTES = 24
"""Bytes one lane feeds the MFMA for FP6 (32 elements x 6 bits).

FP4 and FP8 both drive :class:`PreshuffledBLoader` with ``lane_bytes=16`` --
FP8 splits its 32-byte fragment into two 16-byte halves at a K stride -- so 24
is the only width that needs the plane-split layout.
"""


def _as_shuffleable_mxfp4_b(
    wd: WeightData, lane_bytes: int = 16
) -> np.ndarray | None:
    """Return ``wd`` as a numpy view if it's a shuffleable MX B weight.

    Shuffleable when the dtype is packed-MX (uint8) or MXFP8
    (``float8_e4m3fn``) and the dims are MFMA-tile-aligned: ``N % 16 == 0``
    (NLane=16) and ``K_BYTES`` a whole number of MFMA K tiles
    (``4 * lane_bytes``). The shuffle reshapes hardcode those factors, so
    non-aligned dims would crash on reshape. Both shuffles are pure byte
    permutations, so FP8 bytes are reinterpreted as uint8 here; the caller
    restores the dtype. Returns ``None`` when the weight isn't shuffleable.

    The uint8 reinterpret happens on the MAX Buffer, *before* numpy ever sees
    the bytes: numpy has no float8 dtypes, and the exception ``np.from_dlpack``
    raises for them is numpy-version-dependent (``RuntimeError`` < 2.5.0,
    ``BufferError`` >= 2.5.0, numpy gh-30937). Dispatching on that exception
    type is what silently skipped every MXFP8 expert under numpy 2.5 and served
    unshuffled weights to the preb kernel (KERN-3393).
    """
    if wd.dtype not in (DType.uint8, DType.float8_e4m3fn):
        return None
    arr = np.from_dlpack(wd.to_buffer().view(DType.uint8, wd.shape.static_dims))
    mfma_k_bytes = _MFMA_K_LANES * lane_bytes
    if (
        arr.ndim != 2
        or arr.shape[0] % _MFMA_MN_LANES != 0
        or arr.shape[1] % mfma_k_bytes != 0
    ):
        return None
    return arr


def _as_shuffleable_mxfp4_b_scale(wd: WeightData) -> np.ndarray | None:
    """Return ``wd`` as a uint8 view if it's a shuffleable MXFP4 B scale.

    Shuffleable when dtype is E8M0 and dims are cell-aligned for
    ``Shuffler.scale_4d_grouped_layout``: ``N % 32 == 0`` (S_MN_BLOCK) and
    ``K_SCALES % 8 == 0`` (S_K_BLOCK). The 2D src reshape used by
    :func:`_shuffle_scale_4d` hardcodes those factors. Returns ``None``
    when not shuffleable. E8M0 bytes are reinterpreted as uint8 for byte
    permutation; the dtype is restored to E8M0 by the caller.
    """
    if wd.dtype != DType.float8_e8m0fnu:
        return None
    arr = np.from_dlpack(wd.to_buffer().view(DType.uint8, wd.shape.static_dims))
    if arr.ndim != 2 or arr.shape[0] % 32 != 0 or arr.shape[1] % 8 != 0:
        return None
    return arr


def _shuffle_b_5d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 expert B bytes into ``Shuffler.b_5d_grouped_layout``.

    Reshape ``[N, K_BYTES]`` row-major into the 5D tile structure
    ``(N0, NLane=16, K0, KLane=4, KPack=16)`` and transpose into
    ``(N0, K0, KLane, NLane, KPack)`` so C-order strides match
    ``b_5d_grouped_layout`` in ``block_scaled_preshuffle_layouts.mojo``. ``dst``
    is a contiguous ``(N, K_BYTES)`` slot the caller owns.
    """
    N, K_BYTES = src.shape
    src_v = src.reshape(N // 16, 16, K_BYTES // 64, 4, 16).transpose(
        0, 2, 3, 1, 4
    )
    dst_v = dst.reshape(N // 16, K_BYTES // 64, 4, 16, 16)
    np.copyto(dst_v, src_v)


def _shuffle_b_planes(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP6 expert B bytes into ``Shuffler.b_plane_byte_off`` order.

    An FP6 lane fragment is 24 bytes, which no power-of-two load covers, so the
    kernel splits it into a 16-byte and an 8-byte plane and issues one
    naturally-aligned load per plane. The planes are stacked within each MFMA
    tile: with ``tile_bytes = 16 * (4 * 24) = 1536``, plane 0 occupies
    ``[0, 1024)`` (64 lanes x 16B) and plane 1 ``[1024, 1536)`` (64 lanes x 8B),
    each ordered ``(klane, nlane, byte)``.

    Plane 0 is source fragment bytes ``[0:16]`` and plane 1 is bytes
    ``[16:24]``, so the whole permutation is a reshape plus transpose. ``dst``
    is a contiguous ``(N, K_BYTES)`` slot the caller owns.
    """
    N, K_BYTES = src.shape
    mfma_k_bytes = _MFMA_K_LANES * MXFP6_LANE_BYTES
    n0, k0 = N // _MFMA_MN_LANES, K_BYTES // mfma_k_bytes

    # (n0, nlane, k0, klane, byte) -> (n0, k0, klane, nlane, byte)
    src_v = src.reshape(
        n0, _MFMA_MN_LANES, k0, _MFMA_K_LANES, MXFP6_LANE_BYTES
    ).transpose(0, 2, 3, 1, 4)

    plane0_bytes = _MFMA_K_LANES * _MFMA_MN_LANES * _MFMA_LANE_BYTES
    dst_v = dst.reshape(n0, k0, mfma_k_bytes * _MFMA_MN_LANES)
    np.copyto(
        dst_v[..., :plane0_bytes].reshape(
            n0, k0, _MFMA_K_LANES, _MFMA_MN_LANES, _MFMA_LANE_BYTES
        ),
        src_v[..., :_MFMA_LANE_BYTES],
    )
    np.copyto(
        dst_v[..., plane0_bytes:].reshape(
            n0,
            k0,
            _MFMA_K_LANES,
            _MFMA_MN_LANES,
            MXFP6_LANE_BYTES - _MFMA_LANE_BYTES,
        ),
        src_v[..., _MFMA_LANE_BYTES:],
    )


def _shuffle_scale_4d(src: np.ndarray, dst: np.ndarray) -> None:
    """Permute MXFP4 B-scale bytes into ``Shuffler.scale_4d_grouped_layout``.

    Reshape ``[MN, K_SCALES]`` row-major into the 6D decomposition
    ``(MN_block, MN_pack=2, MN_lane=16, K_block, K_pack=2, K_lane=4)``
    and transpose into the dst axis order
    ``(MN_block, K_block, K_lane, MN_lane, K_pack, MN_pack)`` so C-order
    strides match the 4D-cell byte layout addressed by
    ``Shuffler.scale_4d_byte_off``. Within each i32 cell the bytes land
    in ``(mn_pack, k_pack) = {(0,0), (1,0), (0,1), (1,1)}`` order at
    byte offsets ``{0, 1, 2, 3}`` — what the preb kernel's OPSEL byte
    selector reads.
    """
    MN, K_SCALES = src.shape
    src_v = src.reshape(MN // 32, 2, 16, K_SCALES // 8, 2, 4).transpose(
        0, 3, 5, 2, 4, 1
    )
    dst_v = dst.reshape(MN // 32, K_SCALES // 8, 4, 16, 2, 2)
    np.copyto(dst_v, src_v)


def preshuffle_block_scaled_b_experts(
    state_dict: dict[str, WeightData],
    *,
    include_shared_weights: bool = False,
    lane_bytes: int = 16,
) -> int:
    """MX B preshuffle of all per-expert weights in-place on CPU.

    Walks ``state_dict``, groups expert weights by ``(prefix, proj)``,
    rewrites each group's WeightData entries with the bytes laid out in the
    layout ``lane_bytes`` selects, so the AMD
    ``block_scaled_grouped_matmul_amd_preb`` kernel reads them with coalesced
    DRAM->VGPR loads. Raises when a matched group holds any weight that isn't
    MX-packed (uint8 or float8_e4m3fn) with tile-aligned dims: the caller flips
    ``block_scaled_preshuffled_b`` for the whole config, so a skipped weight
    would be read as if it had been permuted.

    One numpy buffer per ``(prefix, proj)`` group keeps allocation count
    at ~180. Per-expert allocations would mean ~70k mmap chunks, blowing
    past glibc's M_MMAP_MAX (65536).

    Args:
        state_dict: Weights to permute in place.
        include_shared_weights: Also permute ``mlp.shared_experts`` weights.
        lane_bytes: Bytes one lane feeds the MFMA. 16 for MXFP4 and MXFP8
            (``b_5d_grouped_layout``); :data:`MXFP6_LANE_BYTES` for MXFP6,
            which selects the plane-split layout instead.

    Returns:
        How many expert weights the FQN patterns matched. Counted under
        virtual devices too, where the byte permutation is skipped, so a
        caller can tell "matched nothing" from "did the work" in both modes.
    """
    if lane_bytes not in (_MFMA_LANE_BYTES, MXFP6_LANE_BYTES):
        raise ValueError(
            f"unsupported preshuffle lane_bytes {lane_bytes}; expected "
            f"{_MFMA_LANE_BYTES} (MXFP4/MXFP8) or {MXFP6_LANE_BYTES} (MXFP6)"
        )
    shuffle = (
        _shuffle_b_planes if lane_bytes == MXFP6_LANE_BYTES else _shuffle_b_5d
    )
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_WEIGHT_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)
            continue

        if include_shared_weights:
            if m := _SHARED_EXPERT_WEIGHT_RE.match(name):
                groups[m["prefix"], m["proj"]].append(name)
                continue

    if not groups:
        if unmatched := [
            n for n in state_dict if _LOOSE_EXPERT_WEIGHT_RE.search(n)
        ]:
            raise ValueError(
                "MX expert preshuffle matched no weights, but this checkpoint "
                f"has expert tensors (e.g. {unmatched[0]!r}), so the preb "
                "kernel would read them row-major. Teach this module's FQN "
                "patterns about this naming."
            )
        return 0

    t0 = time.perf_counter()
    n_total = 0
    # Under virtual devices skip the byte copy; the group check still runs.
    permute = not is_virtual_device_mode()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (
                    arr := _as_shuffleable_mxfp4_b(state_dict[name], lane_bytes)
                )
                is not None
            ]
            if len(shuffleable) != len(names):
                # Any skipped weight in a matched group is a bug, zero and
                # partial alike: the caller flips `block_scaled_preshuffled_b` for
                # the whole config, so a skipped weight would be read as if
                # it had been permuted (KERN-3393 served exactly that).
                skipped = sorted(set(names) - {n for n, _ in shuffleable})
                raise ValueError(
                    "MX expert-weight preshuffle skipped "
                    f"{len(skipped)}/{len(names)} weights in a matched "
                    f"group, e.g. {skipped[0]!r} with dtype "
                    f"{state_dict[skipped[0]].dtype}. Every weight in a "
                    "group must have a shuffleable dtype (uint8 or "
                    "float8_e4m3fn) and tile-aligned dims."
                )

            kept_names, srcs = zip(*shuffleable, strict=True)
            n_total += len(srcs)
            if not permute:
                continue
            N, K_BYTES = srcs[0].shape
            buf = np.empty((len(srcs), N, K_BYTES), dtype=np.uint8)
            list(pool.map(shuffle, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                # The slab is uint8; restore the source dtype so MXFP8 weights
                # stay float8_e4m3fn for downstream dtype checks.
                state_dict[name] = dataclasses.replace(
                    WeightData.from_numpy(slot, name=state_dict[name].name),
                    dtype=state_dict[name].dtype,
                )

    logger.info(
        "MXFP4 B preshuffle%s: %d experts across %d groups in %.1fs",
        "" if permute else " (dummy op under virtual compilation)",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )
    return n_total


def preshuffle_block_scaled_b_scales(
    state_dict: dict[str, WeightData],
    *,
    include_shared_weights: bool = False,
) -> int:
    """MXFP4 B-scale preshuffle of all per-expert scales in-place on CPU.

    Walks ``state_dict``, groups expert scales by ``(prefix, proj)``,
    rewrites each group's WeightData entries with bytes laid out in
    ``scale_4d_grouped_layout`` so the AMD preb grouped-matmul kernel
    can issue direct-VGPR i32 scale loads (one 2x2 cell per lane).
    Raises when a matched group holds any scale that isn't E8M0 with
    cell-aligned dims (``N % 32 == 0`` and ``K_SCALES % 8 == 0``), for
    the same reason as :func:`preshuffle_block_scaled_b_experts`.

    Companion to :func:`preshuffle_block_scaled_b_experts`; should be called
    immediately after it so weight and scale layouts stay in sync.

    Returns:
        How many expert scales the FQN patterns matched, on the same terms
        as :func:`preshuffle_block_scaled_b_experts`.
    """
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for name in state_dict:
        if m := _EXPERT_SCALE_RE.match(name):
            groups[m["prefix"], m["proj"]].append(name)
            continue

        if include_shared_weights:
            if m := _SHARED_EXPERT_SCALE_RE.match(name):
                groups[m["prefix"], m["proj"]].append(name)
                continue

    if not groups:
        if unmatched := [
            n for n in state_dict if _LOOSE_EXPERT_SCALE_RE.search(n)
        ]:
            raise ValueError(
                "MX expert B-scale preshuffle matched no scales, but this "
                f"checkpoint has expert scales (e.g. {unmatched[0]!r}), so the "
                "preb kernel would read them row-major. Teach this module's "
                "FQN patterns about this naming."
            )
        return 0

    t0 = time.perf_counter()
    n_total = 0
    # Under virtual devices skip the byte copy; the group check still runs.
    permute = not is_virtual_device_mode()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for names in groups.values():
            shuffleable = [
                (name, arr)
                for name in names
                if (arr := _as_shuffleable_mxfp4_b_scale(state_dict[name]))
                is not None
            ]
            if len(shuffleable) != len(names):
                # Same invariant as the weight preshuffle above: the caller
                # flips `block_scaled_preshuffled_b` for the whole config, so a
                # skipped scale would be read as if it had been permuted.
                skipped = sorted(set(names) - {n for n, _ in shuffleable})
                raise ValueError(
                    "MX expert B-scale preshuffle skipped "
                    f"{len(skipped)}/{len(names)} scales in a matched "
                    f"group, e.g. {skipped[0]!r} with dtype "
                    f"{state_dict[skipped[0]].dtype}. Every scale in a "
                    "group must be E8M0 with cell-aligned dims."
                )

            kept_names, srcs = zip(*shuffleable, strict=True)
            n_total += len(srcs)
            if not permute:
                continue
            MN, K_SCALES = srcs[0].shape
            buf = np.empty((len(srcs), MN, K_SCALES), dtype=np.uint8)
            list(pool.map(_shuffle_scale_4d, srcs, buf))
            for name, slot in zip(kept_names, buf, strict=True):
                # from_numpy infers uint8 from the slab dtype; restore the
                # E8M0 metadata so downstream graph-compiler dtype checks
                # (e.g. grouped_dynamic_block_scaled_matmul_amd) still pass.
                state_dict[name] = dataclasses.replace(
                    WeightData.from_numpy(slot, name=state_dict[name].name),
                    dtype=DType.float8_e8m0fnu,
                )

    logger.info(
        "MXFP4 B-scale preshuffle%s: %d experts across %d groups in %.1fs",
        "" if permute else " (dummy op under virtual compilation)",
        n_total,
        len(groups),
        time.perf_counter() - t0,
    )
    return n_total


def shuffle_block_scaled_b_dense_arrays(
    b: np.ndarray,
    b_scale: np.ndarray,
    *,
    lane_bytes: int = MXFP6_LANE_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutes a raw dense B weight and its E8M0 scales into preshuffled layouts.

    Thin numpy-only counterpart to :func:`preshuffle_block_scaled_b_dense`,
    for callers that hold plain ``[N, K_BYTES]`` / ``[N, K_SCALES]`` arrays
    rather than checkpoint ``WeightData`` -- e.g. a benchmark harness that
    generates synthetic weights directly.

    Args:
        b: Dense B weight, uint8 ``[N, K_BYTES]``.
        b_scale: B's E8M0 scales, uint8-viewed ``[N, K_SCALES]``.
        lane_bytes: Bytes one lane feeds the MFMA. Only
            :data:`MXFP6_LANE_BYTES` (24) is supported today.

    Returns:
        The ``(b, b_scale)`` pair with bytes permuted into the plane-split /
        packed-scale layouts.
    """
    if lane_bytes != MXFP6_LANE_BYTES:
        raise ValueError(
            "shuffle_block_scaled_b_dense_arrays only supports MXFP6 "
            f"(lane_bytes={MXFP6_LANE_BYTES}) today, got {lane_bytes}"
        )
    b_dst = np.empty_like(b)
    _shuffle_b_planes(b, b_dst)
    scale_dst = np.empty_like(b_scale)
    _shuffle_scale_4d(b_scale, scale_dst)
    return b_dst, scale_dst


def preshuffle_block_scaled_b_dense(
    state_dict: dict[str, WeightData],
    names: Iterable[str],
    *,
    lane_bytes: int = MXFP6_LANE_BYTES,
) -> None:
    """MX B preshuffle of dense (non-expert) weights in-place on CPU.

    The single-tensor counterpart of :func:`preshuffle_block_scaled_b_experts`
    + :func:`preshuffle_block_scaled_b_scales`: each name in ``names`` is
    already one complete ``[N, K_BYTES]`` dense weight (a plain Linear layer,
    not a batch of per-expert weights), so this shuffles the weight and its
    ``.weight_scale`` sibling directly with no expert-name grouping. Raises on
    any name whose weight or scale isn't shuffleable, for the same reason the
    expert helpers do: the caller flips the matmul op's ``preshuffled_b``
    parameter for the whole weight, so a silently-skipped tensor would be read
    as if it had been permuted.

    Args:
        state_dict: Weights to permute in place.
        names: Names of dense ``.weight`` tensors to preshuffle; each name's
            ``f"{name}_scale"`` sibling is preshuffled too.
        lane_bytes: Bytes one lane feeds the MFMA. Only
            :data:`MXFP6_LANE_BYTES` (24, the plane-split layout) is
            supported today; dense MXFP4/MXFP8 matmul has no preshuffled path
            yet.
    """
    if lane_bytes != MXFP6_LANE_BYTES:
        raise ValueError(
            "preshuffle_block_scaled_b_dense only supports MXFP6 "
            f"(lane_bytes={MXFP6_LANE_BYTES}) today, got {lane_bytes}"
        )

    t0 = time.perf_counter()
    n_total = 0
    for name in names:
        wd = state_dict[name]
        arr = _as_shuffleable_mxfp4_b(wd, lane_bytes)
        if arr is None:
            raise ValueError(
                f"{name!r} is not a shuffleable MX weight (dtype "
                f"{wd.dtype}, shape {wd.shape}); needs uint8/float8_e4m3fn "
                f"with N % 16 == 0 and K_BYTES a multiple of {4 * lane_bytes}"
            )
        scale_name = f"{name}_scale"
        sf_wd = state_dict[scale_name]
        sf_arr = _as_shuffleable_mxfp4_b_scale(sf_wd)
        if sf_arr is None:
            raise ValueError(
                f"{scale_name!r} is not a shuffleable E8M0 scale (dtype "
                f"{sf_wd.dtype}, shape {sf_wd.shape}); needs N % 32 == 0 "
                "and K_SCALES % 8 == 0"
            )

        dst, sf_dst = shuffle_block_scaled_b_dense_arrays(
            arr, sf_arr, lane_bytes=lane_bytes
        )
        state_dict[name] = dataclasses.replace(
            WeightData.from_numpy(dst, name=wd.name), dtype=wd.dtype
        )
        state_dict[scale_name] = dataclasses.replace(
            WeightData.from_numpy(sf_dst, name=sf_wd.name),
            dtype=DType.float8_e8m0fnu,
        )
        n_total += 1

    logger.info(
        "MXFP6 dense B preshuffle: %d weights in %.1fs",
        n_total,
        time.perf_counter() - t0,
    )
