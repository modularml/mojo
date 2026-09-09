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
"""Expert-parallel forward pass for MoE layers.

Provides :func:`forward_moe_sharded_layers`, the single entry point for
running EP-sharded layers (EP MoE *or* DP replicated MLP) in the
forward pass.  Internally it checks whether the shards are EP-enabled
MoE and dispatches to the EP-specific logic; otherwise it falls back to
:func:`forward_sharded_layers`.

The caller is responsible for calling
:meth:`EPBatchManager.fetch_buffers` before invoking this function.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    ops,
)

from ..comm.ep.ep_manager import EPBatchManager
from ..kernels import moe_eplb_remap, moe_router_single_group_eplb
from ..layer import Module
from ..transformer.distributed_transformer import forward_sharded_layers
from .moe import MoE

logger = logging.getLogger("max.serve")

# Stream selector for the shared expert. Stream 0 is the default (routed/EP)
# stream; binding the shared expert to a side stream lets it run concurrently
# with the routed-expert dispatch/combine.
_SHARED_EXPERT_STREAM_ID = 1


def _shared_experts_on_side_stream(
    shards: Sequence[Module], xs: Sequence[TensorValue]
) -> list[TensorValue]:
    """Runs the replicated shared expert on a dedicated side stream.

    Wraps the per-device shared-expert forwards in a single ``mo.sequence``
    (via :func:`~max.graph.ops.side_stream`) tagged with the side-stream id. The
    graph compiler binds the whole region to a per-device side-stream
    device-context view and inserts the boundary cross-stream synchronization,
    so the shared expert overlaps the routed-expert dispatch/combine on the main
    stream. Because the region is not isolated-from-above, the body captures each
    shard's weights directly from the enclosing scope -- no subgraph, weight
    placeholderization, or dim rebinding is required.

    Args:
        shards: Per-device sharded shared-expert modules, aligned with ``xs``.
        xs: Per-device inputs; one shared-expert output is returned per input,
            in the same order.
    """
    shard_list = list(shards)
    x_list = [TensorValue(x) for x in xs]

    def body(*ins: TensorValue) -> list[TensorValue]:
        return [shard(x) for shard, x in zip(shard_list, ins, strict=True)]

    return ops.side_stream(
        x_list,
        body,
        result_types=[x.type for x in x_list],
        stream_id=_SHARED_EXPERT_STREAM_ID,
    )


def _compute_shared_experts(
    moe_shards: list[MoE],
    xs: list[TensorValue],
    device_idxs: Sequence[int],
) -> list[TensorValue | None]:
    """Returns the shared-expert output for each device index in ``device_idxs``.

    Honors ``MODULAR_OVERLAP_SHARED_EXPERT`` (default on): when enabled, runs the
    replicated shared expert on a side stream that overlaps the routed-expert
    path; when ``0``, runs it inline on the default stream (baseline for A/B
    comparison). The returned list is aligned with ``device_idxs``. The element
    type is widened to ``TensorValue | None`` so callers can splice the result
    directly into the per-device ``shared_outs`` list (which is ``None`` on
    devices that skip the shared expert).
    """
    if os.environ.get("MODULAR_OVERLAP_SHARED_EXPERT", "1") == "0":
        return [moe_shards[i].shared_experts(xs[i]) for i in device_idxs]
    return [
        *_shared_experts_on_side_stream(
            [moe_shards[i].shared_experts for i in device_idxs],
            [xs[i] for i in device_idxs],
        )
    ]


def _ep_forward(
    moe_shards: list[MoE],
    xs: list[TensorValue],
    eplb_counter_buffers: list[BufferValue] | None = None,
    layer_idx_per_device: list[TensorValue] | None = None,
) -> list[TensorValue]:
    """Runs the EP MoE forward pass using multi-device dispatch and combine.

    Uses single multi-device graph ops for both dispatch and combine,
    with per-shard gate and local expert compute in between:
    gate -> multi-device dispatch -> local compute -> multi-device combine.

    Returns:
        outputs: One output tensor per shard.
    """

    all_topk_ids: list[TensorValue] = []
    all_router_weights: list[TensorValue] = []
    all_input_scales: list[TensorValue | None] = []
    device_ids: list[int] = []

    batch_mgr = moe_shards[0].ep_batch_manager
    _eplb_remap_logged = False
    for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
        gate = shard.gate
        device = shard.devices[0]

        use_fused_eplb_router = (
            batch_mgr.config.eplb_enabled
            and getattr(gate, "n_group", 1) == 1
            and not _eplb_is_identity_placement(batch_mgr)
            and hasattr(
                gate, "compute_scores"
            )  # only Kimi gate has this for now
        )

        if use_fused_eplb_router:
            assert layer_idx_per_device, (
                "EPLB requires per-device layer_idx tensors; got "
                f"{layer_idx_per_device!r}"
            )
            layer_idx_t = layer_idx_per_device[device.id]
            log2phy = ops.buffer_load(
                batch_mgr._eplb_log2phy_per_device[device.id]
            )
            logcnt = ops.buffer_load(
                batch_mgr._eplb_logcnt_per_device[device.id]
            )

            scores = gate.compute_scores(x)  # type: ignore[attr-defined]
            phy_idx, log_idx, router_weight = moe_router_single_group_eplb(
                expert_scores=scores,
                expert_bias=gate.e_score_correction_bias,  # type: ignore[attr-defined]
                logcnt=logcnt,
                log2phy=log2phy,
                layer_idx=layer_idx_t,
                n_routed_experts=gate.num_experts,
                n_experts_per_tok=gate.num_experts_per_token,
                norm_weights=gate.norm_topk_prob,  # type: ignore[attr-defined]
                num_log=gate.num_experts,
                max_replicas=batch_mgr.config.max_replicas,
                hash_decorrelate=getattr(
                    batch_mgr.config, "eplb_hash_decorrelate", True
                ),
                routed_scaling_factor=gate.routed_scaling_factor,  # type: ignore[attr-defined]
            )

            # Histogram still consumes LOGICAL ids — emit it BEFORE swapping.
            if eplb_counter_buffers:
                _accumulate(
                    eplb_counter_buffers[i],
                    log_idx.reshape([-1]),
                    device,
                    gate.num_experts,
                )

            if not _eplb_remap_logged and i == 0:
                logger.info(
                    "EPLB: fused mo.moe.single.group.router.eplb installed "
                    "in compiled graph (layer_idx=%s, max_replicas=%d)",
                    shard.layer_idx,
                    batch_mgr.config.max_replicas,
                )
                _eplb_remap_logged = True

            router_idx = phy_idx
        else:
            # Original chained path. Covers DSV3 (n_group > 1) and any
            # build where the gate doesn't yet expose compute_scores.
            router_idx, router_weight = gate(x)
            router_idx = ops.cast(router_idx, DType.int32)

            if eplb_counter_buffers:
                _accumulate(
                    eplb_counter_buffers[i],
                    ops.cast(router_idx, DType.int32).reshape([-1]),
                    device,
                    gate.num_experts,
                )

            if batch_mgr.config.eplb_enabled:
                assert layer_idx_per_device, (
                    "EPLB requires per-device layer_idx tensors; got "
                    f"{layer_idx_per_device!r}"
                )
                layer_idx_t = layer_idx_per_device[device.id]
                if not _eplb_remap_logged and i == 0:
                    logger.info(
                        "EPLB: fused mo.moe.eplb.remap installed in compiled graph "
                        "(layer_idx=%s, max_replicas=%d)",
                        shard.layer_idx,
                        batch_mgr.config.max_replicas,
                    )
                    _eplb_remap_logged = True
                router_idx = _eplb_remap(
                    router_idx, shard, batch_mgr, layer_idx_t
                )

        all_topk_ids.append(router_idx)
        all_router_weights.append(router_weight)
        all_input_scales.append(shard._ep_dispatch_input_scales())
        device_ids.append(shard.devices[0].id)

    # Collect non-None scales into a list (all-or-nothing for NVFP4).
    scales: list[TensorValue] | None = None
    if all_input_scales[0] is not None:
        scales = [s for s in all_input_scales if s is not None]

    # When the model has an unfused shared expert and non-allreduce EP, split
    # the per-device dispatch and combine into async launch + wait and run the
    # shared-expert subgraph in the gap. It reads only ``x`` and has no data
    # dependency on dispatch, so the graph compiler can schedule it
    # concurrently with the EP comms on each device's stream.
    has_unfused_shared = (
        moe_shards[0].has_shared_experts
        and not batch_mgr.config.fused_shared_expert
    )
    overlap_shared_expert = (
        has_unfused_shared and not batch_mgr.config.use_allreduce
    )

    # Decide the MXFP4 EP A-scale preshuffle fold before dispatch because the
    # dispatch scales output shape depends on it. All dispatch paths now thread
    # the fusion parameters. `MoE` handles this as a no-op; `MoEQuantized`
    # enables the fold when applicable.
    for shard in moe_shards:
        shard.configure_ep_scale_fusion(dispatch_supports_fold=True)

    shared_outs: list[TensorValue | None] | None = None

    if batch_mgr.config.use_allreduce:
        # launch per-device dispatch since they don't need to do cross-device
        # communication.
        all_dispatch_results = []
        for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
            shard_mgr = shard.ep_batch_manager
            dispatch_result = shard_mgr.ep_dispatch(
                x,
                all_topk_ids[i],
                device_ids[i],
                input_scales=scales[i] if scales is not None else None,
            )
            all_dispatch_results.append(dispatch_result)
        if has_unfused_shared:
            # All devices hold the same ``x`` (TP attention replicates the
            # input). The caller AllReduces the per-device combine outputs in
            # ``_post_mlp``, so adding ``shared_experts(x)`` on every device
            # would multiply the shared contribution by ``n_devices`` after
            # the reduction. Add it on device 0 only so ``AllReduce.sum``
            # recovers a single copy.
            (shared_dev0,) = _compute_shared_experts(moe_shards, xs, [0])
            shared_outs = [
                shared_dev0 if i == 0 else None for i in range(len(moe_shards))
            ]
    elif overlap_shared_expert:
        # Per-device async dispatch so we can interleave the shared-expert
        # subgraph between launch and wait.
        for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
            shard.ep_batch_manager.ep_dispatch_async(
                x,
                all_topk_ids[i],
                device_ids[i],
                input_scales=scales[i] if scales is not None else None,
            )
        shared_outs = _compute_shared_experts(
            moe_shards, xs, range(len(moe_shards))
        )
        all_dispatch_results = [
            shard.ep_batch_manager.ep_dispatch_wait(device_ids[i])
            for i, shard in enumerate(moe_shards)
        ]
    else:
        # Multi-device dispatch (single op).
        all_dispatch_results = batch_mgr.ep_dispatch_all(
            xs, all_topk_ids, device_ids, input_scales=scales
        )
        if has_unfused_shared:
            shared_outs = _compute_shared_experts(
                moe_shards, xs, range(len(moe_shards))
            )

    # Estimated total token-expert pairs across all devices.
    total_tokens = ops.shape_to_tensor(xs[0].shape)[0]
    for x in xs[1:]:
        total_tokens = total_tokens + ops.shape_to_tensor(x.shape)[0]
    estimated_total_m = (
        total_tokens
        * moe_shards[0].num_experts_per_token
        // batch_mgr.config.n_gpus_per_node
    ).cast(DType.uint32)

    # Per-shard local expert compute.
    all_down_projs: list[TensorValue] = []
    for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
        expert_inputs = all_dispatch_results[i]
        down = shard._local_ep_compute(expert_inputs, x, estimated_total_m)
        all_down_projs.append(down)

    if batch_mgr.config.use_allreduce:
        # launch per-device combine since they don't need to do cross-device
        # communication.
        combine_results: list[TensorValue] = []
        for i, shard in enumerate(moe_shards):
            shard_mgr = shard.ep_batch_manager
            combine_result = shard_mgr.ep_combine(
                all_down_projs[i],
                all_router_weights[i],
                device_ids[i],
                all_topk_ids[i],
            )
            combine_results.append(combine_result)
    elif overlap_shared_expert:
        # Per-device async combine. The shared-expert subgraph was issued
        # earlier between dispatch_async and dispatch_wait; combine_async +
        # combine_wait gives the scheduler a second window to absorb any
        # remaining shared-expert work.
        for i, shard in enumerate(moe_shards):
            shard.ep_batch_manager.ep_combine_async(
                all_down_projs[i], device_ids[i]
            )
        combine_results = [
            shard.ep_batch_manager.ep_combine_wait(
                all_router_weights[i], device_ids[i]
            )
            for i, shard in enumerate(moe_shards)
        ]
    else:
        # Multi-device combine (single op).
        combine_results = batch_mgr.ep_combine_all(
            all_down_projs, all_router_weights, device_ids
        )

    outputs: list[TensorValue] = []
    for i, x in enumerate(xs):
        out = combine_results[i]
        shared_out = shared_outs[i] if shared_outs is not None else None
        if shared_out is not None:
            out += shared_out
        outputs.append(out.cast(x.dtype))

    return outputs


def _accumulate(
    counter_buf: BufferValue,
    router_idx_flat: TensorValue,
    device: DeviceRef,
    num_experts: int,
) -> None:
    """Atomic-equivalent on-GPU histogram via broadcast equality + reduction."""
    counter = ops.buffer_load(counter_buf)

    expert_ids = ops.range(
        start=0,
        stop=num_experts,
        step=1,
        out_dim=num_experts,
        dtype=DType.int32,
        device=device,
    )

    matches = ops.equal(
        ops.unsqueeze(router_idx_flat, axis=-1),
        ops.unsqueeze(expert_ids, axis=0),
    )

    increment_2d = ops.sum(matches.cast(DType.int32), axis=0)

    increment = ops.reshape(increment_2d, [num_experts]).cast(DType.int64)

    ops.buffer_store(counter_buf, counter + increment)


def _eplb_remap(
    router_idx: TensorValue,
    shard: MoE,
    batch_mgr: EPBatchManager,
    layer_idx_t: TensorValue,
) -> TensorValue:
    """Fused EPLB logical-to-physical id remap.

    Single Mojo kernel (``mo.moe.eplb.remap``) that replaces the legacy
    7-op chain (gather logcnt -> range -> mod -> mul + adds -> gather
    log2phy). Caches the current layer's slice of ``logcnt`` and
    ``log2phy`` in SMEM and writes physical ids in one launch.

    Identity-bypassed at graph build time when the EPLB plan is the
    identity permutation. With a non-identity plan we MUST remap even
    at ``max_replicas == 1``, because weights are loaded at permuted
    physical slots and the gate emits logical ids — so a bare
    ``max_replicas`` check would be wrong.

    Returns ``[N, K]`` int32 physical ids.
    """
    if _eplb_is_identity_placement(batch_mgr):
        return router_idx

    device = shard.devices[0]
    num_log = shard.gate.num_experts
    max_replicas = batch_mgr.config.max_replicas

    log2phy = ops.buffer_load(batch_mgr._eplb_log2phy_per_device[device.id])
    logcnt = ops.buffer_load(batch_mgr._eplb_logcnt_per_device[device.id])

    return moe_eplb_remap(
        router_idx=router_idx,
        logcnt=logcnt,
        log2phy=log2phy,
        layer_idx=layer_idx_t,
        num_log=num_log,
        max_replicas=max_replicas,
        n_experts_per_tok=int(router_idx.shape[1]),
        hash_decorrelate=getattr(
            batch_mgr.config, "eplb_hash_decorrelate", True
        ),
    )


def forward_moe_sharded_layers(
    shards: Sequence[Callable[[TensorValue], TensorValue]],
    xs: list[TensorValue],
    eplb_counter_buffers: list[BufferValue] | None = None,
    layer_idx_per_device: list[TensorValue] | None = None,
) -> list[TensorValue]:
    """Forward pass through DP-sharded layers (EP MoE or replicated MLP/MoE).

    For EP-enabled MoE shards this runs the full expert-parallel
    communication path (dispatch -> local compute -> combine).
    For everything else (replicated MLP, non-EP MoE) it falls back to
    :func:`forward_sharded_layers`.

    Args:
        shards: Per-device shard callables (MoE, MLP, etc.).
        xs: Input tensors, one per shard.

    Returns:
        outputs: Output tensors, one per shard.
    """
    first = shards[0]
    if (
        hasattr(first, "_ep_batch_manager")
        and first._ep_batch_manager is not None
    ):
        return _ep_forward(
            cast(list[MoE], list(shards)),
            xs,
            eplb_counter_buffers,
            layer_idx_per_device,
        )
    return forward_sharded_layers(shards, xs)


def _eplb_is_identity_placement(batch_mgr: EPBatchManager) -> bool:
    """True iff EPLB's host-side plan is the identity permutation.

    When True, ``router_idx`` from the gate is already in physical-id
    space and no kernel-side remap is needed.
    """
    plan = batch_mgr._eplb_phy2log
    if plan is None:
        return True

    n_phy = plan.shape[1]
    ident = np.broadcast_to(np.arange(n_phy, dtype=plan.dtype), plan.shape)
    return bool(np.array_equal(plan, ident))
