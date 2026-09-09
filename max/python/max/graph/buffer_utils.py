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

"""Provides on-device dtype casting utilities for :obj:`max.driver.Buffer` tensors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max._core.engine import Model
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession  # type: ignore
from max.graph import DeviceRef, Graph, TensorType

# Per-session, per-(src,dst,device) compiled cast models.
_CAST_MODELS: dict[tuple[int, str, int, int, int], Model] = {}
# Lazily-created sessions when one is not provided.
_SESSION_CACHE: dict[tuple[str, int], InferenceSession] = {}


def _get_or_create_session(
    device: Device, session: InferenceSession | None
) -> InferenceSession:
    if session is not None:
        return session
    key = (device.label, device.id)
    s = _SESSION_CACHE.get(key)
    if s is None:
        s = InferenceSession(devices=[device])
        _SESSION_CACHE[key] = s
    return s


def _get_or_create_cast_model(
    old_dtype: DType,
    new_dtype: DType,
    device: Device,
    session: InferenceSession,
) -> Model:
    # Include session id in the cache key: models are tied to the session they were loaded into.
    key: tuple[int, str, int, int, int] = (
        id(session),
        str(old_dtype),
        device.id,
        hash(device.label),
        hash(str(new_dtype)),
    )
    model = _CAST_MODELS.get(key)
    if model is None:
        with Graph(
            "cast",
            input_types=[
                TensorType(
                    dtype=old_dtype,
                    shape=["dim"],
                    device=DeviceRef.from_device(device),
                )
            ],
        ) as graph:
            graph.output(graph.inputs[0].tensor.cast(new_dtype))
        model = session.load(graph)
        _CAST_MODELS[key] = model
    return model


def cast_tensor_to(
    tensor: Buffer,
    new_dtype: DType,
    session: InferenceSession | None = None,
) -> Buffer:
    """Cast a tensor to a new dtype on-device (no host round-trips).

    If a session is provided, reuse it (recommended inside pipelines).
    Otherwise a tiny per-device session is created/cached lazily.
    """
    if tensor.dtype == new_dtype:
        return tensor

    # The graph compiler rejects f16 graphs on aarch64 host targets (see
    # ValidateDevicesPass), so on host devices hop f16 through numpy, which
    # supports float16 natively, and graph-cast only the f32 leg if needed.
    if tensor.device.is_host and DType.float16 in (tensor.dtype, new_dtype):
        if tensor.dtype == DType.float16:
            f32 = Buffer.from_numpy(
                np.from_dlpack(tensor).astype(np.float32)
            ).to(tensor.device)
            return cast_tensor_to(f32, new_dtype, session=session)
        f32 = cast_tensor_to(tensor, DType.float32, session=session)
        return Buffer.from_numpy(np.from_dlpack(f32).astype(np.float16)).to(
            tensor.device
        )

    sess = _get_or_create_session(tensor.device, session)
    model = _get_or_create_cast_model(
        tensor.dtype, new_dtype, tensor.device, sess
    )

    flat = tensor.view(tensor.dtype, [tensor.num_elements]).to(tensor.device)
    out = model(flat)[0]
    assert isinstance(out, Buffer)
    return out.view(new_dtype, tensor.shape)


def cast_dlpack_to(
    raw_tensor: DLPackArray,
    old_dtype: DType,
    new_dtype: DType,
    device: Device,
    session: InferenceSession | None = None,
) -> Buffer:
    """Wrap a DLPack array then cast it to the requested dtype on the given device."""
    t = Buffer.from_dlpack(raw_tensor)
    if t.dtype != old_dtype:
        t = t.view(old_dtype, t.shape)
    t = t.to(device)
    return cast_tensor_to(t, new_dtype, session=session)


def cast_tensors_to(
    tensors: Sequence[Buffer] | None,
    new_dtype: DType,
    session: InferenceSession | None = None,
) -> list[Buffer]:
    """Cast a sequence of tensors to the requested dtype on their current devices."""
    if not tensors:
        return []
    return [cast_tensor_to(t, new_dtype, session=session) for t in tensors]
