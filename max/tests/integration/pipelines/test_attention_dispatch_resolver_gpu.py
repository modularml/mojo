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

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from max.nn.kv_cache.utils import MHAAttnKey

N_KV_HEADS = 8
# Every head count ``compute_mla_dispatch_scalars_runtime`` enumerates. A count
# missing an arm raises out of both the resolver and the reference graph, so
# this list is what keeps the enumeration reachable end to end; that the arm
# forwards to the specialization it names is pinned on the Mojo side, in
# ``max/kernels/test/gpu/nn/test_mla_decode_split_policy.mojo``.
MLA_NUM_HEADS = [8, 12, 16, 24, 32, 48, 64, 128]
TEST_CASES = [
    (1, 17, 128),
    (4, 17, 768),
    (16, 17, 2048),
    (64, 17, 8192),
]
MLA_TEST_CASES = [
    (1, 1, 128),
    (16, 1, 2048),
    (64, 1, 8192),
]


def _build_reference_mha_model(
    session: InferenceSession, device: DeviceRef, n_kv_heads: int
) -> Model:
    with Graph(
        "decode_num_partitions_reference",
        input_types=[
            TensorType(DType.int64, shape=[2], device=DeviceRef.CPU())
        ],
    ) as graph:
        (request,) = graph.inputs
        (num_partitions,) = ops.custom(
            "mo.mha.decode.get_num_partitions",
            values=[request.tensor],
            out_types=[
                TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())
            ],
            parameters={"n_kv_heads": n_kv_heads},
            device=device,
        )
        graph.output(num_partitions.tensor)
    return session.load(graph)


def _resolve_reference(
    model: Model, batch_size: int, max_cache_valid_length: int
) -> int:
    request = Buffer.from_numpy(
        np.array([batch_size, max_cache_valid_length], dtype=np.int64)
    )
    (output,) = model(request)
    return int(output.to_numpy()[0])


def _build_reference_mla_model(
    session: InferenceSession,
    device: DeviceRef,
    num_heads: int,
    *,
    is_fp8_kv: bool = False,
) -> Model:
    with Graph(
        "mla_dispatch_args_reference",
        input_types=[
            TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
            TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
            TensorType(DType.int64, shape=[1], device=DeviceRef.CPU()),
        ],
    ) as graph:
        batch_size_val = graph.inputs[0].tensor
        max_cache_val = graph.inputs[1].tensor
        q_max_seq_len_val = graph.inputs[2].tensor
        (scalars,) = ops.custom(
            "mo.mla.compute_dispatch_args.scalar",
            device=device,
            values=[batch_size_val, max_cache_val, q_max_seq_len_val],
            out_types=[
                TensorType(shape=[3], dtype=DType.int64, device=DeviceRef.CPU())
            ],
            parameters={"num_heads": num_heads, "is_fp8_kv": is_fp8_kv},
        )
        graph.output(scalars.tensor)
    return session.load(graph)


def _resolve_mla_reference(
    model: Model,
    batch_size: int,
    max_prompt_length: int,
    max_cache_valid_length: int,
) -> np.ndarray:
    (output,) = model(
        Buffer.from_numpy(np.array([batch_size], dtype=np.int64)),
        Buffer.from_numpy(np.array([max_cache_valid_length], dtype=np.int64)),
        Buffer.from_numpy(np.array([max_prompt_length], dtype=np.int64)),
    )
    return output.to_numpy()


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(devices=[CPU(), Accelerator()])


@pytest.fixture(scope="module")
def gpu_device_ref() -> DeviceRef:
    return DeviceRef.GPU()


@pytest.fixture(scope="module", params=MLA_NUM_HEADS)
def mla_num_heads(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(scope="module")
def mha_params(
    gpu_device_ref: DeviceRef,
) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=128,
        num_layers=1,
        devices=[gpu_device_ref],
        n_kv_heads=N_KV_HEADS,
    )


@pytest.fixture(scope="module")
def mla_params(
    gpu_device_ref: DeviceRef,
    mla_num_heads: int,
) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=[gpu_device_ref],
        num_q_heads=mla_num_heads,
    )


@pytest.fixture(scope="module")
def mla_params_fp8(
    gpu_device_ref: DeviceRef, mla_num_heads: int
) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        devices=[gpu_device_ref],
        num_q_heads=mla_num_heads,
    )


@pytest.fixture(scope="module")
def reference_mha_model(
    gpu_session: InferenceSession, gpu_device_ref: DeviceRef
) -> Model:
    return _build_reference_mha_model(
        gpu_session, gpu_device_ref, n_kv_heads=N_KV_HEADS
    )


@pytest.fixture(scope="module")
def reference_mla_model(
    gpu_session: InferenceSession,
    gpu_device_ref: DeviceRef,
    mla_num_heads: int,
) -> Model:
    return _build_reference_mla_model(
        gpu_session,
        gpu_device_ref,
        num_heads=mla_num_heads,
    )


@pytest.fixture(scope="module")
def reference_mla_model_fp8(
    gpu_session: InferenceSession,
    gpu_device_ref: DeviceRef,
    mla_num_heads: int,
) -> Model:
    return _build_reference_mla_model(
        gpu_session, gpu_device_ref, num_heads=mla_num_heads, is_fp8_kv=True
    )


@pytest.mark.parametrize(
    ("batch_size", "max_prompt_length", "max_cache_valid_length"),
    TEST_CASES,
)
def test_mha_dispatch_resolver_matches_reference_graph(
    mha_params: MHAKVCacheParams,
    reference_mha_model: Model,
    batch_size: int,
    max_prompt_length: int,
    max_cache_valid_length: int,
) -> None:
    expected_num_partitions = _resolve_reference(
        reference_mha_model, batch_size, max_cache_valid_length
    )

    key = mha_params.resolve_attn_key(
        batch_size, max_prompt_length, max_cache_valid_length
    )
    assert isinstance(key, MHAAttnKey)
    assert key.num_partitions == expected_num_partitions

    # MHA packs a 4-int CPU buffer ending in max_cache_valid_length.
    metadata = key.pack_into_buffer(CPU(), max_cache_valid_length).to_numpy()
    np.testing.assert_array_equal(
        metadata,
        np.array(
            [
                batch_size,
                max_prompt_length,
                expected_num_partitions,
                max_cache_valid_length,
            ],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    ("batch_size", "max_prompt_length", "max_cache_valid_length"),
    MLA_TEST_CASES,
)
def test_mla_dispatch_resolver_matches_reference_graph(
    mla_params: MLAKVCacheParams,
    reference_mla_model: Model,
    batch_size: int,
    max_prompt_length: int,
    max_cache_valid_length: int,
) -> None:
    key = mla_params.resolve_attn_key(
        batch_size, max_prompt_length, max_cache_valid_length
    )
    # MLA packs a 3-int buffer on the accelerator.
    np.testing.assert_array_equal(
        key.pack_into_buffer(Accelerator(), max_cache_valid_length).to_numpy(),
        _resolve_mla_reference(
            reference_mla_model,
            batch_size,
            max_prompt_length,
            max_cache_valid_length,
        ),
    )


@pytest.mark.parametrize(
    ("batch_size", "max_prompt_length", "max_cache_valid_length"),
    MLA_TEST_CASES,
)
def test_mla_fp8_dispatch_resolver_matches_reference_graph(
    mla_params_fp8: MLAKVCacheParams,
    reference_mla_model_fp8: Model,
    batch_size: int,
    max_prompt_length: int,
    max_cache_valid_length: int,
) -> None:
    key = mla_params_fp8.resolve_attn_key(
        batch_size, max_prompt_length, max_cache_valid_length
    )
    np.testing.assert_array_equal(
        key.pack_into_buffer(Accelerator(), max_cache_valid_length).to_numpy(),
        _resolve_mla_reference(
            reference_mla_model_fp8,
            batch_size,
            max_prompt_length,
            max_cache_valid_length,
        ),
    )
