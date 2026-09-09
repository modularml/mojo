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
"""ops.random tests."""

import pytest
from conftest import tensor_types
from hypothesis import assume, event, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

big_ints = st.one_of(
    st.integers(max_value=-1),
    st.integers(min_value=2**64),
)

UNSUPPORTED_DTYPES = {
    DType.bool,
    DType.float4_e2m1fn,
    DType.float6_e2m3fn,
    DType.float6_e3m2fn,
    DType.float8_e8m0fnu,
}

unsupported_dtypes = st.sampled_from(list(UNSUPPORTED_DTYPES))
supported_dtypes = st.sampled_from(list(set(DType) - UNSUPPORTED_DTYPES))


unsupported_tensor_types = tensor_types(dtypes=unsupported_dtypes)
supported_tensor_types = tensor_types(dtypes=supported_dtypes)


@given(like=...)
def test_random__no_seed(like: TensorType) -> None:
    with Graph("no_seed"):
        with pytest.raises(RuntimeError):
            ops.random.uniform(like)


@given(
    like=supported_tensor_types,
    seed=st.integers(min_value=0, max_value=2**64 - 1),
)
def test_random__static_seed(like: TensorType, seed: int) -> None:
    with Graph("static_seed"):
        ops.random.set_seed(seed)
        result = ops.random.uniform(like)
    assert result.type == like


@given(seed=big_ints)
def test_random__static_seed_out_of_bounds(seed: int) -> None:
    with Graph("static_seed"):
        with pytest.raises((ValueError, OverflowError)):
            ops.random.set_seed(seed)


@given(like=supported_tensor_types)
def test_random__dynamic_seed(like: TensorType) -> None:
    with Graph(
        "dynamic_seed", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.uniform(like)
    assert result.type == like


def test_random__dynamic_seed__bad_seed_dtype() -> None:
    with Graph(
        "bad_seed",
        input_types=[TensorType(DType.float32, [1], DeviceRef.CPU())],
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)


def test_random__dynamic_seed__bad_seed_shape() -> None:
    with Graph(
        "bad_seed", input_types=[TensorType(DType.uint64, [2], DeviceRef.CPU())]
    ) as graph:
        with pytest.raises(ValueError):
            ops.random.set_seed(graph.inputs[0].tensor)


def test_random__seed_type_factory() -> None:
    seed_type = ops.random.SeedType(DeviceRef.CPU())
    assert seed_type.dtype == DType.uint64
    assert list(seed_type.shape) == [1]
    assert seed_type.device == DeviceRef.CPU()


def test_random__rank0_uint64_normalized_to_rank1() -> None:
    with Graph(
        "rank0_seed",
        input_types=[TensorType(DType.uint64, [], DeviceRef.CPU())],
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.uniform(
            TensorType(DType.float32, [4], DeviceRef.CPU())
        )
    assert list(result.type.shape) == [4]


def test_random__set_seed_int_then_op_on_other_device() -> None:
    with Graph("set_seed_int_gpu_op"):
        ops.random.set_seed(0)
        result = ops.random.uniform(
            TensorType(DType.float32, [4], DeviceRef.GPU())
        )
    assert result.type.device == DeviceRef.GPU()


@given(like=supported_tensor_types)
def test_gaussian(like: TensorType) -> None:
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.gaussian(like)
    assert result.type == like


@given(like=..., mean=..., std=...)
def test_gaussian__parameterized(
    like: TensorType, mean: float, std: float
) -> None:
    assume(std != 0)
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.gaussian(like, mean=mean, std=std)
    assert result.type == like


@pytest.mark.skip("GEX-2100")
@given(like=supported_tensor_types, mean=...)
def test_gaussian__zero_std(like: TensorType, mean: float) -> None:
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        with pytest.raises(Exception):
            ops.random.gaussian(like, mean=mean, std=0)


@given(like=supported_tensor_types)
def test_uniform(like: TensorType) -> None:
    with Graph(
        "uniform", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.uniform(like)
    assert result.type == like


@given(like=supported_tensor_types, range=...)
def test_uniform__parameterized(
    like: TensorType, range: tuple[int, int]
) -> None:
    assume(range[0] < range[1])
    with Graph(
        "uniform", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        try:
            result = ops.random.uniform(like, range=range)
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("Range out of bounds")
        else:
            event("Uniform with valid range")
            assert result.type == like


@pytest.mark.skip("GEX-2100")
@given(like=supported_tensor_types, lower=...)
def test_uniform__zero_range(like: TensorType, lower: float) -> None:
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        with pytest.raises(Exception):
            ops.random.uniform(like, range=(lower, lower))


@pytest.mark.skip("GEX-2100")
@given(like=supported_tensor_types, range=...)
def test_uniform__inverted_range(
    like: TensorType, range: tuple[float, float]
) -> None:
    assume(range[0] > range[1])
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        with pytest.raises(Exception):
            ops.random.uniform(like, range=range)


@given(like=tensor_types(dtypes=supported_dtypes, device=DeviceRef.GPU()))
def test_gaussian__gpu_support(like: TensorType) -> None:
    with Graph(
        "gaussian", input_types=[ops.random.SeedType(like.device)]
    ) as graph:
        ops.random.set_seed(graph.inputs[0].tensor)
        result = ops.random.gaussian(like)
    assert result.type == like
