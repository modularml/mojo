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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType, ops

DEVICE = Accelerator() if accelerator_count() else CPU()


def test_allgather() -> None:
    input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    signal_buffer = Tensor.zeros([], device=DEVICE)
    [result] = F.functional(ops.allgather)([input], [signal_buffer])
    assert result.real


def test_allreduce() -> None:
    input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    signal_buffer = Tensor.zeros([], device=DEVICE)
    [result] = F.functional(ops.allreduce.sum)([input], [signal_buffer])
    assert result.real


def test_as_interleaved_complex() -> None:
    # needs even last dimension
    complex_input = Tensor.ones([2, 4], dtype=DType.float32, device=DEVICE)
    result = F.as_interleaved_complex(complex_input)
    assert result.real


def test_avg_pool2d() -> None:
    # needs 4D input with NHWC format
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.avg_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    assert result.real


def test_band_part() -> None:
    # needs at least 2D tensor
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.band_part(tensor_2d, num_lower=-1, num_upper=0)
    assert result.real


def test_broadcast_to() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.broadcast_to(tensor_2d, [4, 6])
    assert result.real


def test_broadcast_to_expand() -> None:
    """Test broadcast_to expanding a dimension of size 1."""
    tensor = Tensor.ones([3, 1], dtype=DType.float32, device=DEVICE)
    result = F.broadcast_to(tensor, [3, 4])
    assert result.real
    assert result.shape == [3, 4]


def test_broadcast_to_add_leading_dim() -> None:
    """Test broadcast_to adding a new leading dimension."""
    tensor = Tensor.ones([3, 4], dtype=DType.float32, device=DEVICE)
    result = F.broadcast_to(tensor, [2, 3, 4])
    assert result.real
    assert result.shape == [2, 3, 4]


def test_broadcast_to_incompatible_shapes_error() -> None:
    """Test broadcast_to raises error for incompatible shapes."""
    # Trying to broadcast [4, 6] to [3, 6] should fail because 4 != 3 and 4 != 1
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    with pytest.raises(
        ValueError,
        match=r"input dimension.*must be either 1 or equal",
    ):
        F.broadcast_to(tensor, [3, 6])


def test_tensor_broadcast_to_method() -> None:
    """Test the Tensor.broadcast_to() method."""
    tensor = Tensor.ones([3, 1], dtype=DType.float32, device=DEVICE)
    result = tensor.broadcast_to([3, 4])
    assert result.real
    assert result.shape == [3, 4]


def test_cast() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.cast(tensor_2d, DType.int64)
    assert result.real
    assert result.dtype == DType.int64


def test_chunk() -> None:
    # split into 2 chunks along axis 0
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    chunks = F.chunk(tensor_2d, chunks=2, axis=0)
    for chunk in chunks:
        assert chunk.real


def test_complex_mul() -> None:
    lhs = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    rhs = Tensor.ones([2, 1, 2], dtype=DType.float32, device=DEVICE)
    result = F.complex_mul(lhs, rhs)
    assert result.shape == [2, 2, 2]
    assert result.real


def test_concat() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.concat([tensor_2d, tensor_2d], axis=0)
    assert result.real
    assert result.shape.static_dims == [8, 6]


@pytest.mark.unique_shard
def test_cond_true_branch() -> None:
    pred = Tensor.full([], True, dtype=DType.bool, device=DEVICE)

    def then_fn():  # noqa: ANN202
        return F.constant(42.0, DType.float32, DEVICE)

    def else_fn():  # noqa: ANN202
        return F.constant(0.0, DType.float32, DEVICE)

    results = F.cond(
        pred,
        [TensorType(DType.float32, [], device=DEVICE)],
        then_fn,
        else_fn,
    )
    assert len(results) == 1
    result_cpu = results[0].to(CPU())
    np.testing.assert_equal(np.from_dlpack(result_cpu), 42.0)


@pytest.mark.unique_shard
def test_cond_false_branch() -> None:
    pred = Tensor.full([], False, dtype=DType.bool, device=DEVICE)

    def then_fn():  # noqa: ANN202
        return F.constant(42.0, DType.float32, DEVICE)

    def else_fn():  # noqa: ANN202
        return F.constant(0.0, DType.float32, DEVICE)

    results = F.cond(
        pred,
        [TensorType(DType.float32, [], device=DEVICE)],
        then_fn,
        else_fn,
    )
    assert len(results) == 1
    result_cpu = results[0].to(CPU())
    np.testing.assert_equal(np.from_dlpack(result_cpu), 0.0)


@pytest.mark.unique_shard
def test_cond_with_functional_ops_in_branches() -> None:
    """Branches using F.xxx operations return Tensor, which must be converted."""
    pred = Tensor.full([], True, dtype=DType.bool, device=DEVICE)
    x = Tensor.ones([4], dtype=DType.float32, device=DEVICE)

    def then_fn():  # noqa: ANN202
        return F.mul(x, F.constant(2.0, DType.float32, DEVICE))

    def else_fn():  # noqa: ANN202
        return F.negate(x)

    results = F.cond(
        pred,
        [TensorType(DType.float32, [4], device=DEVICE)],
        then_fn,
        else_fn,
    )
    assert len(results) == 1
    result_cpu = results[0].to(CPU())
    expected = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(np.from_dlpack(result_cpu), expected)


def test_cond_no_results() -> None:
    pred = Tensor.full([], True, dtype=DType.bool, device=DEVICE)

    def then_fn() -> None:
        _ = F.constant(1.0, DType.float32, DEVICE)

    def else_fn() -> None:
        _ = F.constant(0.0, DType.float32, DEVICE)

    results = F.cond(pred, None, then_fn, else_fn)
    assert len(results) == 0


@pytest.mark.unique_shard
def test_while_loop_tensor_only_surface() -> None:
    """``F.while_loop`` exposes a Tensor-only surface (MXF-486 / MXF-435).

    The ``predicate`` and ``body`` callbacks receive and return ``Tensor``,
    never ``TensorValue``. This exercises both crossings inside
    ``_while_loop_graph``: the inbound ``TensorValue -> Tensor`` wrap before
    the callbacks run, and the outbound ``Tensor -> TensorValue`` coercion
    afterwards. Regression test: the ActionSet dispatcher cutover (#86904)
    dropped the inbound wrap, so a callback written to the documented Tensor
    contract (``x < 10``) raised ``AttributeError`` at graph-construction
    time. The ``isinstance`` assertions pin the inbound crossing directly.
    """
    x = Tensor.full([], 0, dtype=DType.int32, device=DEVICE)

    def predicate(x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        # ``ops.while_loop`` requires the predicate result on CPU.
        return (x < 10).to(CPU())

    def body(x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        return x + 1

    (result,) = F.while_loop(x, predicate, body)
    assert isinstance(result, Tensor)
    result_cpu = result.to(CPU())
    np.testing.assert_equal(np.from_dlpack(result_cpu), 10)


@pytest.mark.unique_shard
def test_while_loop_multiple_args_tensor_only_surface() -> None:
    """``F.while_loop`` Tensor-only surface holds for multi-arg callbacks too."""
    x = Tensor.full([], 0, dtype=DType.int32, device=DEVICE)
    y = Tensor.full([], 0, dtype=DType.int32, device=DEVICE)

    def predicate(x: Tensor, y: Tensor) -> Tensor:
        assert isinstance(x, Tensor) and isinstance(y, Tensor)
        return (x < 10).to(CPU())

    def body(x: Tensor, y: Tensor) -> list[Tensor]:
        assert isinstance(x, Tensor) and isinstance(y, Tensor)
        return [x + 1, y + 2]

    x_out, y_out = F.while_loop((x, y), predicate, body)
    np.testing.assert_equal(np.from_dlpack(x_out.to(CPU())), 10)
    np.testing.assert_equal(np.from_dlpack(y_out.to(CPU())), 20)


def test_constant() -> None:
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.constant(1.0, DType.float32, device_ref)
    assert result.real


def test_conv2d() -> None:
    # NHWC input: [batch, height, width, in_channels]
    # RSCF filter: [height, width, in_channels/groups, out_channels]
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    weight = Tensor.ones(
        [3, 3, 2, 1], dtype=DType.float32, device=DEVICE
    )  # [H, W, in_ch, out_ch]
    result = F.conv2d(tensor_4d, weight)
    assert result.real


@pytest.mark.skip("KERNELS-1975")
def test_conv2d_transpose() -> None:
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    weight = Tensor.ones(
        [3, 3, 1, 2], dtype=DType.float32, device=DEVICE
    )  # [H, W, out_ch, in_ch]
    result = F.conv2d_transpose(tensor_4d, weight)
    assert result.real


@pytest.mark.unique_shard
def test_conv3d() -> None:
    # NDHWC input: [batch, depth, height, width, in_channels]
    # QRSCF filter: [depth, height, width, in_channels/groups, out_channels]
    tensor_5d = Tensor.ones(
        [1, 2, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, D, H, W, C]
    weight_3d = Tensor.ones(
        [2, 3, 3, 2, 1], dtype=DType.float32, device=DEVICE
    )  # [D, H, W, in_ch, out_ch]
    result = F.conv3d(tensor_5d, weight_3d)
    assert result.real


def test_flatten() -> None:
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.flatten(tensor_3d, start_dim=1, end_dim=2)
    assert result.real


@pytest.mark.unique_shard
def test_fold() -> None:
    # needs shape [N, C * kernel_size[0] * kernel_size[1], L]
    # For kernel_size=[2, 2], we need C * 4 channels
    kernel_size = [2, 2]
    tensor_3d = Tensor.ones(
        [1, 4, 4], dtype=DType.float32, device=DEVICE
    )  # [N, C*kernel_prod, L]
    result = F.fold(tensor_3d, output_size=[3, 3], kernel_size=kernel_size)
    assert result.real


def test_gather() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices = Tensor.full([2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather(tensor_2d, indices, axis=0)
    assert result.real


def test_gather_nd() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    result = F.gather_nd(tensor_2d, indices_nd)
    assert result.real


def test_hann_window() -> None:
    result = F.hann_window(4, device=DEVICE)
    assert result.real


@pytest.mark.skipif(
    isinstance(DEVICE, CPU), reason="IRFFT only supported on GPU"
)
def test_irfft() -> None:
    tensor_2d = Tensor.ones([4, 8], dtype=DType.float32, device=DEVICE)
    result = F.irfft(tensor_2d, n=14, axis=-1)
    assert result.real


def test_layer_norm() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    gamma = Tensor.ones(
        [6], dtype=DType.float32, device=DEVICE
    )  # normalization weights
    beta = Tensor.zeros(
        [6], dtype=DType.float32, device=DEVICE
    )  # normalization bias
    result = F.layer_norm(tensor_2d, gamma, beta, epsilon=1e-5)
    assert result.real


def test_masked_scatter() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    mask = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    source = Tensor.ones([24], dtype=DType.float32, device=DEVICE)
    result = F.masked_scatter(tensor_2d, mask, source, out_dim=24)
    assert result.real


def test_matmul() -> None:
    a = Tensor.ones([4, 3], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([3, 6], dtype=DType.float32, device=DEVICE)
    result = F.matmul(a, b)
    assert result.real


def test_max_pool2d() -> None:
    tensor_4d = Tensor.ones(
        [1, 4, 4, 2], dtype=DType.float32, device=DEVICE
    )  # [N, H, W, C]
    result = F.max_pool2d(tensor_4d, kernel_size=(2, 2), stride=1)
    assert result.real


def test_nonzero() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.nonzero(
        tensor_2d, out_dim=24
    )  # assuming all elements are nonzero
    assert result.real


def test_outer() -> None:
    vec_a = Tensor.ones([3], dtype=DType.float32, device=DEVICE)
    vec_b = Tensor.ones([4], dtype=DType.float32, device=DEVICE)
    result = F.outer(vec_a, vec_b)
    assert result.real


def test_pad() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.pad(tensor_2d, [1, 1, 2, 2])  # pad left, right, top, bottom
    assert result.real


def test_permute() -> None:
    tensor_3d = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = F.permute(tensor_3d, [2, 0, 1])
    assert result.real


def test_print(capfd: pytest.CaptureFixture[str]) -> None:
    # A realized (eager) tensor has concrete storage and no graph value, so
    # `F.print` echoes its value directly to stdout.
    tensor = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    assert tensor.real
    F.print(tensor, "eager_tensor")
    assert "Tensor eager_tensor=" in capfd.readouterr().out


def test_print_symbolic(capfd: pytest.CaptureFixture[str]) -> None:
    # A symbolic (graph) tensor has no concrete storage, so `F.print` records a
    # print op that fires when the compiled graph runs.
    @module_dataclass
    class PrintModule(Module[[Tensor], Tensor]):
        def forward(self, x: Tensor) -> Tensor:
            F.print(x, "graph_tensor")
            return x

    input_type = TensorType(
        DType.float32, [2, 2], device=DeviceRef.from_device(DEVICE)
    )
    compiled = PrintModule().compile(input_type)
    compiled(Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE))
    assert "graph_tensor" in capfd.readouterr().out


def test_arange() -> None:
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.arange(0, 10, 1, dtype=DType.int32, device=device_ref)
    assert result.real


@pytest.mark.unique_shard
def test_repeat_interleave() -> None:
    # repeat_interleave not supported on GPU, use CPU
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=CPU())
    result = F.repeat_interleave(tensor_2d, 2, axis=0)
    assert result.real


def test_reshape() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.reshape(tensor_2d, [6, 4])
    assert result.real


# FIXME: Crashes if not in its own shard
@pytest.mark.unique_shard
def test_scatter() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_scatter = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones([2, 2], dtype=DType.float32, device=DEVICE)
    result = F.scatter(
        tensor_2d, source_scatter, indices_scatter, axis=0
    )  # updates, indices order
    assert result.real


def test_scatter_nd() -> None:
    # Create input tensor to scatter into
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    indices_nd = Tensor.full([2, 2], 0, dtype=DType.int64, device=DEVICE)
    source_scatter = Tensor.ones(
        [2], dtype=DType.float32, device=DEVICE
    )  # match indices shape
    result = F.scatter_nd(tensor_2d, source_scatter, indices_nd)
    assert result.real


def test_slice_tensor() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.slice_tensor(tensor_2d, [slice(0, 2), slice(1, 4)])
    assert result.real


def test_slice_tensor_negative_int_index() -> None:
    """Regression test for MXF-243: hidden[:, -1] fails at compile time."""
    hidden = Tensor.ones((2, 4, 8), dtype=DType.float32, device=CPU())
    last_token = hidden[:, -1]
    assert last_token.shape == [2, 8]
    np.testing.assert_allclose(np.from_dlpack(last_token), 1.0)

    second_to_last = hidden[:, -2]
    assert second_to_last.shape == [2, 8]
    np.testing.assert_allclose(np.from_dlpack(second_to_last), 1.0)

    last_batch = hidden[-1]
    assert last_batch.shape == [4, 8]
    np.testing.assert_allclose(np.from_dlpack(last_batch), 1.0)


def test_split() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    splits = cast(
        list[Tensor], F.split(tensor_2d, [2, 2], axis=0)
    )  # split_sizes as sequence
    for split_tensor in splits:
        assert split_tensor.real


def test_squeeze() -> None:
    # needs tensor with size-1 dimension
    tensor_with_one = Tensor.ones([4, 1, 6], dtype=DType.float32, device=DEVICE)
    result = F.squeeze(tensor_with_one, axis=1)
    assert result.real


def test_stack() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    tensors_to_stack = [tensor_2d, tensor_2d]
    result = F.stack(tensors_to_stack, axis=0)
    assert result.real


def test_tile() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.tile(tensor_2d, [2, 1])
    assert result.real


def test_top_k() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    values, indices = F.top_k(tensor_2d, k=3, axis=-1)
    assert values.real
    assert indices.real


def test_transfer_to() -> None:
    # transfer to same device (should be no-op)
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    device_ref = DeviceRef.from_device(DEVICE)
    result = F.transfer_to(tensor_2d, device_ref)
    assert result.real


def test_transpose() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.transpose(tensor_2d, axis_1=0, axis_2=1)
    assert result.real


def test_unsqueeze() -> None:
    tensor_2d = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.unsqueeze(tensor_2d, axis=1)
    assert result.real


def test_where() -> None:
    condition = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    x = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    y = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = F.where(condition, x, y)
    assert result.real


def test_functional_returns_tensor() -> None:
    @F.functional
    def returns_tensor():  # noqa: ANN202
        return Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)

    result = returns_tensor()
    assert result.real


@pytest.mark.skip(
    reason="reduce_op axis=None multi-dim reduction needs debugging"
)
def test_sum_axis_none() -> None:
    """Test that F.sum with axis=None reduces over all dimensions."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor = Tensor(data, dtype=DType.float32, device=DEVICE)
    result = F.sum(tensor, axis=None)
    assert isinstance(result, Tensor)
    result._sync_realize()
    assert result.shape == [1, 1]  # keepdim=True reduces each axis to 1
    expected_sum = sum(sum(row) for row in data)  # 21.0
    result_value = result.item()
    assert abs(result_value - expected_sum) < 1e-5


@pytest.mark.skip(
    reason="reduce_op axis=None multi-dim reduction needs debugging"
)
def test_min_axis_none() -> None:
    """Test that F.min with axis=None reduces over all dimensions."""
    data = [[1.2, 3.5, 2.1], [2.3, 1.9, 4.2]]
    tensor = Tensor(data, dtype=DType.float32, device=DEVICE)
    result = F.min(tensor, axis=None)
    assert isinstance(result, Tensor)
    result._sync_realize()
    assert result.shape == [1, 1]
    expected_min = 1.2
    result_value = result.item()
    assert abs(result_value - expected_min) < 1e-5


@pytest.mark.skip(
    reason="reduce_op axis=None multi-dim reduction needs debugging"
)
def test_argmin_axis_none() -> None:
    """Test that F.argmin with axis=None returns flattened index."""
    data = [[1.2, 3.5, 2.1], [2.3, 1.9, 4.2]]
    tensor = Tensor(data, dtype=DType.float32, device=DEVICE)
    result = F.argmin(tensor, axis=None)
    assert isinstance(result, Tensor)
    result._sync_realize()
    assert result.shape == [1, 1]
    # argmin reduces per-axis: first axis 1 → argmin in each row,
    # then axis 0 → argmin among those. Result is index within last reduction.
    result_value = result.item()
    assert isinstance(result_value, int)


@pytest.mark.skip(
    reason="reduce_op axis=None multi-dim reduction needs debugging"
)
def test_axis_none_preserves_default_behavior() -> None:
    """Test that default axis=-1 behavior is unchanged."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor = Tensor(data, dtype=DType.float32, device=DEVICE)
    # Default behavior (axis=-1)
    result_default = F.sum(tensor)
    assert isinstance(result_default, Tensor)
    result_default._sync_realize()
    # Explicit axis=-1
    result_explicit = F.sum(tensor, axis=-1)
    assert isinstance(result_explicit, Tensor)
    result_explicit._sync_realize()
    # Both should give same result: [6.0, 15.0]
    assert result_default.shape == result_explicit.shape == [2, 1]
    # Should be different from axis=None
    result_none = F.sum(tensor, axis=None)
    assert isinstance(result_none, Tensor)
    result_none._sync_realize()
    assert result_none.shape == [1, 1]
    assert result_none.shape != result_default.shape
    # Verify axis=None gives total sum
    assert abs(result_none.item() - 21.0) < 1e-5
