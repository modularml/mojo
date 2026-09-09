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
"""Tests for Gemma4 vision pooling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    VISION_DEFAULT_OUTPUT_LENGTH,
    VISION_HIDDEN_SIZE,
    VISION_POOLING_KERNEL_SIZE,
    TorchGemma4VisionPooler,
)
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.vision_model.pooling import (
    Gemma4VisionPooler,
    avg_pool_by_positions,
    compute_pool_gather_index,
)

TORCH_DTYPE = torch.bfloat16

_INPUT_SEQ_LEN = VISION_DEFAULT_OUTPUT_LENGTH * VISION_POOLING_KERNEL_SIZE**2
_GRID_W = 42
_GRID_H = 60


def _make_grid(w: int, h: int) -> np.ndarray:
    """Build a regular (x, y) patch-position grid, shape [w*h, 2]."""
    return np.array(
        [[x, y] for y in range(h) for x in range(w)],
        dtype=np.int32,
    )


def _buf_to_torch(buf: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buf).cpu().float()


# ---------------------------------------------------------------------------
# avg_pool_by_positions (NumPy) vs torch reference
# ---------------------------------------------------------------------------


class TestAvgPoolByPositions:
    """Tests for the NumPy avg_pool_by_positions function."""

    @pytest.mark.parametrize(
        "grid_w, grid_h, output_length, k",
        [
            (6, 6, 4, 3),
            (3, 3, 1, 3),
            (3, 6, 2, 3),
        ],
        ids=[
            "6x6-to-2x2",
            "3x3-to-1x1",
            "3x6-to-1x2",
        ],
    )
    def test_weights_match_torch_reference(
        self, grid_w: int, grid_h: int, output_length: int, k: int
    ) -> None:
        """NumPy weights must match the torch reference _avg_pool_by_positions."""
        hidden_size = 8
        input_seq_len = grid_w * grid_h

        positions_2d = _make_grid(grid_w, grid_h)
        assert positions_2d.shape == (input_seq_len, 2)

        np_weights = avg_pool_by_positions([positions_2d], [output_length], k)

        torch_ref = TorchGemma4VisionPooler(hidden_size, output_length)
        x_eye = torch.eye(input_seq_len, dtype=torch.float32).unsqueeze(0)
        positions_torch = torch.from_numpy(positions_2d).unsqueeze(0)
        ref_output, _ = torch_ref._avg_pool_by_positions(
            x_eye, positions_torch, output_length
        )
        ref_weights = ref_output[0].numpy()

        np.testing.assert_allclose(np_weights, ref_weights, atol=1e-6)

    def test_no_pooling_path(self) -> None:
        """When k=1, weights are identity (no spatial pooling)."""
        seq_len = 9
        hidden_size = 8

        side = int(seq_len**0.5)
        positions_np = _make_grid(side, side)

        weights = avg_pool_by_positions([positions_np], [seq_len], k=1)

        # k=1 means each patch maps 1:1 to an output bin.
        # Weight matrix should be a permutation with values 1/1**2=1.0.
        assert weights.shape == (seq_len, seq_len)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)

        # Verify our weights give the same hidden_states (scaled by
        # sqrt(hidden_size)) as the torch reference passthrough path.
        torch.manual_seed(99)
        x = torch.randn(1, seq_len, hidden_size, dtype=TORCH_DTYPE)

        # Compute np_pooled BEFORE the torch reference, because the torch
        # passthrough path uses ``hidden_states *= root_hidden_size`` which
        # mutates x in-place.
        np_pooled = (
            torch.from_numpy(weights).float()
            @ x[0].float()
            * math.sqrt(hidden_size)
        ).to(TORCH_DTYPE)

        torch_pooler = TorchGemma4VisionPooler(hidden_size, seq_len)
        padding = torch.ones(1, seq_len, dtype=torch.bool)
        positions_torch = torch.from_numpy(positions_np).unsqueeze(0)
        ref_output, _ref_mask = torch_pooler(
            x, positions_torch, padding, seq_len
        )

        torch.testing.assert_close(
            np_pooled, ref_output[0], rtol=2e-2, atol=2e-2
        )

    def test_non_square_grid(self) -> None:
        """Weights must be correct for a non-square rectangular grid."""
        # 6 wide * 4 tall = 24 patches, k=2, output = 24/4 = 6 bins.
        grid_w, grid_h = 6, 4
        input_seq_len = grid_w * grid_h
        output_length = 6
        k = 2

        positions_np = _make_grid(grid_w, grid_h)

        np_weights = avg_pool_by_positions([positions_np], [output_length], k)

        np.testing.assert_allclose(
            np_weights.sum(axis=1), np.ones(output_length), atol=1e-6
        )
        np.testing.assert_allclose(
            np_weights.sum(axis=0), np.full(input_seq_len, 0.25), atol=1e-6
        )

    def test_ragged_two_images(self) -> None:
        """Block-diagonal weights for two images of different sizes."""
        k = 2
        pos1 = _make_grid(4, 4)
        out1 = 4
        pos2 = _make_grid(6, 6)
        out2 = 9
        weights = avg_pool_by_positions([pos1, pos2], [out1, out2], k=k)

        total_patches = 16 + 36
        total_output = out1 + out2
        assert weights.shape == (total_output, total_patches)

        # Block-diagonal: image 1 block is top-left, image 2 block is
        # bottom-right. Cross-image entries must be zero.
        assert weights[:out1, 16:].sum() == 0.0, "no cross-image leakage"
        assert weights[out1:, :16].sum() == 0.0, "no cross-image leakage"

        # Each block row should sum to 1 (full average).
        np.testing.assert_allclose(
            weights.sum(axis=1), np.ones(total_output), atol=1e-6
        )

    def test_ragged_three_images_matches_individual(self) -> None:
        """Ragged batch of 3 images must equal per-image results assembled."""
        k = 3
        grids = [_make_grid(6, 6), _make_grid(3, 3), _make_grid(6, 3)]
        outputs = [4, 1, 2]

        # Combined.
        combined = avg_pool_by_positions(grids, outputs, k)

        # Per-image, then assemble block-diagonal.
        blocks = [
            avg_pool_by_positions([g], [o], k)
            for g, o in zip(grids, outputs, strict=False)
        ]
        patch_counts = [g.shape[0] for g in grids]
        total_patches = sum(patch_counts)
        total_output = sum(outputs)
        expected = np.zeros((total_output, total_patches), dtype=np.float32)

        row_off, col_off = 0, 0
        for block, n_out, n_patches in zip(
            blocks, outputs, patch_counts, strict=False
        ):
            expected[
                row_off : row_off + n_out, col_off : col_off + n_patches
            ] = block
            row_off += n_out
            col_off += n_patches

        np.testing.assert_allclose(combined, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gemma4VisionPooler (MAX graph) vs torch reference forward
# ---------------------------------------------------------------------------


def _pool_inputs(
    positions_list: list[np.ndarray], out_lens: list[int], k: int
) -> np.ndarray:
    """Build the ``pool_gather_index`` graph input from grids.

    Mirrors what ``pack_vision_buffers`` ships to the vision graph: a per-output
    gather table ``[num_pooled, k**2]`` of patch indices.
    """
    return compute_pool_gather_index(positions_list, out_lens, k)


class TestGemma4VisionPooler:
    """Tests for the MAX Gemma4VisionPooler graph module (gather pooling)."""

    @staticmethod
    def _run(
        hidden: torch.Tensor,
        pool_gather_index: np.ndarray,
        k: int,
        *,
        on_gpu: bool = False,
        name: str = "pooler",
    ) -> torch.Tensor:
        """Compile + execute the pooler on a single device, return bf16 output."""
        _, hidden_size = hidden.shape
        device = Accelerator(0) if on_gpu else CPU()
        dev_ref = DeviceRef.GPU() if on_gpu else DeviceRef.CPU()
        pooler = Gemma4VisionPooler(
            hidden_size=hidden_size, pooling_kernel_size=k
        )
        session = InferenceSession(devices=[device])

        with Graph(
            name,
            input_types=[
                TensorType(DType.bfloat16, hidden.shape, device=dev_ref),
                TensorType(
                    DType.int32,
                    list(pool_gather_index.shape),
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, idx_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(idx_in, TensorValue)
            graph.output(pooler(h_in, idx_in))

        compiled = session.load(graph, weights_registry={})
        (result,) = compiled.execute(
            Buffer.from_dlpack(hidden).to(device),
            Buffer.from_dlpack(pool_gather_index).to(device),
        )
        return _buf_to_torch(result).to(TORCH_DTYPE)

    def test_output_shape(self) -> None:
        """Pooler output must be [num_pooled, hidden_size]."""
        hidden_size = 16
        k = 3
        positions_np = _make_grid(6, 6)  # 36 patches -> 4 pooled (k=3)
        gather_index = _pool_inputs([positions_np], [4], k)

        h = torch.randn(36, hidden_size, dtype=TORCH_DTYPE)
        out = self._run(h, gather_index, k, name="shape")
        assert out.shape == (4, hidden_size)

    def test_matches_dense_matmul(self) -> None:
        """Gather pooling must equal the block-diagonal ``(W @ h) * scale``."""
        hidden_size = 16
        k = 2
        pos1, pos2 = _make_grid(4, 4), _make_grid(6, 6)
        out1, out2 = 4, 9
        total_patches = pos1.shape[0] + pos2.shape[0]

        torch.manual_seed(5)
        h = torch.randn(total_patches, hidden_size, dtype=TORCH_DTYPE)

        # Dense reference computed in numpy from the kept reference function.
        np_weights = avg_pool_by_positions([pos1, pos2], [out1, out2], k)
        dense = (
            torch.from_numpy(np_weights).float()
            @ h.float()
            * math.sqrt(hidden_size)
        ).to(TORCH_DTYPE)

        gather_index = _pool_inputs([pos1, pos2], [out1, out2], k)
        out = self._run(h, gather_index, k, name="dense_eq")

        torch.testing.assert_close(out, dense, rtol=2e-2, atol=2e-2)

    def test_values_match_torch_reference(self) -> None:
        """Full pooler output must match torch reference forward."""
        hidden_size = 16
        output_length = 4
        k = 3
        positions_np = _make_grid(6, 6)
        input_seq_len = positions_np.shape[0]

        torch.manual_seed(42)
        x = torch.randn(1, input_seq_len, hidden_size, dtype=TORCH_DTYPE)

        torch_pooler = TorchGemma4VisionPooler(hidden_size, output_length)
        ref_output, _ = torch_pooler(
            x,
            torch.from_numpy(positions_np).unsqueeze(0),
            torch.ones(1, output_length, dtype=torch.bool),
            output_length,
        )

        gather_index = _pool_inputs([positions_np], [output_length], k)
        out = self._run(x[0], gather_index, k, name="ref")

        torch.testing.assert_close(out, ref_output[0], rtol=2e-2, atol=2e-2)

    def test_scale_factor(self) -> None:
        """Output must be scaled by sqrt(hidden_size) / k**2 over each bin."""
        hidden_size = 16
        k = 2
        # 2x2 grid pools (k=2) to a single bin of all 4 patches.
        positions_np = _make_grid(2, 2)
        gather_index = _pool_inputs([positions_np], [1], k)

        torch.manual_seed(1)
        h = torch.randn(4, hidden_size, dtype=TORCH_DTYPE)
        out = self._run(h, gather_index, k, name="scale")

        # output[0] = mean-pool(all 4 patches) * sqrt(hidden); gather sums the
        # 4 rows then scales by sqrt(hidden) * 1/k**2.
        ref = (
            h.float().sum(dim=0, keepdim=True)
            * math.sqrt(hidden_size)
            / (k * k)
        ).to(TORCH_DTYPE)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_ragged_two_images(self) -> None:
        """Pooler with ragged block-diagonal pooling for two images."""
        hidden_size = 16
        k = 2
        pos1, pos2 = _make_grid(4, 4), _make_grid(6, 6)
        n1, out1 = 16, 4
        out2 = 9
        total_patches = n1 + pos2.shape[0]

        torch.manual_seed(77)
        h = torch.randn(total_patches, hidden_size, dtype=TORCH_DTYPE)

        gather_index = _pool_inputs([pos1, pos2], [out1, out2], k)
        out = self._run(h, gather_index, k, name="ragged")

        torch_pooler1 = TorchGemma4VisionPooler(hidden_size, out1)
        torch_pooler2 = TorchGemma4VisionPooler(hidden_size, out2)
        ref1, _ = torch_pooler1(
            h[:n1].unsqueeze(0).clone(),
            torch.from_numpy(pos1).unsqueeze(0),
            torch.ones(1, out1, dtype=torch.bool),
            out1,
        )
        ref2, _ = torch_pooler2(
            h[n1:].unsqueeze(0).clone(),
            torch.from_numpy(pos2).unsqueeze(0),
            torch.ones(1, out2, dtype=torch.bool),
            out2,
        )
        ref_combined = torch.cat([ref1[0], ref2[0]], dim=0)

        torch.testing.assert_close(out, ref_combined, rtol=2e-2, atol=2e-2)

    def test_production_dimensions(self) -> None:
        """End-to-end test (on GPU) with gemma-4-31B-it production dimensions."""
        hidden_size = VISION_HIDDEN_SIZE
        input_seq_len = _INPUT_SEQ_LEN
        output_length = VISION_DEFAULT_OUTPUT_LENGTH
        k = VISION_POOLING_KERNEL_SIZE

        positions_np = _make_grid(_GRID_W, _GRID_H)
        assert positions_np.shape == (input_seq_len, 2)

        torch.manual_seed(7)
        x = torch.randn(1, input_seq_len, hidden_size, dtype=TORCH_DTYPE)
        torch_pooler = TorchGemma4VisionPooler(hidden_size, output_length)
        ref_output, _ = torch_pooler(
            x,
            torch.from_numpy(positions_np).unsqueeze(0),
            torch.ones(1, output_length, dtype=torch.bool),
            output_length,
        )

        gather_index = _pool_inputs([positions_np], [output_length], k)
        out = self._run(x[0], gather_index, k, on_gpu=True, name="prod")

        torch.testing.assert_close(out, ref_output[0], rtol=5e-2, atol=0.07)
