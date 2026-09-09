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

"""Pins Qwen3.5's M-RoPE against the Transformers reference.

An image collapses many soft-token patches into far fewer position steps on
each of three axes, so the positions of everything after it depend on the
image's grid. A text-only prompt has no image to splice and all three axes
degenerate to the flat token index, which is why no text-only gate can see a
mistake here. These tests cover the two halves that carry the numerics:

- the positions themselves, against ``Qwen3_5Model.get_rope_index``;
- the frequency table those positions produce, against
  ``Qwen3_5TextRotaryEmbedding``, whose interleaved layout assigns axes to
  frequencies round-robin rather than in contiguous blocks;
- the decode steps that follow, against
  ``Qwen3_5Model.compute_3d_position_ids``, since the corrected positions have
  to keep going once the prompt is behind the request.

The last tests pin the degeneracy the text-only path relies on, and the two
places an FP8 KV cache has to turn an image away rather than serve it with
flat positions.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.qwen3_5.batch_processor import (
    Qwen3_5BatchProcessor,
)
from max.pipelines.architectures.qwen3_5.layers.attention import (
    Qwen3_5Attention,
)
from max.pipelines.architectures.qwen3_5.layers.text_rotary import (
    Qwen3_5TextRotaryEmbedding,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
    VisionEncodingData,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_rope_index,
)
from max.pipelines.context import (
    ImageMetadata,
    TextContext,
    TokenBuffer,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding as HFQwen3_5TextRotaryEmbedding,
)

# RadixArk/Qwen3.8-27B-NVFP4's declared text/vision geometry.
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
MROPE_SECTION = [11, 11, 10]
ROPE_THETA = 1e7
SPATIAL_MERGE_SIZE = 2
HIDDEN_SIZE = 5120
NUM_HEADS = 24
NUM_KV_HEADS = 4
PAGE_SIZE = 128
MAX_SEQ_LEN = 4096

IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054

# (T, H, W) patch grids, pre-merge. Square, wider-than-tall and
# taller-than-wide all exercise the `max(h, w)` advance differently.
GRIDS = [(1, 16, 16), (1, 8, 24), (1, 24, 8), (1, 4, 4)]

DECODE_STEPS = 3


def _hf_rope_index(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Calls ``Qwen3_5Model.get_rope_index`` without building the model.

    It only reads ``self.config.vision_config`` and calls
    ``self.get_vision_position_ids``, so instantiating the 27B module (and its
    vision tower) just to reach it would be pure cost.
    """
    shim = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
        )
    )
    shim.get_vision_position_ids = functools.partial(
        Qwen3_5Model.get_vision_position_ids, shim
    )
    return Qwen3_5Model.get_rope_index(shim, **kwargs)


def _prompt_with_image(
    grid: tuple[int, int, int],
    *,
    leading_text: int = 5,
    trailing_text: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds ``(input_ids, mm_token_type_ids)`` for text-image-text."""
    _, h, w = grid
    num_image_tokens = (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)
    ids = (
        [1] * leading_text
        + [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_image_tokens
        + [VISION_END_TOKEN_ID]
        + [1] * trailing_text
    )
    types = (
        [0] * (leading_text + 1)
        + [1] * num_image_tokens
        + [0] * (1 + trailing_text)
    )
    return (
        np.array(ids, dtype=np.int64).reshape(1, -1),
        np.array(types, dtype=np.int32).reshape(1, -1),
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_get_rope_index_matches_transformers(
    grid: tuple[int, int, int],
) -> None:
    """MAX's NumPy positions match ``Qwen3_5Model.get_rope_index`` exactly."""
    input_ids, mm_token_type_ids = _prompt_with_image(grid)
    grid_thw = np.array([grid], dtype=np.int64)

    max_positions, max_deltas = get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        input_ids=input_ids,
        image_grid_thw=grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=np.ones_like(input_ids),
    )

    hf_positions, hf_deltas = _hf_rope_index(
        input_ids=torch.from_numpy(input_ids),
        mm_token_type_ids=torch.from_numpy(mm_token_type_ids),
        image_grid_thw=torch.from_numpy(grid_thw),
        attention_mask=torch.ones_like(torch.from_numpy(input_ids)),
    )

    np.testing.assert_array_equal(max_positions, hf_positions.numpy())
    np.testing.assert_array_equal(max_deltas, hf_deltas.numpy())

    # Guards the test itself: a flat arange must NOT match, or a regression
    # that reverted to flat positions would still pass the assert above.
    flat = np.arange(input_ids.shape[1], dtype=np.int64)
    assert not np.array_equal(max_positions[0, 0], flat)


def _max_freqs_table(positions: np.ndarray) -> np.ndarray:
    """Runs ``Qwen3_5TextRotaryEmbedding`` over ``[3, total_seq_len]``."""
    device = DeviceRef.CPU()
    rope = Qwen3_5TextRotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=NUM_HEADS,
        theta=ROPE_THETA,
        max_seq_len=MAX_SEQ_LEN,
        mrope_section=MROPE_SECTION,
        head_dim=ROTARY_DIM,
        interleaved=True,
        scaling_params=None,
    )
    with Graph(
        "qwen3_5_mrope_table",
        input_types=[
            TensorType(
                DType.int64, shape=[3, positions.shape[1]], device=device
            )
        ],
    ) as graph:
        graph.output(rope.freqs_cis_position_ids(graph.inputs[0].tensor))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    return np.from_dlpack(model.execute(positions)[0])


@pytest.mark.parametrize("grid", GRIDS)
def test_freqs_table_matches_transformers(grid: tuple[int, int, int]) -> None:
    """The interleaved-M-RoPE cos/sin table matches Transformers'."""
    input_ids, mm_token_type_ids = _prompt_with_image(grid)
    positions, _ = _hf_rope_index(
        input_ids=torch.from_numpy(input_ids),
        mm_token_type_ids=torch.from_numpy(mm_token_type_ids),
        image_grid_thw=torch.from_numpy(np.array([grid], dtype=np.int64)),
        attention_mask=torch.ones_like(torch.from_numpy(input_ids)),
    )

    hf_rope = HFQwen3_5TextRotaryEmbedding(
        config=SimpleNamespace(
            max_position_embeddings=MAX_SEQ_LEN,
            head_dim=HEAD_DIM,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
                "mrope_section": MROPE_SECTION,
            },
        )
    )
    cos, sin = hf_rope(torch.zeros(1, dtype=torch.float32), positions)

    table = _max_freqs_table(positions.squeeze(1).numpy())
    half = ROTARY_DIM // 2
    # The graph emits (cos, sin) pairs per frequency; Transformers emits cos
    # and sin separately over the doubled `cat((freqs, freqs))` layout.
    np.testing.assert_allclose(
        table[:, 0::2], cos[0, :, :half].numpy(), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        table[:, 1::2], sin[0, :, :half].numpy(), rtol=0, atol=1e-6
    )


def _hf_decode_positions(
    *, rope_delta: int, past_length: int, num_new_tokens: int = 1
) -> npt.NDArray[np.int64]:
    """Continues positions the way Transformers does at decode time.

    With no grids passed and a filled cache, ``compute_3d_position_ids`` takes
    its ``rope_deltas`` branch, which is all a decode step after an image has
    to go on: the prompt's positions are gone by then.
    """
    positions = Qwen3_5Model.compute_3d_position_ids(
        SimpleNamespace(rope_deltas=torch.tensor([[rope_delta]])),
        input_ids=None,
        inputs_embeds=torch.zeros(1, num_new_tokens, 1),
        past_key_values=SimpleNamespace(get_seq_length=lambda: past_length),
    )
    return positions.squeeze(1).numpy()


def _image_context(
    grid: tuple[int, int, int],
) -> tuple[Qwen3VLTextAndVisionContext, npt.NDArray[np.int64]]:
    """Builds the context an image request reaches the batcher with."""
    input_ids, mm_token_type_ids = _prompt_with_image(grid)
    positions, deltas = _hf_rope_index(
        input_ids=torch.from_numpy(input_ids),
        mm_token_type_ids=torch.from_numpy(mm_token_type_ids),
        image_grid_thw=torch.from_numpy(np.array([grid], dtype=np.int64)),
        attention_mask=torch.ones_like(torch.from_numpy(input_ids)),
    )
    tokens = input_ids[0]
    image_token_indices = np.where(tokens == IMAGE_TOKEN_ID)[0]
    prefill_positions = positions.squeeze(1).numpy()
    context = Qwen3VLTextAndVisionContext(
        max_length=len(tokens) + DECODE_STEPS,
        tokens=TokenBuffer(tokens.copy()),
        images=[
            ImageMetadata(
                start_idx=int(image_token_indices[0]),
                end_idx=int(image_token_indices[-1]) + 1,
                pixel_values=np.zeros(0, dtype=np.float32),
            )
        ],
        vision_token_ids=[IMAGE_TOKEN_ID],
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        rope_delta=int(deltas.item()),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        image_token_indices=image_token_indices.astype(np.int32),
        decoder_position_ids=prefill_positions,
        vision_data=None,
    )
    return context, prefill_positions


def _max_decoder_position_ids(
    contexts: Sequence[TextContext],
) -> npt.NDArray[np.int64]:
    """Runs the batcher's position math for one step, on the host.

    ``_decoder_position_ids`` reaches nothing but ``self.runtime.devices``,
    while a real batcher needs a pipeline config and a loaded model.
    """
    batcher: Any = SimpleNamespace(runtime=SimpleNamespace(devices=[CPU()]))
    buffer = Qwen3_5BatchProcessor._decoder_position_ids(batcher, contexts)
    return buffer.to_numpy()


@pytest.mark.parametrize("grid", GRIDS)
def test_decode_continues_positions_past_an_image(
    grid: tuple[int, int, int],
) -> None:
    """Decode steps carry on from the image-corrected positions.

    An image spans more tokens than positions, so the prompt ends lower than
    its own length and a decode step counting from the flat token index would
    leave a gap. The batcher closes it with the request's rope delta, which is
    what Transformers does with its cached ``rope_deltas``.
    """
    context, prefill_positions = _image_context(grid)
    prompt_length = len(context.tokens)

    np.testing.assert_array_equal(
        _max_decoder_position_ids([context]), prefill_positions
    )
    # Without a delta to apply there would be nothing here to get wrong.
    assert context.rope_delta < 0

    for step in range(DECODE_STEPS):
        context.update(new_token=1)
        np.testing.assert_array_equal(
            _max_decoder_position_ids([context]),
            _hf_decode_positions(
                rope_delta=context.rope_delta,
                past_length=prompt_length + step,
            ),
        )


def test_decode_positions_of_a_batch_stay_with_their_requests() -> None:
    """A text-only neighbour is not dragged onto the image request's delta."""
    image_context, image_prefill = _image_context(GRIDS[0])
    text_tokens = np.arange(7, dtype=np.int64) + 1
    text_context = TextContext(
        max_length=len(text_tokens) + DECODE_STEPS,
        tokens=TokenBuffer(text_tokens),
    )
    contexts = [image_context, text_context]

    np.testing.assert_array_equal(
        _max_decoder_position_ids(contexts),
        np.concatenate(
            [
                image_prefill,
                _hf_decode_positions(
                    rope_delta=0,
                    past_length=0,
                    num_new_tokens=len(text_tokens),
                ),
            ],
            axis=1,
        ),
    )

    for _ in range(DECODE_STEPS):
        for context in contexts:
            context.update(new_token=1)
        expected = np.concatenate(
            [
                _hf_decode_positions(
                    rope_delta=image_context.rope_delta,
                    past_length=len(image_context.tokens) - 1,
                ),
                _hf_decode_positions(
                    rope_delta=0, past_length=len(text_context.tokens) - 1
                ),
            ],
            axis=1,
        )
        np.testing.assert_array_equal(
            _max_decoder_position_ids(contexts), expected
        )


def test_flat_positions_reproduce_static_table() -> None:
    """On this CPU backend, flat positions land on exactly the prior table.

    With no image every axis carries the flat token index, so the per-token
    table must reproduce the static one row for row -- bit for bit, not
    within a tolerance, or the logic regressed. This does not hold on every
    backend: GPU transcendentals are measurably less exact than the ones the
    static table folds in at compile time (see
    ``Qwen3_5TextRotaryEmbedding``'s docstring), so on GPU this is a few
    bfloat16 ULP, not bit-identical -- paid by every request once a
    checkpoint enables M-RoPE, not only ones with an image.
    """
    seq_len = 37
    flat = np.arange(seq_len, dtype=np.int64)
    table = _max_freqs_table(np.tile(flat, (3, 1)))

    static_rope = Llama3RotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=NUM_HEADS,
        theta=ROPE_THETA,
        max_seq_len=MAX_SEQ_LEN,
        head_dim=ROTARY_DIM,
        interleaved=True,
        scaling_params=None,
    )
    with Graph("qwen3_5_static_table", input_types=[]) as graph:
        graph.output(static_rope.freqs_cis)
    session = InferenceSession(devices=[CPU()])
    static = np.from_dlpack(session.load(graph).execute()[0])

    np.testing.assert_array_equal(table, static[:seq_len])


def _vision_data() -> VisionEncodingData:
    """Builds the encoder payload an unencoded image carries into the batcher.

    Only its presence decides the rejection under test, so each array is the
    emptiest one its field's dtype admits.
    """
    return VisionEncodingData(
        image_grid_thw=np.zeros((1, 3), dtype=np.int32),
        video_grid_thw=None,
        vision_position_ids=np.zeros(0, dtype=np.int32),
        max_grid_size=np.zeros((), dtype=np.int32),
        weights=np.zeros((1, 0), dtype=np.float32),
        indices=np.zeros((1, 0), dtype=np.int64),
        cu_seqlens=np.zeros(1, dtype=np.uint32),
        max_seqlen=np.zeros(1, dtype=np.uint32),
        concatenated_pixel_values=np.zeros(0, dtype=np.float32),
    )


def _batcher_without_mrope() -> Qwen3_5BatchProcessor:
    """Builds the batcher an FP8 KV cache configuration ends up with.

    Its ``__init__`` wants a loaded pipeline, and its ``super()`` call needs a
    real instance rather than the namespace the other helpers here pass, so
    the few fields the rejection path reaches are filled in directly.
    """
    batcher: Any = object.__new__(Qwen3_5BatchProcessor)
    batcher.runtime = SimpleNamespace(
        devices=[CPU()],
        signal_buffers=[],
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=1)
        ),
    )
    batcher._state_cache = SimpleNamespace(
        claim=lambda request_id: None,
        slot_idx_for=lambda request_ids, prealloc: prealloc,
        conv_pools=[],
        rec_pools=[],
    )
    batcher._slot_idx_prealloc = []
    batcher._mrope_enabled = False
    return batcher


def test_batcher_rejects_an_image_when_mrope_is_off() -> None:
    """Without the positions input, an image request is refused outright.

    The graph compiled for an FP8 KV cache has nowhere to put the corrected
    positions, and serving the request anyway is the silent wrong answer this
    whole change exists to remove.
    """
    context, _ = _image_context(GRIDS[0])
    context.vision_data = _vision_data()

    with pytest.raises(ValueError, match="cannot serve image prompts"):
        _batcher_without_mrope().prepare_initial_token_inputs([[context]])


def _fp8_attention() -> Qwen3_5Attention:
    """Builds the layer an FP8 KV cache would compile if M-RoPE were let in."""
    attention = Qwen3_5Attention(
        rope=Qwen3_5TextRotaryEmbedding(
            dim=HIDDEN_SIZE,
            n_heads=NUM_HEADS,
            theta=ROPE_THETA,
            max_seq_len=MAX_SEQ_LEN,
            mrope_section=MROPE_SECTION,
            head_dim=ROTARY_DIM,
            interleaved=True,
            scaling_params=None,
        ),
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        kv_params=MHAKVCacheParams(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            num_layers=1,
            page_size=PAGE_SIZE,
            devices=[DeviceRef.CPU()],
        ),
        layer_idx=0,
        dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
    )
    # Every projection of a standalone layer calls its weight `weight`, and a
    # graph refuses the second one under that name. Serving qualifies them
    # while loading the checkpoint; there is none here, so borrow just that.
    for name, weight in attention.raw_state_dict().items():
        weight.name = name
    return attention


def test_fp8_attention_takes_per_token_freq_rows() -> None:
    """The FP8 store path carries M-RoPE positions, like the bf16 path.

    Builds the layer an FP8 KV cache compiles and checks it reaches the fused
    store with the row ids intact.
    """
    device = DeviceRef.CPU()
    with Graph(
        "qwen3_5_fp8_mrope",
        input_types=[
            TensorType(
                DType.bfloat16,
                shape=["total_seq_len", HIDDEN_SIZE],
                device=device,
            ),
            TensorType(DType.uint32, shape=["batch_plus_one"], device=device),
            BufferType(
                DType.float8_e4m3fn,
                shape=["num_blocks", 2, 1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM],
                device=device,
            ),
            TensorType(DType.uint32, shape=["batch"], device=device),
            TensorType(
                DType.uint32, shape=["batch", "num_pages"], device=device
            ),
            TensorType(DType.uint32, shape=[], device=device),
            TensorType(DType.uint32, shape=[], device=device),
            TensorType(
                DType.float32,
                shape=["total_seq_len", ROTARY_DIM],
                device=device,
            ),
            TensorType(DType.uint32, shape=[1, "total_seq_len"], device=device),
            TensorType(DType.int64, shape=[4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        (
            x,
            input_row_offsets,
            kv_blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            freqs_cis,
            freq_row_ids,
            dispatch_metadata,
        ) = graph.inputs
        out = _fp8_attention()(
            ops.constant(0, DType.uint32, device),
            x.tensor,
            PagedCacheValues(
                kv_blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
                attention_dispatch_metadata=dispatch_metadata.tensor,
            ),
            freqs_cis.tensor,
            input_row_offsets.tensor,
            freq_row_ids.tensor,
        )
        graph.output(out)

    mlir = str(graph._mlir_op)
    assert "rope_split_store.ragged.paged.with_position_id" in mlir
