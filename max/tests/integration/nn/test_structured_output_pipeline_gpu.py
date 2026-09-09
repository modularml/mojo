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
"""End-to-end integration test for grammar-constrained sampling pipeline.

The graph stitches together every primitive added in DRIV-135/DRIV-136:

  1. A first GPU ``argmax`` over a hand-crafted logits vector to pick a
     token deterministically.
  2. A D2H ``mo.inplace_memcpy`` of that token into pinned host memory.
  3. ``mo.launch_host_func`` runs a Python callback on the GPU stream
     that advances an llguidance ``LLMatcher`` with the just-arrived
     token and writes the next-step "additive logits mask" (``0.0`` for
     allowed tokens, ``-inf`` for forbidden) into a second pinned buffer.
  4. An H2D ``mo.inplace_memcpy`` of the mask into a GPU buffer.
  5. A second GPU argmax that adds the mask to a fresh logits vector and
     samples the next token.

The grammar is the regex ``"ab"``. After consuming token ``"a"``, only
token ``"b"`` is allowed. Without the FSM mask the second logits vector
would argmax to ``"c"``; with the mask applied it must argmax to ``"b"``.
"""

import numpy as np
import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max import driver
from max.driver import CPU, Accelerator, Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, ops
from max.nn import kernels

# Tiny ASCII vocabulary keeps the test self-contained and deterministic.
#   token 0 -> "a"
#   token 1 -> "b"
#   token 2 -> "c"
#   token 3 -> ""    (EOS)
_VOCAB: list[bytes] = [b"a", b"b", b"c", b""]
_VOCAB_SIZE = len(_VOCAB)
_EOS_TOKEN_ID = 3


class _AsciiTokenizer:
    """Minimal byte tokenizer for grammar-constrained sampling tests."""

    eos_token_id: int = _EOS_TOKEN_ID
    bos_token_id: int | None = None
    tokens: list[bytes] = _VOCAB

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        result: list[int] = []
        for byte_val in s:
            ch = bytes([byte_val])
            for i, token in enumerate(self.tokens):
                if token == ch:
                    result.append(i)
                    break
        return result


def test_structured_output_pipeline_e2e() -> None:
    """End-to-end: GPU argmax -> D2H -> FSM update -> H2D -> masked argmax."""
    accelerator = Accelerator()
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip("Requires CUDA/HIP accelerator")

    gpu_ref = DeviceRef.from_device(accelerator)
    cpu_ref = DeviceRef.CPU()

    # llguidance setup: a regex grammar that matches exactly "ab".
    ll_tokenizer = LLTokenizer(
        TokenizerWrapper(_AsciiTokenizer()), n_vocab=_VOCAB_SIZE
    )
    grammar = LLMatcher.grammar_from_regex("ab")
    matcher = LLMatcher(ll_tokenizer, grammar)

    # Pinned host buffers for the cross-device transfers. Backed by
    # `DevicePinnedBuffer` (page-locked host memory tied to the GPU)
    # so D2H/H2D copies can be properly asynchronous and `to_numpy()`
    # exposes a zero-copy host view.
    token_pinned = DevicePinnedBuffer(
        dtype=DType.int64, shape=[1], device=accelerator
    )
    mask_pinned = DevicePinnedBuffer(
        dtype=DType.float32, shape=[_VOCAB_SIZE], device=accelerator
    )
    token_pinned_view = token_pinned.to_numpy()
    mask_pinned_view = mask_pinned.to_numpy()
    token_pinned_view[0] = -1
    mask_pinned_view[:] = -1.0

    # Host callback: read the just-arrived D2H token, advance the FSM,
    # convert the bitmask into an additive logits mask (0.0 allowed,
    # -inf forbidden), and write it into mask_pinned in place.
    def host_callback() -> None:
        token = int(token_pinned_view[0])
        matcher.consume_token(token)
        # compute_logit_bias() returns bytes of length n_vocab with
        # 200 = allowed, 0 = forbidden.
        bias_bytes = matcher.compute_logit_bias()
        bias = np.frombuffer(bias_bytes, dtype=np.uint8)[:_VOCAB_SIZE]
        additive = np.where(
            bias > 0, np.float32(0.0), np.float32(-np.inf)
        ).astype(np.float32)
        np.copyto(mask_pinned_view, additive)

    trampoline_ptr, user_data_ptr = driver.__unsafe_pack_py_host_func(
        host_callback
    )

    # Build the graph.
    with Graph(
        "structured_output_pipeline",
        input_types=[
            BufferType(DType.int64, [1], device=cpu_ref),
            BufferType(DType.float32, [_VOCAB_SIZE], device=cpu_ref),
            BufferType(DType.float32, [_VOCAB_SIZE], device=gpu_ref),
            BufferType(DType.int64, [2], device=cpu_ref),
        ],
    ) as graph:
        token_pinned_in = graph.inputs[0].buffer
        mask_pinned_in = graph.inputs[1].buffer
        mask_gpu_in = graph.inputs[2].buffer
        payload_in = graph.inputs[3].buffer

        # 1. Generate token1 from a constant logits vector. Logits favor
        #    token 0 ("a") - the only token allowed at the start of "ab".
        logits1 = ops.constant(
            np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32),
            dtype=DType.float32,
            device=gpu_ref,
        )
        token1 = ops.argmax(logits1, axis=-1)  # shape [1], int64, GPU

        # 2. D2H copy of token1 into pinned CPU memory.
        kernels.inplace_memcpy(dst=token_pinned_in, src=token1)

        # 3. Host callback: consume the token, write the additive mask
        #    into mask_pinned. Runs as a stream callback after the D2H
        #    copy completes.
        kernels.launch_host_func(payload=payload_in, device=gpu_ref)

        # 4. H2D copy of the mask from pinned CPU memory to GPU.
        mask_cpu_tensor = ops.buffer_load(mask_pinned_in)
        kernels.inplace_memcpy(dst=mask_gpu_in, src=mask_cpu_tensor)

        # 5. Sample the next token. Without the mask, argmax of logits2
        #    would pick token 2 ("c") because of the 5.0 entry; with the
        #    mask applied (only token 1 is allowed) the masked argmax
        #    is forced to token 1 ("b").
        logits2 = ops.constant(
            np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32),
            dtype=DType.float32,
            device=gpu_ref,
        )
        mask_gpu_tensor = ops.buffer_load(mask_gpu_in)
        masked_logits2 = logits2 + mask_gpu_tensor
        token2 = ops.argmax(masked_logits2, axis=-1)
        graph.output(token2)

    session = InferenceSession(devices=[accelerator, CPU()])
    model = session.load(graph)

    # Stage the GPU mask buffer (zero-init; overwritten by H2D) and the
    # host_func payload buffer.
    mask_gpu_buffer = Buffer.from_numpy(
        np.zeros(_VOCAB_SIZE, dtype=np.float32)
    ).to(accelerator)
    payload = Buffer(dtype=DType.int64, shape=[2], device=CPU())
    payload[0] = trampoline_ptr
    payload[1] = user_data_ptr

    [final_token] = model.execute(
        token_pinned, mask_pinned, mask_gpu_buffer, payload
    )
    accelerator.synchronize()

    final_token_np = final_token.to(CPU()).to_numpy()
    assert final_token_np.shape == (1,), final_token_np.shape
    assert final_token_np[0] == 1, (
        f"Expected the FSM-masked argmax to pick token 1 ('b') after "
        f"consuming token 0 ('a'), got {final_token_np[0]}. "
        f"D2H token={token_pinned.to_numpy()[0]}, "
        f"mask={mask_pinned.to_numpy()}"
    )
