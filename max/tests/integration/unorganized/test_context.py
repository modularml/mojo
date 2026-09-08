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

import pickle
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest
from max.pipelines.context import (
    EOSTracker,
    GenerationStatus,
    ImageMetadata,
    PixelContext,
    SamplingParams,
    SpecDecodingState,
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.context import FUTURE_TOKEN
from max.pipelines.modeling.types import (
    RequestID,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)


def dataclass_equal(left: Any, right: Any) -> bool:
    """Deep equality for dataclasses, handling numpy arrays and nested dataclasses.

    - Requires both `left` and `right` to be dataclass instances of the same type.
    - For each field:
        * If both values are dataclasses, compare them recursively.
        * If both values are numpy arrays, use np.array_equal.
        * Otherwise, use regular ==.
    """

    def _eq(lv: Any, rv: Any) -> bool:
        # Identity fast-path
        if lv is rv:
            return True

        # Nested dataclasses: recurse
        if is_dataclass(lv) and is_dataclass(rv):
            if type(lv) is not type(rv):
                return False
            for f in fields(lv):
                if not _eq(getattr(lv, f.name), getattr(rv, f.name)):
                    return False
            return True

        # NumPy array handling
        if isinstance(lv, np.ndarray) or isinstance(rv, np.ndarray):
            if not (isinstance(lv, np.ndarray) and isinstance(rv, np.ndarray)):
                return False  # one is array, the other is not
            return np.array_equal(lv, rv)

        # Fallback: normal equality
        return lv == rv

    if not (is_dataclass(left) and is_dataclass(right)):
        raise TypeError("dataclass_equal expects two dataclass instances")

    if type(left) is not type(right):
        return False

    for f in fields(left):
        if not _eq(getattr(left, f.name), getattr(right, f.name)):
            return False

    return True


def test_context__get_min_token_logit_mask() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={4}),
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids() -> (
    None
):
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_tracker=EOSTracker(eos_token_ids={4, 5}),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids_multistep() -> (
    None
):
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_tracker=EOSTracker(eos_token_ids={4, 5}),
    )
    vocab_mask = context.get_min_token_logit_mask(4)
    assert len(vocab_mask) == 4
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[2].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[3].tolist() == []

    context.update(1)
    context.update(1)
    context.update(1)
    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []


def test_context__get_min_token_logit_mask_with_no_eos_token_ids() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_no_min_new_tokens() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={4, 5}),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__eos() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={4}),
    )
    assert context.eos_tracker.eos_token_ids == {4}
    assert context.is_initial_prompt
    context.update(4)
    assert not context.is_initial_prompt
    assert len(context.tokens) == 5
    assert context.status == GenerationStatus.END_OF_SEQUENCE


def test_context__max_length() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=6,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )
    for i in range(2):
        assert context.status == GenerationStatus.ACTIVE
        context.update(i)
    assert context.status == GenerationStatus.MAXIMUM_LENGTH


def test_context__current_length() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    assert len(context.tokens) == 4
    assert context.is_initial_prompt

    context.update(4)
    assert not context.is_initial_prompt
    assert len(context.tokens) == 5

    # Currently, there are 5 tokens, we are saying
    # here is the next one, and we've generated 3 tokens
    # including that one, so increment the current length
    # accordingly.
    for i in range(3):
        context.update(5 + i)

    assert len(context.tokens) == 8


def test_context__seq_len() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    assert context.tokens.active_length == 4
    context.update(4)
    assert context.tokens.active_length == 1
    for i in range(5):
        context.update(5 + i)
    assert context.tokens.active_length == 1


def test_context__needs_ce() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    # There are 4 unencoded prompt tokens
    assert context.tokens.active_length == 4
    assert context.tokens.generated_length == 0

    # Encode 2/4 prompt tokens
    context.tokens.chunk(2)
    assert context.tokens.active_length == 2
    assert context.tokens.generated_length == 0
    context.update(98)  # token 98 is discarded
    assert context.tokens.all.tolist() == [0, 1, 2, 3]

    # There are 2 unencoded prompt tokens left
    assert context.tokens.generated_length == 0
    assert context.tokens.active_length == 2

    # Create a bunch of draft tokens like in spec decoding
    context.update(99)
    context.update(100)
    context.update(101)
    context.update(102)
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 99, 100, 101, 102]
    context.tokens.rewind_processing(2)

    # Even though the active length is 3, we are not in CE mode!
    assert context.tokens.active.tolist() == [100, 101, 102]
    assert context.tokens.active_length == 3
    assert context.tokens.generated_length > 0


def test_context__skip_processing() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    # Can't trim more tokens than the context has.
    with pytest.raises(ValueError):
        context.tokens.skip_processing(n=999)

    # Trimming 0 tokens does nothing.
    assert (context.tokens.active == np.array([0, 1, 2, 3])).all()
    assert context.tokens.active_length == 4
    assert len(context.tokens) == 4

    # Trimming 2 tokens should remove the first 2 tokens of prompt.
    context.tokens.skip_processing(n=2)
    assert (context.tokens.active == np.array([2, 3])).all()
    assert context.tokens.active_length == 2
    assert len(context.tokens) == 4  # does not change

    # Can't trim prompt to 0 tokens.
    with pytest.raises(ValueError):
        context.tokens.skip_processing(n=2)


def test_context__update_beyond_chunk_size() -> None:
    # This check evaluates whether we can update this array.
    # However, behaviour with max serve for updating, is slightly
    # different than behaviour off the server, as the text context
    # moves between the api worker & server worker.
    # Before making changes to resize behaviour, ensure you
    # test with the server, not just the `generate` entrypoint.
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    # 128, is the CHUNK_SIZE defined in context
    for i in range(128):
        context.update(i)


def test_context__reset() -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )
    assert context.tokens.active_length == 4
    assert context.tokens.active.tolist() == [0, 1, 2, 3]
    context.update(4)
    assert context.tokens.active_length == 1
    assert context.tokens.active.tolist() == [4]
    context.reset()
    assert context.tokens.active_length == 5
    assert context.tokens.active.tolist() == [0, 1, 2, 3, 4]
    context.update(5)
    assert context.tokens.active_length == 1
    assert context.tokens.active.tolist() == [5]


def test_context_sampling_params_integration() -> None:
    """Tests that TextContext properly stores and maintains SamplingParams."""
    custom_params = SamplingParams(
        top_k=25,
        temperature=0.7,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        repetition_penalty=1.15,
    )

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        sampling_params=custom_params,
    )

    # Verify the sampling params persist through context operations
    context.update(5)
    assert context.sampling_params is custom_params
    assert context.sampling_params.top_k == 25

    context.reset()
    assert context.sampling_params is custom_params
    assert context.sampling_params.temperature == 0.7
    assert context.sampling_params.temperature == 0.7


def test_context_sampling_params_stop() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_sequences=[[1, 2]]),
        sampling_params=custom_params,
    )

    context.update(1)
    context.update(2)
    print(context.tokens.generated)
    assert context.is_done
    assert np.array_equal(context.tokens.generated, np.array([1, 2]))

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_sequences=[[2], [3, 1]]),
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(3)

    assert not context.is_done
    assert np.array_equal(context.tokens.generated, np.array([1, 3]))


def test_context_sampling_params_eos_token_ids() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={5, 4, 2}),
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(2)

    assert context.is_done
    assert np.array_equal(context.tokens.generated, np.array([1, 2]))

    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={5, 4, 2}),
        sampling_params=custom_params,
    )
    context.update(3)
    context.update(6)

    assert not context.is_done
    assert np.array_equal(context.tokens.generated, np.array([3, 6]))


def test_context_serializable() -> None:
    # Test that we can encode a sample TextContext with Pickle
    original_context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextContext)
    assert dataclass_equal(pickle_decoded, original_context)

    # Test that we can encode a sample TextContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert dataclass_equal(msgpack_decoded, original_context)


def test_dkv_cache_hint_survives_serialization() -> None:
    # The hint is carried, never read, so the only thing MAX owes it is that
    # the bytes reaching the model worker are the bytes that arrived on the
    # request (CLIN-1630). Its predecessor needed a tagged msgspec struct to
    # cross this boundary; raw bytes need nothing, and this pins that.
    hint = b'{"version":2,"instances":[{"instance_name":"dkv-peer"}]}'
    original_context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        dkv_cache_hint=hint,
    )

    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext)
    decoded = deserialize(serialize(original_context))

    assert decoded.dkv_cache_hint == hint


def test_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextContext) with Pickle
    original_context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded[0] == original_tuple[0]
    assert dataclass_equal(pickle_decoded[1], original_tuple[1])

    # Test that we can encode a tuple of (str, TextContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, TextContext],
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded[0] == original_tuple[0]
    assert dataclass_equal(msgpack_decoded[1], original_tuple[1])


def test_text_and_vision_context_serializable() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 0, 2, 3, 4], dtype=np.int64)),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=2,
                pixel_values=np.array([99]),
            )
        ],
        vision_token_ids=[0],
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert dataclass_equal(pickle_decoded, original_context)

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextAndVisionContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert dataclass_equal(msgpack_decoded, original_context)


def test_text_and_vision_context_serializable_empty_pixel_values() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        images=[],
        vision_token_ids=[98],
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert dataclass_equal(pickle_decoded, original_context)

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextAndVisionContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert dataclass_equal(msgpack_decoded, original_context)


def test_text_and_vision_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextAndVisionContext) with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 0, 2, 3, 4], dtype=np.int64)),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=2,
                pixel_values=np.array([99]),
            )
        ],
        vision_token_ids=[0],
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded[0] == original_tuple[0]
    assert dataclass_equal(pickle_decoded[1], original_tuple[1])

    # Test that we can encode a tuple of (str, TextAndVisionContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(tuple[str, TextAndVisionContext])
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded[0] == original_tuple[0]
    assert dataclass_equal(msgpack_decoded[1], original_tuple[1])


def test_text_context_update_with_future_token() -> None:
    context = TextContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={42}),
    )

    with pytest.raises(
        ValueError,
        match=r"Cannot realize a future token when there are no generated tokens",
    ):
        context.realize_future_token(123)

    context.update_with_future_token()
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4, FUTURE_TOKEN]

    context.realize_future_token(5)
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4, 5]

    with pytest.raises(
        ValueError, match=r"Attempted to realize a non-future token"
    ):
        context.realize_future_token(6)

    context.update_with_future_token()
    with pytest.raises(ValueError, match=r"Cannot have multiple future tokens"):
        context.update_with_future_token()

    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4, 5, FUTURE_TOKEN]
    assert context.status == GenerationStatus.ACTIVE
    # The unrealized placeholder is held back from the output; only the
    # realized prefix streams. (Previously this raised; the consumed window
    # is now clamped to the realized prefix so an in-flight forward's
    # placeholder can never leak into a response.)
    output = context.to_generation_output()
    assert output.tokens == [5]
    assert FUTURE_TOKEN not in output.tokens

    context.realize_future_token(42)
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4, 5, 42]
    assert context.status == GenerationStatus.END_OF_SEQUENCE
    assert context.to_generation_output().tokens == [42]


def test_text_context_future_token_skipped_during_chunked_prefill() -> None:
    """Chunked-prefill continuations must not write a future-token placeholder.

    The overlap pipeline calls ``update_with_future_token`` on every context in
    the batch each step, including requests still in (chunked) prefill. For an
    actively-chunked context, ``advance_token_buffer`` advances the chunk and
    early-returns WITHOUT writing a ``FUTURE_TOKEN``, so a later chunked step
    must not raise "Cannot have multiple future tokens."
    """
    context = TextContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={42}),
    )

    context.tokens.chunk(4)
    assert context.tokens.actively_chunked
    context.update_with_future_token()
    assert context.tokens.generated_length == 0
    assert FUTURE_TOKEN not in context.tokens.all.tolist()

    context.tokens.chunk(2)
    assert context.tokens.actively_chunked
    context.update_with_future_token()
    assert context.tokens.generated_length == 0
    assert FUTURE_TOKEN not in context.tokens.all.tolist()

    assert not context.tokens.actively_chunked
    context.update_with_future_token()
    assert context.tokens.all.tolist()[-1] == FUTURE_TOKEN

    context.realize_future_token(9)
    assert context.tokens.all.tolist()[-1] == 9


def test_text_context_last_realized_token() -> None:
    """last_realized_token skips an unrealized overlap placeholder."""
    context = TextContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={42}),
    )
    assert context.last_realized_token == 4

    context.update(5)
    assert context.last_realized_token == 5

    context.update_with_future_token()
    assert context.tokens.all.tolist()[-1] == FUTURE_TOKEN
    assert context.last_realized_token == 5

    context.realize_future_token(6)
    assert context.last_realized_token == 6


def test_text_context_update_with_preemption_and_future_token() -> None:
    context = TextContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        eos_tracker=EOSTracker(eos_token_ids={42}),
    )
    assert context.tokens.generated_length == 0

    context.update_with_future_token()
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4, FUTURE_TOKEN]
    assert context.tokens.generated_length == 1

    # Notice that the future token is deleted when the context is reset.
    context.reset()
    assert context.tokens.all.tolist() == [0, 1, 2, 3, 4]
    assert context.tokens.generated_length == 0


def test_text_context_to_generation_output_validates_vocab_size() -> None:
    """Generated tokens must be non-negative and within vocab when vocab_size is set."""
    request_id = RequestID()

    context = TextContext(
        request_id=request_id,
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2], dtype=np.int64)),
        eos_tracker=EOSTracker(),
        vocab_size=100,
    )
    context.update(42)
    output = context.to_generation_output()
    assert output.tokens == [42]
    assert output.request_id == request_id

    negative_context = TextContext(
        request_id=request_id,
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2], dtype=np.int64)),
        eos_tracker=EOSTracker(),
        vocab_size=100,
    )
    negative_context.update(-1)
    with pytest.raises(
        RuntimeError,
        match=r"Generated negative token_id=-1",
    ):
        negative_context.to_generation_output()

    oob_context = TextContext(
        request_id=request_id,
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2], dtype=np.int64)),
        eos_tracker=EOSTracker(),
        vocab_size=10,
    )
    oob_context.update(10)
    with pytest.raises(
        RuntimeError,
        match=r"Generated out-of-vocabulary token_id=10.*\(valid range: \[0, 10\)\)",
    ):
        oob_context.to_generation_output()

    unset_vocab_context = TextContext(
        request_id=request_id,
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2], dtype=np.int64)),
        eos_tracker=EOSTracker(),
    )
    unset_vocab_context.update(999)
    output = unset_vocab_context.to_generation_output()
    assert output.tokens == [999]


def test_vision_context_reset() -> None:
    context = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        images=[
            ImageMetadata(
                start_idx=0,
                end_idx=1,
                pixel_values=np.array([10, 11, 12, 13, 14]),
            )
        ],
        vision_token_ids=[0],
    )
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.tokens.processed_length == 0
    assert context.tokens.active_length == 5
    assert context.needs_vision_encoding is True

    # The pixel values should remain set after update, but needs_vision_encoding should be False.
    context.update(5)
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.needs_vision_encoding is False
    assert context.tokens.processed_length == 5
    assert context.tokens.active_length == 1

    # The pixel values should be restored after reset.
    context.reset()
    assert len(context.images) == 1
    assert context.images[0].pixel_values.tolist() == [10, 11, 12, 13, 14]
    assert context.tokens.processed_length == 0
    assert context.tokens.active_length == 6
    assert context.needs_vision_encoding is True


def test_context__chunked_prefill_needs_ce_edge_case() -> None:
    """Test that needs_ce behaves correctly during chunked prefill processing.

    This test reproduces the production edge case where a context in chunked prefill
    reaches the end of prompt processing (_start_idx == _prompt_len) but hasn't
    started completion generation (_completion_start_idx == _completion_end_idx),
    while status remains ACTIVE and needs_ce incorrectly returns False.
    """
    # Test parameters
    n = 32  # Initial prompt length
    chunk_size = 8  # Chunked prefill chunk size
    m = n + chunk_size + 5  # Additional tokens: m > (n + chunk_size)

    # a. Create a random prompt of length n
    initial_prompt = np.arange(n, dtype=np.int64)

    context = TextContext(
        max_length=200,  # Large enough to accommodate all tokens
        tokens=TokenBuffer(initial_prompt),
    )

    # Verify initial state
    assert context.tokens.active_length == n
    assert context.tokens.generated_length == 0
    assert context.tokens.prompt_length == n
    assert context.tokens.generated_length == 0

    # b. Generate n + m tokens, where m > (n + chunk_size)
    for i in range(m):
        context.update(n + i)

    # Verify we've generated the expected number of tokens
    assert len(context.tokens) == n + m
    assert (
        context.tokens.generated_length > 0
    )  # All original prompt processed, completion generated
    assert context.tokens.generated_length == m

    # c. Reset the context object
    context.reset()

    # After reset, all tokens become the new prompt
    new_prompt_len = n + m
    assert context.tokens.active_length == new_prompt_len
    assert context.tokens.prompt_length == new_prompt_len
    assert context.tokens.generated_length == 0
    assert context.status == GenerationStatus.ACTIVE
    # Critical: completion indices are reset to equal values (the edge case setup)
    _ = context.to_generation_output()
    assert context.tokens.generated_length == 0

    # d. Simulate chunked prefill processing chunk by chunk
    processed_tokens = 0

    while processed_tokens < new_prompt_len:
        # Calculate current chunk size
        remaining_tokens = new_prompt_len - processed_tokens
        current_chunk_size = min(remaining_tokens, chunk_size)
        if current_chunk_size < remaining_tokens:
            context.tokens.chunk(current_chunk_size)

        # Before simulating the update call, verify needs_ce
        assert context.tokens.generated_length == 0, (
            f"needs_ce should be True when processing chunk at tokens "
            f"{processed_tokens} to {processed_tokens + current_chunk_size}"
        )

        # Simulate the chunked prefill path in update() method
        context.update(1)

        # Verify that indices were updated correctly
        assert (
            context.tokens.processed_length
            == processed_tokens + current_chunk_size
        )
        processed_tokens += current_chunk_size

        # Key test: verify needs_ce behavior after chunk processing
        if processed_tokens < new_prompt_len:
            # Still have prompt tokens to process
            assert context.tokens.current_position == new_prompt_len
            assert context.tokens.generated_length == 0

        else:
            # We've reached the critical edge case:
            # - All prompt tokens processed (_start_idx == _prompt_len)
            # - One completion tokens generated (_completion_start_idx == _completion_end_idx)
            # - Status is still ACTIVE
            assert context.tokens.current_position == new_prompt_len + 1

            assert (
                context.tokens.processed_length == context.tokens.prompt_length
            ), "Should have processed all prompt tokens"
            assert context.status == GenerationStatus.ACTIVE, (
                "Status should still be ACTIVE"
            )

            assert context.tokens.generated_length > 0, (
                "Processed all prompt tokens but no completion tokens generated",
            )
            assert context.tokens.generated_length == 1

    # Verify final state - the single generated token from chunked prefill was consumed
    _ = context.to_generation_output()
    assert context.tokens.processed_length == new_prompt_len
    assert context.tokens.active_length == 1
    assert context.tokens.generated_length == 1

    # Now simulate generating the first completion token
    # This should transition the context out of the edge case
    context.update(999)  # Generate first actual completion token

    # Verify proper transition to completion generation
    assert context.tokens.active_length == 1
    assert context.tokens.generated_length > 0
    assert (
        context.status == GenerationStatus.ACTIVE
    )  # Still active, but generating completions


def test_text_and_vision_context_post_init() -> None:
    # ok (contains one <vision_token_id>)
    _ = ImageMetadata(
        start_idx=0,
        end_idx=1,
        pixel_values=np.array([99]),
    )

    # not ok since start_idx is negative
    with pytest.raises(ValueError):
        _ = ImageMetadata(
            start_idx=-1,
            end_idx=1,
            pixel_values=np.array([99]),
        )

    # not ok since there are no room for any <vision_token_id>
    with pytest.raises(ValueError):
        _ = ImageMetadata(
            start_idx=0,
            end_idx=0,
            pixel_values=np.array([99]),
        )

    # ok (no images)
    _ = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        images=[],
        vision_token_ids=[98],
    )


def test_text_and_vision_context_happy_case() -> None:
    # fmt: off
    #                                      |<-- img0 --->|                         |<--- img1 -->|
    #                   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
    tokens = np.array([51, 52, 53, 54, 97, 98, 98, 98, 98, 99, 55, 56, 57, 58, 97, 98, 98, 98, 98, 99, 59, 60, 61, 62])
    # fmt: on
    ctx = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=5,
                end_idx=9,
                pixel_values=np.array([99]),
            ),
            ImageMetadata(
                start_idx=15,
                end_idx=19,
                pixel_values=np.array([99]),
            ),
        ],
        vision_token_ids=[98],
    )

    assert ctx.compute_image_aligned_idx(7) == 5
    assert ctx.compute_image_aligned_idx(8) == 5
    assert ctx.compute_image_aligned_idx(9) == 9
    assert ctx.compute_image_aligned_idx(10) == 10

    assert ctx.compute_image_aligned_idx(13) == 13
    assert ctx.compute_image_aligned_idx(14) == 14
    assert ctx.compute_image_aligned_idx(15) == 15
    assert ctx.compute_image_aligned_idx(17) == 15
    assert ctx.compute_image_aligned_idx(18) == 15
    assert ctx.compute_image_aligned_idx(19) == 19
    assert ctx.compute_image_aligned_idx(20) == 20

    assert ctx.image_idx == 0
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 2

    ctx.tokens.skip_processing(9)
    assert ctx.image_idx == 1
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 1

    ctx.tokens.skip_processing(5)
    assert ctx.image_idx == 1
    assert ctx.needs_vision_encoding is True
    assert len(ctx.next_images) == 1

    ctx.tokens.skip_processing(5)
    assert ctx.image_idx == 2
    assert ctx.needs_vision_encoding is False
    assert len(ctx.next_images) == 0


def test_text_and_vision_context_adjacent_images_allowed() -> None:
    # Image ranges are half-open [start_idx, end_idx), so two images whose
    # token runs touch (next.start_idx == prev.end_idx) do not overlap. Chat
    # templates that emit no separator token between consecutive images
    # produce exactly such touching ranges, and they must be accepted.
    # fmt: off
    #                                  |<-img0->|<-img1->|
    #                   0   1   2   3   4   5   6   7   8   9
    tokens = np.array([51, 52, 53, 54, 98, 98, 98, 98, 59, 60])
    # fmt: on
    ctx = TextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=4,
                end_idx=6,
                pixel_values=np.array([99]),
            ),
            ImageMetadata(
                start_idx=6,
                end_idx=8,
                pixel_values=np.array([99]),
            ),
        ],
        vision_token_ids=[98],
    )
    assert len(ctx.images) == 2
    assert ctx.needs_vision_encoding is True


def test_text_and_vision_context_sad_case() -> None:
    # fmt: off
    #                                      |<-- img0 --->|                         |<--- img1 -->|
    #                   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
    tokens = np.array([51, 52, 53, 54, 97, 98, 98, 98, 98, 99, 55, 56, 57, 58, 97, 98, 98, 98, 98, 99, 59, 60, 61, 62])
    # fmt: on

    with pytest.raises(ValueError, match="Images must be non-overlapping"):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=TokenBuffer(tokens),
            images=[
                ImageMetadata(
                    start_idx=5,
                    end_idx=9,
                    pixel_values=np.array([99]),
                ),
                # This overlaps with img0 (starts at 8, before img0 ends at 9).
                ImageMetadata(
                    start_idx=8,
                    end_idx=19,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    with pytest.raises(ValueError, match="Images must be sorted"):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=TokenBuffer(tokens),
            images=[
                ImageMetadata(
                    start_idx=15,
                    end_idx=19,
                    pixel_values=np.array([99]),
                ),
                ImageMetadata(
                    start_idx=5,
                    end_idx=9,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    with pytest.raises(
        ValueError, match="Images must be before the end of the token array"
    ):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=TokenBuffer(tokens),
            images=[
                ImageMetadata(
                    start_idx=20,
                    end_idx=25,
                    pixel_values=np.array([99]),
                ),
            ],
            vision_token_ids=[98],
        )

    # When the buffer is actively chunked (chunked prefill), current_position
    # may bisect an image — the vision encoder cache handles re-encoding.
    token_buffer = TokenBuffer(tokens)
    token_buffer.chunk(7)  # bisects img0 at 5-9

    ctx = TextAndVisionContext(
        max_length=50,
        tokens=token_buffer,
        images=[
            ImageMetadata(
                start_idx=5,
                end_idx=9,
                pixel_values=np.array([99]),
            ),
            ImageMetadata(
                start_idx=15,
                end_idx=19,
                pixel_values=np.array([99]),
            ),
        ],
        vision_token_ids=[98],
    )
    # img0 is bisected but this is valid during chunked prefill.
    assert ctx.image_idx == 0
    assert ctx.needs_vision_encoding

    with pytest.raises(
        ValueError,
        match="Images must be filled with <vision_token_id>",
    ):
        _ = TextAndVisionContext(
            max_length=50,
            tokens=TokenBuffer(tokens),
            images=[
                ImageMetadata(
                    start_idx=5, end_idx=9, pixel_values=np.array([99])
                ),
            ],
            vision_token_ids=[123],
        )


def does_not_raise_due_to_check_in_property_method() -> None:
    """This test ensures that `isinstance`, `hasattr`, and other class introspection
    methods do not execute the body of any methods and throw an
    exception.
    """

    ctx = TextContext(
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    # isinstance checks against concrete context classes should not raise.
    # The original bug report (GENAI-318) indicated that isinstance on VLM
    # contexts threw a ValueError; validate the concrete-class equivalents.
    assert isinstance(ctx, TextContext)
    _ = isinstance(ctx, TextAndVisionContext)
    _ = isinstance(ctx, PixelContext)


def test_context__spec_decoding_state_lazy_init() -> None:
    """Tests that spec_decoding_state is lazily initialized and reset clears it."""
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([0, 1, 2, 3], dtype=np.int64)),
    )

    # Initially None
    assert context._spec_decoding_state is None

    # Lazy initialization on first access
    state = context.spec_decoding_state
    assert isinstance(state, SpecDecodingState)
    assert state.draft_tokens_to_verify == []

    # Same instance on subsequent access
    assert context.spec_decoding_state is state

    # Mutate the state
    state.draft_tokens_to_verify = [10, 20, 30]

    # Reset clears the state
    context.update(4)
    context.reset()
    assert context._spec_decoding_state is None

    # Re-access creates a fresh state
    new_state = context.spec_decoding_state
    assert isinstance(new_state, SpecDecodingState)
    assert new_state is not state
    assert new_state.draft_tokens_to_verify == []


def test_context__spec_decoding_state_serializable() -> None:
    """Tests that TextContext with SpecDecodingState can be serialized."""
    context = TextContext(
        request_id=RequestID(),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
    )

    # Initialize state with non-default values
    context.spec_decoding_state.draft_tokens_to_verify = [10, 20]

    # Pickle round-trip
    pickle_encoded = pickle.dumps(context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextContext)
    assert pickle_decoded._spec_decoding_state is not None
    assert pickle_decoded.spec_decoding_state.draft_tokens_to_verify == [10, 20]

    # MsgPack round-trip
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext)
    msgpack_encoded = serialize(context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert isinstance(msgpack_decoded, TextContext)
    assert msgpack_decoded.spec_decoding_state.draft_tokens_to_verify == [
        10,
        20,
    ]


def test_pixel_context_serializable() -> None:
    # Test that we can encode a sample PixelContext with Pickle
    original_context = PixelContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        negative_tokens=TokenBuffer(np.array([5, 6], dtype=np.int64)),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, PixelContext)
    assert dataclass_equal(pickle_decoded, original_context)

    # Test that we can encode a sample PixelContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(PixelContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert dataclass_equal(msgpack_decoded, original_context)


def test_pixel_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, PixelContext) with Pickle
    original_context = PixelContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
        negative_tokens=TokenBuffer(np.array([5, 6], dtype=np.int64)),
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded[0] == original_tuple[0]
    assert dataclass_equal(pickle_decoded[1], original_tuple[1])

    # Test that we can encode a tuple of (str, PixelContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, PixelContext],
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded[0] == original_tuple[0]
    assert dataclass_equal(msgpack_decoded[1], original_tuple[1])


_VISION_TOKEN_ID = 98


def _windowed_vision_context(
    image_spans: list[tuple[int, int]],
    seq_len: int,
) -> TextAndVisionContext:
    """Build a TextAndVisionContext with ``(start, end)`` images."""
    tokens = np.ones(seq_len, dtype=np.int64)
    images = []
    for start, end in image_spans:
        tokens[start:end] = _VISION_TOKEN_ID
        images.append(
            ImageMetadata(
                start_idx=start,
                end_idx=end,
                pixel_values=np.zeros((2, 3), dtype=np.float32),
            )
        )
    return TextAndVisionContext(
        tokens=TokenBuffer(tokens),
        max_length=4096,
        vision_token_ids=[_VISION_TOKEN_ID],
        images=images,
    )


def test_next_images_in_window_drops_images_ahead_of_window() -> None:
    context = _windowed_vision_context([(4, 8), (12, 16), (20, 24)], seq_len=30)
    # Nothing processed yet, full window: all three images unencoded.
    assert len(context.next_images_in_window) == 3

    # Chunk the active window to [0, 16): the third image (start_idx=20) is
    # ahead of the window and must be dropped from the tail.
    context.tokens.chunk(16)
    windowed = context.next_images_in_window
    assert [img.start_idx for img in windowed] == [4, 12]

    # next_images_in_window is a prefix of next_images (drops only the tail).
    assert [img.start_idx for img in windowed] == [
        img.start_idx for img in context.next_images[: len(windowed)]
    ]


def test_next_images_in_window_includes_bisected_images_whole() -> None:
    context = _windowed_vision_context([(4, 8), (12, 16), (20, 24)], seq_len=30)
    # The chunk boundary bisects the second image (12..16): it overlaps the
    # window, so it is included whole (the encoder cannot split an image).
    context.tokens.chunk(14)
    assert [img.start_idx for img in context.next_images_in_window] == [4, 12]

    # A processed prefix that bisects the second image: it still overlaps the
    # window [14, 30), so it stays; the fully-behind first image is dropped.
    behind = _windowed_vision_context([(4, 8), (12, 16), (20, 24)], seq_len=30)
    behind.tokens.skip_processing(14)
    assert [img.start_idx for img in behind.next_images_in_window] == [12, 20]
