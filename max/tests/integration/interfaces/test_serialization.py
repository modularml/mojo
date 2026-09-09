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
"""Tests for serialization utilities."""

import gc
import pickle
from typing import Any

import msgspec
import numpy as np
import numpy.typing as npt
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.context.status import GenerationStatus
from max.pipelines.modeling.types import RequestID
from max.pipelines.modeling.types.utils.serialization import (
    _OOB_SIZE_THRESHOLD,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
    msgpack_numpy_oob_decoder,
    msgpack_numpy_oob_encoder,
)
from max.pipelines.request.open_responses import (
    ImageGenerationDetails,
    OutputImageContent,
    OutputVideoContent,
    Usage,
)


class SampleData(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """Sample data structure with numpy array for testing."""

    array: npt.NDArray[np.integer[Any]]
    value: int
    request_id: RequestID


def test_msgpack_numpy_decoder_pickle_serialization() -> None:
    """Test that MsgpackNumpyDecoder can be pickled and unpickled successfully."""
    # Create test data with a numpy array
    original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    test_data = SampleData(
        array=original_array, value=42, request_id=RequestID()
    )

    # Encode the test data
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_data)

    # Create a decoder
    original_decoder = msgpack_numpy_decoder(SampleData, copy=True)

    # Test that the original decoder works
    decoded_original = original_decoder(encoded_data)
    assert isinstance(decoded_original, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert decoded_original.value == 42

    # Pickle and unpickle the decoder
    pickled_decoder = pickle.dumps(original_decoder)
    unpickled_decoder = pickle.loads(pickled_decoder)

    # Test that the unpickled decoder still works
    decoded_unpickled = unpickled_decoder(encoded_data)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_unpickled.value == 42

    # Verify both decoders produce identical results
    assert np.array_equal(decoded_original.array, decoded_unpickled.array)
    assert decoded_original.value == decoded_unpickled.value


def test_msgpack_numpy_decoder_pickle_with_copy_false() -> None:
    """Test pickling decoder with copy=False parameter."""
    # Create test data
    original_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_data = SampleData(
        array=original_array, value=123, request_id=RequestID()
    )

    # Encode the test data
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_data)

    # Create a decoder with copy=False
    original_decoder = msgpack_numpy_decoder(SampleData, copy=False)

    # Pickle and unpickle the decoder
    pickled_decoder = pickle.dumps(original_decoder)
    unpickled_decoder = pickle.loads(pickled_decoder)

    # Test both decoders
    decoded_original = original_decoder(encoded_data)
    decoded_unpickled = unpickled_decoder(encoded_data)

    # Verify results are correct
    assert isinstance(decoded_original, SampleData)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 123


def test_msgpack_numpy_decoder_pickle_preserves_parameters() -> None:
    """Test that pickling preserves decoder parameters correctly."""
    # Test different parameter combinations
    test_cases = [
        (SampleData, True),
        (SampleData, False),
    ]

    for type_, copy in test_cases:
        # Create decoder with specific parameters
        original_decoder = msgpack_numpy_decoder(type_, copy=copy)

        # Pickle and unpickle
        pickled_decoder = pickle.dumps(original_decoder)
        unpickled_decoder = pickle.loads(pickled_decoder)

        # Verify internal parameters are preserved
        assert unpickled_decoder._type == type_
        assert unpickled_decoder._copy == copy


def test_msgpack_numpy_encoder_pickle_serialization() -> None:
    """Test that MsgpackNumpyEncoder can be pickled and unpickled successfully."""
    # Create test data with a numpy array
    original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    test_data = SampleData(
        array=original_array, value=42, request_id=RequestID()
    )

    # Create an encoder
    original_encoder = msgpack_numpy_encoder()

    # Test that the original encoder works
    encoded_original = original_encoder(test_data)
    assert len(encoded_original) > 0

    # Pickle and unpickle the encoder
    pickled_encoder = pickle.dumps(original_encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)

    # Test that the unpickled encoder still works
    encoded_unpickled = unpickled_encoder(test_data)
    assert len(encoded_unpickled) > 0

    # Verify both encoders produce identical results
    assert encoded_original == encoded_unpickled

    # Verify the encoded data can be decoded correctly
    decoder = msgpack_numpy_decoder(SampleData, copy=True)
    decoded_original = decoder(encoded_original)
    decoded_unpickled = decoder(encoded_unpickled)

    assert isinstance(decoded_original, SampleData)
    assert isinstance(decoded_unpickled, SampleData)
    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 42


def test_msgpack_numpy_encoder_pickle_with_shared_memory() -> None:
    """Test pickling encoder with shared memory parameters."""
    # Create test data
    original_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_data = SampleData(
        array=original_array, value=123, request_id=RequestID()
    )

    # Create an encoder with shared memory enabled
    original_encoder = msgpack_numpy_encoder(
        use_shared_memory=True, shared_memory_threshold=1000
    )

    # Pickle and unpickle the encoder
    pickled_encoder = pickle.dumps(original_encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)

    # Test both encoders
    encoded_original = original_encoder(test_data)
    encoded_unpickled = unpickled_encoder(test_data)

    # Verify results are identical
    assert encoded_original == encoded_unpickled

    # Verify the encoded data can be decoded correctly
    decoder = msgpack_numpy_decoder(SampleData, copy=True)
    decoded_original = decoder(encoded_original)
    decoded_unpickled = decoder(encoded_unpickled)

    assert np.array_equal(decoded_original.array, original_array)
    assert np.array_equal(decoded_unpickled.array, original_array)
    assert decoded_original.value == decoded_unpickled.value == 123


def test_msgpack_numpy_encoder_pickle_preserves_parameters() -> None:
    """Test that pickling preserves encoder parameters correctly."""
    # Test different parameter combinations
    test_cases = [
        (False, 0),
        (True, 1000),
        (False, 5000),
        (True, 0),
    ]

    for use_shared_memory, shared_memory_threshold in test_cases:
        # Create encoder with specific parameters
        original_encoder = msgpack_numpy_encoder(
            use_shared_memory=use_shared_memory,
            shared_memory_threshold=shared_memory_threshold,
        )

        # Pickle and unpickle
        pickled_encoder = pickle.dumps(original_encoder)
        unpickled_encoder = pickle.loads(pickled_encoder)

        # Verify internal parameters are preserved
        assert unpickled_encoder._use_shared_memory == use_shared_memory
        assert (
            unpickled_encoder._shared_memory_threshold
            == shared_memory_threshold
        )


def test_generation_output_serialization() -> None:
    """Test that GenerationOutput can be serialized and deserialized with ZeroMQ."""
    # Create test GenerationOutput with OutputImageContent containing base64 image data
    img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    test_output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputImageContent.from_numpy(img_array, format="png"),
        ],
        usage=Usage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            image_generation_details=ImageGenerationDetails(
                width=64,
                height=64,
                megapixels=0.004096,
                steps=28,
                image_count=1,
            ),
        ),
    )

    # Encode the GenerationOutput
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_output)
    assert len(encoded_data) > 0

    # Decode the GenerationOutput
    decoder = msgpack_numpy_decoder(GenerationOutput, copy=True)
    decoded_output = decoder(encoded_data)

    # Verify the decoded output matches the original
    assert isinstance(decoded_output, GenerationOutput)
    assert decoded_output.request_id == test_output.request_id
    assert decoded_output.final_status == test_output.final_status
    assert decoded_output.is_done == test_output.is_done
    assert decoded_output.usage == test_output.usage
    assert len(decoded_output.output) == len(test_output.output)

    # Verify the OutputImageContent is correct
    assert isinstance(decoded_output.output[0], OutputImageContent)
    assert decoded_output.output[0].type == "output_image"

    # Narrow type for mypy
    decoded_image = decoded_output.output[0]
    test_image = test_output.output[0]
    assert isinstance(decoded_image, OutputImageContent)
    assert isinstance(test_image, OutputImageContent)
    assert decoded_image.image_data == test_image.image_data
    assert decoded_image.format == "png"


def test_generation_output_serialization_with_multiple_images() -> None:
    """Test serialization of GenerationOutput with multiple images."""
    # Create test GenerationOutput with multiple OutputImageContent objects
    img_array1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img_array2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    test_output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputImageContent.from_numpy(img_array1, format="png"),
            OutputImageContent.from_numpy(img_array2, format="jpeg"),
        ],
    )

    # Encode and decode
    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_output)

    decoder = msgpack_numpy_decoder(GenerationOutput, copy=True)
    decoded_output = decoder(encoded_data)

    # Verify all images are preserved
    assert len(decoded_output.output) == 2

    # Narrow types for mypy
    decoded_image_0 = decoded_output.output[0]
    decoded_image_1 = decoded_output.output[1]
    test_image_0 = test_output.output[0]
    test_image_1 = test_output.output[1]
    assert isinstance(decoded_image_0, OutputImageContent)
    assert isinstance(decoded_image_1, OutputImageContent)
    assert isinstance(test_image_0, OutputImageContent)
    assert isinstance(test_image_1, OutputImageContent)

    assert decoded_image_0.format == "png"
    assert decoded_image_1.format == "jpeg"
    assert decoded_image_0.image_data == test_image_0.image_data
    assert decoded_image_1.image_data == test_image_1.image_data


def test_generation_output_serialization_preserves_output_video_frames() -> (
    None
):
    frames = np.random.randint(0, 256, (3, 8, 8, 3), dtype=np.uint8)
    test_output = GenerationOutput(
        request_id=RequestID(),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputVideoContent.from_numpy_frames(
                frames,
                frames_per_second=8,
                format="mp4",
            )
        ],
    )

    encoder = msgpack_numpy_encoder()
    encoded_data = encoder(test_output)

    decoder = msgpack_numpy_decoder(GenerationOutput, copy=True)
    decoded_output = decoder(encoded_data)

    assert len(decoded_output.output) == 1
    decoded_video = decoded_output.output[0]
    assert isinstance(decoded_video, OutputVideoContent)
    assert decoded_video.frames_per_second == 8
    assert decoded_video.format == "mp4"
    assert decoded_video.num_frames == 3
    assert decoded_video.frames is not None
    assert np.array_equal(decoded_video.frames, frames)


# ===----------------------------------------------------------------------=== #
# Out-of-band (OOB) zero-copy serialization
# ===----------------------------------------------------------------------=== #


class _ArrayPayload(msgspec.Struct, kw_only=True):
    """Payload with a numpy array, for out-of-band serialization tests."""

    array: npt.NDArray[Any]
    label: str


def test_oob_roundtrip_large_array_as_separate_frame() -> None:
    """A large array round-trips through OOB, riding as its own frame."""
    arr = np.random.rand(1024 * 1024).astype(np.float32)  # 4 MiB, well above
    frames = msgpack_numpy_oob_encoder()(_ArrayPayload(array=arr, label="x"))

    # Frame 0 is the msgpack stream; the large array is a second frame.
    assert len(frames) == 2

    decoded = msgpack_numpy_oob_decoder(_ArrayPayload)(frames)
    assert isinstance(decoded, _ArrayPayload)
    assert decoded.label == "x"
    assert decoded.array.dtype == np.float32
    assert np.array_equal(decoded.array, arr)


def test_oob_small_array_stays_inline() -> None:
    """An array below the threshold stays inline (a single frame)."""
    arr = np.arange(4, dtype=np.int64)  # 32 B, well below the threshold
    frames = msgpack_numpy_oob_encoder()(_ArrayPayload(array=arr, label="s"))

    assert len(frames) == 1

    decoded = msgpack_numpy_oob_decoder(_ArrayPayload)(frames)
    assert np.array_equal(decoded.array, arr)


def test_oob_threshold_boundary() -> None:
    """The inline/out-of-band split happens exactly at _OOB_SIZE_THRESHOLD."""
    encoder = msgpack_numpy_oob_encoder()
    itemsize = np.dtype(np.uint8).itemsize

    just_below = np.zeros(_OOB_SIZE_THRESHOLD - itemsize, dtype=np.uint8)
    at_threshold = np.zeros(_OOB_SIZE_THRESHOLD, dtype=np.uint8)

    # < threshold: inline (single frame); >= threshold: own out-of-band frame.
    assert len(encoder(_ArrayPayload(array=just_below, label="lo"))) == 1
    assert len(encoder(_ArrayPayload(array=at_threshold, label="hi"))) == 2

    decoder = msgpack_numpy_oob_decoder(_ArrayPayload)
    for arr in (just_below, at_threshold):
        decoded = decoder(encoder(_ArrayPayload(array=arr, label="b")))
        assert np.array_equal(decoded.array, arr)


def test_oob_roundtrip_dtypes_shapes_and_layout() -> None:
    """OOB preserves dtype, shape, and value for assorted arrays."""
    encoder = msgpack_numpy_oob_encoder()
    decoder = msgpack_numpy_oob_decoder(_ArrayPayload)
    cases: list[npt.NDArray[Any]] = [
        np.random.rand(256, 256).astype(np.float32),
        (np.random.rand(256, 256) * 100).astype(np.float16),
        (np.random.rand(256, 256) * 65535).astype(np.uint16),
        np.random.rand(10, 3, 32, 32).astype(np.float32),
        # Non-C-contiguous source is coerced and round-trips intact.
        np.asfortranarray(np.random.rand(512, 512).astype(np.float32)),
    ]
    for arr in cases:
        decoded = decoder(encoder(_ArrayPayload(array=arr, label="d")))
        assert decoded.array.dtype == arr.dtype
        assert decoded.array.shape == arr.shape
        assert np.array_equal(decoded.array, arr)


def test_oob_zero_copy_view_outlives_frames() -> None:
    """A copy=False view stays valid and correct after the frames are dropped.

    Guards the lifetime contract: the decoded array's base chain must retain the
    backing buffer (the received frame, in production) for the view's lifetime.
    """
    arr = np.random.rand(1024 * 1024).astype(np.float32)
    expected = arr.copy()
    frames = msgpack_numpy_oob_encoder()(_ArrayPayload(array=arr, label="v"))
    view = msgpack_numpy_oob_decoder(_ArrayPayload, copy=False)(frames).array

    # The zero-copy view aliases ZMQ-owned memory, so it is returned read-only.
    assert not view.flags.writeable

    # Drop every other reference to the source and the transport frames, force
    # GC, then scribble fresh allocations over any freed memory.
    del arr, frames
    gc.collect()
    _scribble = [np.full(1024 * 1024, i, np.float32) for i in range(4)]

    assert np.array_equal(view, expected)


def test_oob_copy_true_owns_data() -> None:
    """With copy=True the decoded array owns its data (no aliasing)."""
    arr = np.random.rand(1024 * 1024).astype(np.float32)
    frames = msgpack_numpy_oob_encoder()(_ArrayPayload(array=arr, label="c"))
    decoded = msgpack_numpy_oob_decoder(_ArrayPayload, copy=True)(frames)

    assert decoded.array.base is None
    assert decoded.array.flags.writeable  # owned copy is mutable
    assert np.array_equal(decoded.array, arr)


def test_oob_encoder_decoder_pickle_roundtrip() -> None:
    """The OOB encoder/decoder survive pickling (they cross the spawn boundary)."""
    arr = np.random.rand(1024 * 1024).astype(np.float32)
    encoder = pickle.loads(pickle.dumps(msgpack_numpy_oob_encoder()))
    decoder = pickle.loads(
        pickle.dumps(msgpack_numpy_oob_decoder(_ArrayPayload, copy=True))
    )

    decoded = decoder(encoder(_ArrayPayload(array=arr, label="p")))
    assert np.array_equal(decoded.array, arr)
