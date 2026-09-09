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
"""Msgpack support for NumPy arrays."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

import msgspec
import numpy as np
from max.pipelines.context.eos_tracking import EOSTracker
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.request.open_responses import (
    OpenResponsesRequest,
    OutputAudioContent,
    OutputImageContent,
    OutputTextContent,
    OutputVideoContent,
    ReasoningSummaryContent,
    RefusalContent,
)
from pydantic import BaseModel

from .shared_memory import _ndarray_to_shared_memory, _open_shm_array

_logger = logging.getLogger("max.pipelines.modeling.types")

# Type registry for Pydantic models to avoid importlib
_PYDANTIC_TYPE_REGISTRY: dict[str, type] = {}


def _build_type_registry() -> dict[str, type]:
    """Build the static type registry by importing known Pydantic types.

    This avoids the need for dynamic importlib usage during deserialization.

    Returns:
        Dictionary mapping full type paths to type objects.
    """
    registry: dict[str, type] = {}

    # Register each type with its full module path
    for cls in [
        EOSTracker,
        OpenResponsesRequest,
        OutputAudioContent,
        OutputImageContent,
        OutputTextContent,
        OutputVideoContent,
        RefusalContent,
        ReasoningSummaryContent,
        GenerationOutput,
    ]:
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        registry[type_key] = cls

    return registry


# Initialize registry at module load time
_PYDANTIC_TYPE_REGISTRY = _build_type_registry()

_SHARED_MEMORY_THRESHOLD = 24000000

# Minimum array size (bytes) at or above which an array is serialized
# out-of-band (OOB): the raw buffer rides as its own ZMQ frame instead of being
# copied into the msgpack stream. Below this size an OOB frame buys nothing --
# pyzmq itself copies frames smaller than its ``zmq.COPY_THRESHOLD`` (64 KiB by
# default) even on a ``copy=False`` send, so a sub-threshold array would pay a
# copy *plus* the extra-frame and index-indirection overhead. Keeping small
# arrays inline avoids the extra frame; large arrays (multi-megabyte
# ``pixel_values``) are far above this and ride truly zero-copy.
_OOB_SIZE_THRESHOLD = 64 * 1024


def _encode_pydantic(obj: BaseModel) -> dict[str, Any]:
    """Encode a Pydantic model into a msgpack-friendly tagged dict.

    Shared by the inline and out-of-band encoder hooks so the two paths cannot
    drift.

    Args:
        obj: The Pydantic model instance to encode.

    Returns:
        A dict tagged with ``__pydantic__`` carrying the model's type path and
        dumped data.
    """
    return {
        "__pydantic__": True,
        "type": obj.__class__.__module__ + "." + obj.__class__.__qualname__,
        "data": obj.model_dump(mode="python"),
    }


def _encode_inline_numpy(obj: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array inline (copied into the msgpack stream).

    The single source of truth for the inline ``__np__`` wire format, shared by
    the inline and out-of-band encoder hooks (the latter falls back to inline for
    small/0-d/object arrays).

    Args:
        obj: The numpy array to encode inline.

    Returns:
        A dict tagged with ``__np__`` carrying the array's bytes, shape, and
        dtype.
    """
    return {
        "__np__": True,
        "data": obj.tobytes(),
        "shape": obj.shape,
        "dtype": str(obj.dtype),
    }


def _numpy_encoder_hook(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = _SHARED_MEMORY_THRESHOLD,
) -> Callable[[Any], Any]:
    """Create a configurable numpy encoding hook.

    Args:
        use_shared_memory: Whether to attempt shared memory conversion for numpy arrays.
        shared_memory_threshold: Minimum size in bytes for shared memory conversion.
            If 0, all arrays are candidates for conversion.
            The default value is 24MB (24,000,000 bytes), which is chosen based on
            internal micro-benchmarks. These benchmarks indicate that serialization
            using shared memory begins to show a measurable speedup for numpy arrays
            at or above this size, making it a practical default for performance-sensitive
            applications.

    Returns:
        Encoding hook function that handles numpy arrays and optionally converts
        them to shared memory.
    """

    def encode_hook(obj: Any) -> Any:
        """Custom encoder that handles numpy arrays and Pydantic models with optional shared memory conversion."""
        # Handle Pydantic BaseModel instances
        if isinstance(obj, BaseModel):
            return _encode_pydantic(obj)

        if isinstance(obj, np.ndarray):
            # Try shared memory conversion if enabled and array meets threshold
            if (
                use_shared_memory
                and obj.nbytes >= shared_memory_threshold
                and (shm_array := _ndarray_to_shared_memory(obj)) is not None
            ):
                return {
                    "__shm__": True,
                    "name": shm_array.name,
                    "shape": shm_array.shape,
                    "dtype": shm_array.dtype,
                }

            # Fall back to regular numpy encoding
            return _encode_inline_numpy(obj)

        return obj

    return encode_hook


class _MsgpackNumpyEncoder:
    """A pickleable encoder class for msgpack data with numpy arrays.

    This class wraps msgspec.msgpack.Encoder functionality in a pickleable
    container by storing the encoder parameters and recreating the encoder
    as needed.
    """

    def __init__(
        self,
        use_shared_memory: bool = False,
        shared_memory_threshold: int = _SHARED_MEMORY_THRESHOLD,
    ):
        """Initialize the encoder.

        Args:
            use_shared_memory: Whether to attempt shared memory conversion for numpy arrays
            shared_memory_threshold: Minimum size in bytes for shared memory conversion.
                                    If 0, all arrays are candidates for conversion.
        """
        if (
            use_shared_memory
            and float(os.environ.get("MODULAR_MAX_SHM_WATERMARK", 0.9)) == 0.0
        ):
            _logger.warning(
                "MODULAR_MAX_SHM_WATERMARK is set to 0.0, shared memory will be disabled."
            )
            self._use_shared_memory = False
        else:
            self._use_shared_memory = use_shared_memory

        self._shared_memory_threshold = shared_memory_threshold
        self._encoder: msgspec.msgpack.Encoder | None = None
        self._create_encoder()

    def _create_encoder(self) -> None:
        """Create the internal msgspec encoder."""
        enc_hook = _numpy_encoder_hook(
            self._use_shared_memory, self._shared_memory_threshold
        )
        self._encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)

    def __call__(self, obj: Any) -> bytes:
        """Encode object into bytes.

        Args:
            obj: The object to encode

        Returns:
            The encoded bytes
        """
        assert self._encoder is not None
        return self._encoder.encode(obj)

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable encoder)."""
        return {
            "_use_shared_memory": self._use_shared_memory,
            "_shared_memory_threshold": self._shared_memory_threshold,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the encoder."""
        self._use_shared_memory = state["_use_shared_memory"]
        self._shared_memory_threshold = state["_shared_memory_threshold"]
        self._encoder = None
        self._create_encoder()


def msgpack_numpy_encoder(
    use_shared_memory: bool = False,
    shared_memory_threshold: int = _SHARED_MEMORY_THRESHOLD,
) -> _MsgpackNumpyEncoder:
    """Create an encoder function that handles numpy arrays.

    Args:
        use_shared_memory: Whether to attempt shared memory conversion for numpy arrays
        shared_memory_threshold: Minimum size in bytes for shared memory conversion.
                                If 0, all arrays are candidates for conversion.

    Returns:
        A pickleable encoder instance that encodes objects into bytes
    """
    return _MsgpackNumpyEncoder(use_shared_memory, shared_memory_threshold)


def _numpy_oob_encoder_hook(
    aux_buffers: list[Any], oob_threshold: int
) -> Callable[[Any], Any]:
    """Create an out-of-band (OOB) numpy encoding hook.

    Large numpy arrays are appended to ``aux_buffers`` as zero-copy buffers and
    replaced inline by a small ``__np_oob__`` placeholder carrying their index;
    the caller transports the placeholder stream and the aux buffers as separate
    ZMQ frames (see :class:`_MsgpackNumpyOOBEncoder`). Small/0-d/object arrays
    stay inline.

    Args:
        aux_buffers: Mutable list the hook appends out-of-band buffers to. The
            owning encoder clears it before each encode and reads it after.
        oob_threshold: Minimum array size in bytes to send out-of-band.

    Returns:
        Encoding hook function for :class:`msgspec.msgpack.Encoder`.
    """

    def encode_hook(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return _encode_pydantic(obj)

        if isinstance(obj, np.ndarray):
            # Plain-dtype arrays at/above the threshold ride out-of-band as
            # their own frame (no copy into the msgpack stream). Object/void
            # dtypes (whose bytes are not self-describing) and tiny/0-d arrays
            # stay inline.
            if (
                obj.shape
                and obj.nbytes >= oob_threshold
                and obj.dtype.kind not in ("O", "V")
            ):
                # The wire format is always C-contiguous; coerce if needed (a
                # copy, but rare -- pixel_values are produced contiguous).
                contiguous = (
                    obj
                    if obj.flags["C_CONTIGUOUS"]
                    else np.ascontiguousarray(obj)
                )
                index = len(aux_buffers)
                # ``memoryview(...).cast("B")`` is a zero-copy 1-D byte view of
                # the (contiguous) array; it keeps ``contiguous`` alive until the
                # frame is sent.
                aux_buffers.append(memoryview(contiguous).cast("B"))
                return {
                    "__np_oob__": True,
                    "index": index,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                }

            return _encode_inline_numpy(obj)

        return obj

    return encode_hook


class _MsgpackNumpyOOBEncoder:
    """A pickleable msgpack encoder that emits out-of-band numpy buffers.

    Unlike :class:`_MsgpackNumpyEncoder` (which returns a single ``bytes``),
    calling this encoder returns a list of frames: frame 0 is the msgpack stream
    (with ``__np_oob__`` placeholders) and frames 1..N are the raw, zero-copy
    array buffers. Pair it with a multipart, ``copy=False`` ZMQ send and
    :class:`_MsgpackNumpyOOBDecoder` on the receiver.

    Not thread-safe: the per-call out-of-band buffer stash is shared mutable
    state, so use one encoder instance per (single-producer) socket.
    """

    def __init__(self, oob_threshold: int = _OOB_SIZE_THRESHOLD) -> None:
        """Initialize the encoder.

        Args:
            oob_threshold: Minimum array size in bytes to send out-of-band.
        """
        self._oob_threshold = oob_threshold
        self._aux_buffers: list[Any] = []
        self._encoder: msgspec.msgpack.Encoder | None = None
        self._create_encoder()

    def _create_encoder(self) -> None:
        """Create the internal msgspec encoder bound to the aux-buffer stash."""
        self._encoder = msgspec.msgpack.Encoder(
            enc_hook=_numpy_oob_encoder_hook(
                self._aux_buffers, self._oob_threshold
            )
        )

    def __call__(self, obj: Any) -> list[Any]:
        """Encode ``obj`` into a list of transport frames.

        Args:
            obj: The object to encode.

        Returns:
            ``[msgpack_stream, *out_of_band_buffers]``. The buffers alias the
            source arrays (zero-copy); the caller must send them before the
            source arrays are mutated or freed.
        """
        assert self._encoder is not None
        try:
            main = self._encoder.encode(obj)
            # The splat copies the buffer references into ``frames``, which then
            # solely owns them once the ``finally`` clears our stash.
            return [main, *self._aux_buffers]
        finally:
            # Clear in place (not reassign) so the hook's closure stays bound;
            # in ``finally`` so a failed encode cannot leak buffer references
            # (which pin their source arrays) onto the next call.
            self._aux_buffers.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable encoder)."""
        return {"_oob_threshold": self._oob_threshold}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the encoder."""
        self._oob_threshold = state["_oob_threshold"]
        self._aux_buffers = []
        self._encoder = None
        self._create_encoder()


def msgpack_numpy_oob_encoder(
    oob_threshold: int = _OOB_SIZE_THRESHOLD,
) -> _MsgpackNumpyOOBEncoder:
    """Create an out-of-band (zero-copy) msgpack encoder for numpy arrays.

    Args:
        oob_threshold: Minimum array size in bytes to send out-of-band. Smaller
            arrays are encoded inline.

    Returns:
        A pickleable encoder whose ``__call__`` returns a list of transport
        frames (see :class:`_MsgpackNumpyOOBEncoder`).
    """
    return _MsgpackNumpyOOBEncoder(oob_threshold)


class _MsgpackNumpyDecoder:
    """A pickleable decoder class for msgpack data with numpy arrays.

    This class wraps msgspec.msgpack.Decoder functionality in a pickleable
    container by storing the decoder parameters and recreating the decoder
    as needed.
    """

    def __init__(self, type_: Any, copy: bool = False):
        """Initialize the decoder.

        Args:
            type_: The type to decode into
            copy: Whether to copy numpy arrays when deserializing. Defaults to False.
        """
        self._type = type_
        self._copy = copy
        self._decoder: msgspec.msgpack.Decoder[Any] | None = None
        self._create_decoder()

    def _create_decoder(self) -> None:
        """Create the internal msgspec decoder."""
        self._decoder = msgspec.msgpack.Decoder(
            type=self._type, dec_hook=self._dec_hook
        )

    def _dec_hook(self, type_: type, obj: Any) -> Any:
        return _decode_numpy_array(type_, obj, self._copy)

    def __call__(self, data: bytes) -> Any:
        """Decode bytes into the specified type.

        Args:
            data: The bytes to decode

        Returns:
            The decoded object
        """
        assert self._decoder is not None
        return self._decoder.decode(data)

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable decoder)."""
        return {
            "_type": self._type,
            "_copy": self._copy,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the decoder."""
        self._type = state["_type"]
        self._copy = state["_copy"]
        self._decoder = None
        self._create_decoder()


def msgpack_numpy_decoder(
    type_: Any, copy: bool = False
) -> _MsgpackNumpyDecoder:
    """Create a decoder function for the specified type.

    Args:
        type_: The type to decode into
        copy: Copy numpy arrays if true. Defaults to True.
            Copy is set to True by default because most downstream usage of deserialized tensors are MAX driver tensors, which require owned numpy arrays.
            This is a constraint imposed by dlpack & numpy where we cannot create a buffer from read-only data.
            While there is a performance benefit during deserialization to removing copies by default, this often just moves the work downstream to an implicit copy during `Buffer.from_numpy`.
            As a result, it is easier to make the copy explicit here and maintain the pattern that all numpy arrays used in MAX are owned by the current process.

    Returns:
        A pickleable decoder instance that decodes bytes into the specified type
    """
    return _MsgpackNumpyDecoder(type_, copy)


class _MsgpackNumpyOOBDecoder:
    """A pickleable msgpack decoder for out-of-band numpy buffers.

    Counterpart to :class:`_MsgpackNumpyOOBEncoder`. Calling it takes the list of
    received frames ``[msgpack_stream, *out_of_band_buffers]`` (as produced by a
    multipart, ``copy=False`` ZMQ receive) and reconstructs the object, resolving
    each ``__np_oob__`` placeholder against its frame.

    With ``copy=False`` (the default), out-of-band arrays are zero-copy views
    over their frames; numpy's base chain keeps each frame mapped for exactly the
    view's lifetime, so a view may safely outlive the receive call. These views
    alias ZMQ-owned receive memory and are returned read-only so a stray write
    cannot corrupt it; pass ``copy=True`` for an owned, mutable array. Not
    thread-safe: use one decoder instance per (single-consumer) socket.
    """

    def __init__(self, type_: Any, copy: bool = False) -> None:
        """Initialize the decoder.

        Args:
            type_: The type to decode into.
            copy: Whether to copy numpy arrays out of their frames. Defaults to
                False (zero-copy views).
        """
        self._type = type_
        self._copy = copy
        self._aux: Sequence[Any] = ()
        self._decoder: msgspec.msgpack.Decoder[Any] | None = None
        self._create_decoder()

    def _create_decoder(self) -> None:
        """Create the internal msgspec decoder."""
        self._decoder = msgspec.msgpack.Decoder(
            type=self._type, dec_hook=self._dec_hook
        )

    def _dec_hook(self, type_: type, obj: Any) -> Any:
        return _decode_numpy_array(type_, obj, self._copy, self._aux)

    def __call__(self, frames: Sequence[Any]) -> Any:
        """Decode a list of received frames into the specified type.

        Args:
            frames: ``[msgpack_stream, *out_of_band_buffers]``. A single-element
                sequence (no out-of-band arrays) is also accepted.

        Returns:
            The decoded object.
        """
        assert self._decoder is not None
        # frame 0 is the msgpack stream; frames 1..N are out-of-band buffers.
        # Skip the slice on the common no-out-of-band path (e.g. text requests).
        self._aux = frames[1:] if len(frames) > 1 else ()
        try:
            return self._decoder.decode(frames[0])
        finally:
            # Release our reference to the frames. Decoded views (copy=False)
            # keep their own frame alive via the ndarray base chain; copy=True
            # results already own their data.
            self._aux = ()

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling (excluding the non-pickleable decoder)."""
        return {"_type": self._type, "_copy": self._copy}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling and recreate the decoder."""
        self._type = state["_type"]
        self._copy = state["_copy"]
        self._aux = ()
        self._decoder = None
        self._create_decoder()


def msgpack_numpy_oob_decoder(
    type_: Any, copy: bool = False
) -> _MsgpackNumpyOOBDecoder:
    """Create an out-of-band (zero-copy) msgpack decoder for the given type.

    Args:
        type_: The type to decode into.
        copy: Copy numpy arrays out of their frames if true. Defaults to False
            (zero-copy views; see :func:`msgpack_numpy_decoder` for the rationale
            behind defaulting to no-copy at this boundary).

    Returns:
        A pickleable decoder whose ``__call__`` takes a list of received frames
        (see :class:`_MsgpackNumpyOOBDecoder`).
    """
    return _MsgpackNumpyOOBDecoder(type_, copy)


def _array_from_buffer(
    buffer: Any, dtype: str, shape: Any, *, copy: bool
) -> np.ndarray:
    """Reconstruct a numpy array from a raw buffer holding its bytes.

    The single source of truth for materializing both the inline ``__np__`` and
    out-of-band ``__np_oob__`` wire formats (the decode-side peer of
    :func:`_encode_inline_numpy`).

    Args:
        buffer: A buffer-protocol object holding the array data -- an inline
            ``bytes`` payload or a received out-of-band ZMQ frame.
        dtype: The array's dtype string.
        shape: The array's shape.
        copy: If true, return a writable copy; otherwise a read-only zero-copy
            view whose base chain retains ``buffer`` for the view's lifetime.

    Returns:
        The reconstructed numpy array.
    """
    arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    if copy:
        return np.array(arr, copy=True)
    # Read-only so a stray in-place write cannot corrupt buffer-owned memory
    # (e.g. a ZMQ-owned receive frame whose lifetime is tied to this view).
    arr.flags.writeable = False
    return arr


def _decode_numpy_array(
    type_: type, obj: Any, copy: bool, aux: Sequence[Any] = ()
) -> Any:
    """Custom decoder for numpy arrays and Pydantic models from msgspec.

    Args:
        type_: The expected type (not used in this implementation)
        obj: The object to decode
        copy: Whether to copy the array data.
        aux: Out-of-band buffers (one per ``__np_oob__`` placeholder), indexed by
            the placeholder's ``index``. Empty for the inline-only decoder.

    Raises:
        ValueError: If a Pydantic type is not registered in the type registry.
    """

    def _decode_nested(value: Any) -> Any:
        if isinstance(value, dict):
            if any(
                key in value
                for key in ("__pydantic__", "__np__", "__shm__", "__np_oob__")
            ):
                return _decode_numpy_array(type_, value, copy, aux)
            return {k: _decode_nested(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_decode_nested(item) for item in value]
        return value

    # Handle Pydantic BaseModel instances
    if isinstance(obj, dict) and obj.get("__pydantic__") is True:
        type_key = obj["type"]

        # Get the class from the registry
        pydantic_class = _PYDANTIC_TYPE_REGISTRY.get(type_key)

        if pydantic_class is None:
            raise ValueError(
                f"Pydantic type '{type_key}' is not registered in the type registry. "
                f"Please add it to _build_type_registry() in "
                f"max/python/max/pipelines/modeling/types/utils/serialization.py. "
                f"Available types: {list(_PYDANTIC_TYPE_REGISTRY.keys())}"
            )

        try:
            # Reconstruct the Pydantic model from the dumped data
            # Type ignore needed because mypy can't infer that registry contains BaseModel subclasses
            return pydantic_class.model_validate(  # type: ignore[attr-defined]
                _decode_nested(obj["data"])
            )
        except Exception as e:
            _logger.error(f"Failed to validate Pydantic model data: {e}")
            raise

    if isinstance(obj, dict) and obj.get("__np__") is True:
        return _array_from_buffer(
            obj["data"], obj["dtype"], obj["shape"], copy=copy
        )

    if isinstance(obj, dict) and obj.get("__np_oob__") is True:
        # Zero-copy view: the returned array's base chain retains the received
        # ZMQ frame (``aux[index]``), so the frame stays mapped for exactly the
        # view's lifetime -- mirroring the shared-memory path's
        # weakref.finalize(arr, shm.close).
        return _array_from_buffer(
            aux[obj["index"]], obj["dtype"], obj["shape"], copy=copy
        )

    if isinstance(obj, dict) and obj.get("__shm__") is True:
        return _open_shm_array(obj)

    return obj
