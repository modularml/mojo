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
"""Utility classes for parallelized numpy operations."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device, DevicePinnedBuffer
from max.dtype import DType


class ParallelArrayOps:
    """Parallelized numpy array operations for performance-critical data processing.

    Uses ThreadPoolExecutor to parallelize bulk copy operations that release the GIL,
    enabling true multi-threaded execution. Particularly effective for concatenating
    large arrays where memory bandwidth can be saturated across multiple cores.

    Thread pool cleanup is handled automatically via __del__ and weakref.finalize,
    providing defense-in-depth for resource cleanup.

    .. code-block:: python

        import numpy as np
        from max.nn.parallel.numpy_parallel_ops import ParallelArrayOps

        arr1 = np.ones((2, 4), dtype=np.float32)
        arr2 = np.zeros((3, 4), dtype=np.float32)
        arr3 = np.ones((1, 4), dtype=np.float32)

        ops = ParallelArrayOps(max_workers=20)
        result = ops.concatenate([arr1, arr2, arr3], axis=0)
    """

    def __init__(
        self, accelerator: Device | None = None, max_workers: int = 24
    ) -> None:
        """Initialize parallel array operations with a thread pool.

        Args:
            accelerator: The accelerator to allocate pinned memory on. If provided,
                the results of the concatenate operation will be allocated on a
                pinned buffer on the specified accelerator.
            max_workers: Maximum number of worker threads. Default is 24, which works
                well for typical server CPUs. Consider setting to match your expected
                number of arrays (e.g., 20 for up to 20 concurrent copies).
        """
        if accelerator is not None and accelerator.is_host:
            raise ValueError(
                "Unable to allocate pinned memory on CPU. A provided device must be a GPU."
            )
        self._accelerator = accelerator

        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers

        self._shutdown = False
        self._shutdown_lock = threading.Lock()

        # Register weakref finalizer as a safety net for cleanup
        self._finalizer = weakref.finalize(
            self, self._finalize_shutdown, self._pool, self._shutdown_lock
        )

    @staticmethod
    def _finalize_shutdown(
        pool: ThreadPoolExecutor, shutdown_lock: threading.Lock
    ) -> None:
        """Static cleanup method called by weakref.finalize.

        Args:
            pool: The ThreadPoolExecutor to shutdown.
            shutdown_lock: Lock for thread-safe shutdown.
        """
        try:
            with shutdown_lock:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            # Suppress errors during finalization
            pass

    def __del__(self) -> None:
        """Cleanup method called when the instance is being destroyed."""
        try:
            self.shutdown(wait=False)
        except Exception:
            # Suppress errors during cleanup
            pass

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool and release resources.

        Args:
            wait: If True, wait for pending tasks to complete. If False, cancel
                pending tasks immediately.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True
            try:
                self._pool.shutdown(wait=wait, cancel_futures=not wait)
            except Exception:
                pass

    def concatenate(
        self,
        arrays: Sequence[npt.NDArray[Any]],
        axis: int = 0,
        min_chunk_size_mb: float = 50.0,
    ) -> Buffer:
        """Concatenate arrays in parallel along the specified axis.

        Equivalent to np.concatenate but parallelized using thread pool. Automatically
        splits large arrays across multiple workers when there are fewer arrays than
        workers, maximizing memory bandwidth utilization. Uses intelligent heuristics
        to avoid splitting when overhead would exceed benefits.

        This matches np.concatenate's behavior of always returning a copy,
        even when arrays contains a single array.

        Args:
            arrays: List of numpy arrays to concatenate. Must have compatible shapes
                (same shape except along concatenation axis) and identical dtypes.
            axis: Axis along which to concatenate. Negative values count from the end.
                Default is 0.
            min_chunk_size_mb: Minimum chunk size in MB when splitting arrays. Default is
                50 MB. Prevents creating too many small work items that would have excessive
                overhead. Arrays are only split if the resulting chunks would be at least
                this size.

        Returns:
            Concatenated array with the same dtype as the input arrays.

        Raises:
            ValueError: If arrays is empty, shapes are incompatible, or dtypes differ.
            RuntimeError: If the thread pool has been shut down.
        """
        n = len(arrays)
        if n == 0:
            raise ValueError("Cannot concatenate empty list of arrays.")

        # Validate shapes and compute output shape
        first = arrays[0]
        first_shape = first.shape
        first_dtype = first.dtype

        # Normalize negative axis and validate
        if axis < 0:
            axis = len(first_shape) + axis

        if not (0 <= axis < len(first_shape)):
            raise IndexError(
                f"axis {axis} is out of bounds for array of dimension {len(first_shape)}"
            )

        if n == 1:
            # This copy is likely not needed, but it mocks the exact behaviour of numpy.concatenate.
            if self._accelerator is None:
                return Buffer.from_numpy(first.copy())
            else:
                out_max: Buffer = DevicePinnedBuffer(
                    shape=first.shape,
                    dtype=DType.from_numpy(first.dtype),
                    device=self._accelerator,
                )
                np.copyto(out_max.to_numpy(), first)
                return out_max

        # Pre-compute expected shape slices for efficient validation
        # All arrays must match: shape[:axis] and shape[axis+1:]
        expected_prefix = first_shape[:axis]
        expected_suffix = first_shape[axis + 1 :]

        # Validate dtype and compute total size along concatenation axis in a single pass
        concat_dim_size = 0
        offsets = [0] * (n + 1)

        for i, arr in enumerate(arrays):
            # Check dtype (fast equality check, fails fast)
            if arr.dtype != first_dtype:
                raise ValueError(
                    f"All arrays must have the same dtype. "
                    f"arrays[0]={first_dtype}, arrays[{i}]={arr.dtype}"
                )

            # Validate shape compatibility using tuple slicing (much faster than dimension-by-dimension)
            # Fast path: if shapes are identical, both comparisons pass immediately
            arr_shape = arr.shape
            if (
                arr_shape[:axis] != expected_prefix
                or arr_shape[axis + 1 :] != expected_suffix
            ):
                # Find specific dimension for detailed error message
                for dim in range(len(first_shape)):
                    if dim != axis and arr_shape[dim] != first_shape[dim]:
                        raise ValueError(
                            f"All arrays must have same shape except along concat axis. "
                            f"Dimension {dim}: arrays[0]={first_shape[dim]}, arrays[{i}]={arr_shape[dim]}"
                        )

            concat_dim_size += arr.shape[axis]
            offsets[i + 1] = concat_dim_size

        # Check if all arrays are contiguous
        # Non-contiguous arrays significantly slow down performance.
        arrays = [
            np.ascontiguousarray(arr) if not arr.flags["C_CONTIGUOUS"] else arr
            for arr in arrays
        ]

        # Create output shape
        out_shape = list(first_shape)
        out_shape[axis] = concat_dim_size

        # Allocate output tensor. It will be pinned on an accelerator if one is provided.
        max_dtype = DType.from_numpy(first_dtype)
        if self._accelerator is not None:
            device = self._accelerator
            pinned = True
        else:
            device = CPU()
            pinned = False
        if pinned:
            out_max = DevicePinnedBuffer(
                shape=out_shape,
                dtype=max_dtype,
                device=device,
            )
        else:
            out_max = Buffer(
                shape=out_shape,
                dtype=max_dtype,
                device=device,
            )

        # This will alias the underlying memory.
        # It should NOT copy the memory to another buffer.
        out_np = out_max.to_numpy()

        # Pre-compute all slices for parallel copying
        # Strategy: If we have fewer arrays than workers, split large arrays into chunks
        # to better utilize available parallelism and memory bandwidth.
        # Only split when the benefit outweighs the overhead.
        #
        # Heuristics:
        # - 70 MB arrays with 5 arrays (4 workers/array): Don't split (chunk too small)
        # - 140 MB arrays with 2 arrays (12 workers/array): Split into 2-3 chunks each
        # - 280 MB arrays with 2 arrays (12 workers/array): Split into 5+ chunks each
        # This ensures each chunk is >= min_chunk_size_mb to avoid excessive overhead.
        ndim = len(out_np.shape)
        min_chunk_bytes = min_chunk_size_mb * 1024 * 1024

        # Build work items: (src_array, src_slice, dst_slice)
        work_items = []

        for i in range(n):
            arr = arrays[i]
            arr_size_along_axis = arr.shape[axis]
            arr_bytes = arr.nbytes

            # Calculate how many workers are available per array
            potential_workers_per_array = max(1, self._max_workers // n)

            # Only consider splitting if:
            # 1. We have spare workers (potential_workers_per_array > 1)
            # 2. The array is large enough to benefit from splitting
            should_split = False
            num_chunks = 1

            if potential_workers_per_array > 1:
                # Calculate maximum beneficial chunks while respecting min_chunk_size
                # This ensures each chunk is at least min_chunk_size_mb
                max_beneficial_chunks = max(1, int(arr_bytes / min_chunk_bytes))

                # Use the smaller of: available workers or beneficial chunks
                num_chunks = min(
                    potential_workers_per_array, max_beneficial_chunks
                )

                # Only split if we can create at least 2 meaningful chunks
                # and each chunk would be >= min_chunk_size
                if num_chunks >= 2:
                    chunk_bytes = arr_bytes / num_chunks
                    should_split = chunk_bytes >= min_chunk_bytes

            if should_split:
                # Split array into chunks along concatenation axis
                chunk_size = (
                    arr_size_along_axis + num_chunks - 1
                ) // num_chunks

                # Create work items for each chunk
                for chunk_idx in range(num_chunks):
                    src_start = chunk_idx * chunk_size
                    src_end = min(
                        (chunk_idx + 1) * chunk_size, arr_size_along_axis
                    )

                    if src_start >= src_end:
                        break

                    # Source slice (from input array)
                    src_slice_list = [slice(None)] * ndim
                    src_slice_list[axis] = slice(src_start, src_end)
                    src_slice = tuple(src_slice_list)

                    # Destination slice (in output array)
                    dst_start = offsets[i] + src_start
                    dst_end = offsets[i] + src_end
                    dst_slice_list = [slice(None)] * ndim
                    dst_slice_list[axis] = slice(dst_start, dst_end)
                    dst_slice = tuple(dst_slice_list)

                    work_items.append((arr, src_slice, dst_slice))
            else:
                # Don't split - copy entire array in one operation
                src_slice = tuple([slice(None)] * ndim)
                dst_slice_list = [slice(None)] * ndim
                dst_slice_list[axis] = slice(offsets[i], offsets[i + 1])
                dst_slice = tuple(dst_slice_list)
                work_items.append((arr, src_slice, dst_slice))

        # Submit copy tasks
        futures = [
            self._pool.submit(
                self._copy_array_slice, out_np, src_arr, src_slice, dst_slice
            )
            for src_arr, src_slice, dst_slice in work_items
        ]

        # Wait for completion and propagate any exceptions
        for f in as_completed(futures):
            f.result()

        return out_max

    @staticmethod
    def _copy_array_slice(
        out: npt.NDArray[Any],
        src: npt.NDArray[Any],
        src_slice: tuple[slice, ...],
        dst_slice: tuple[slice, ...],
    ) -> None:
        """Worker function that copies a slice of source array into output.

        Runs in a worker thread. Uses np.copyto which releases the GIL, enabling
        true parallel execution across multiple CPU cores.

        Args:
            out: Pre-allocated output array.
            src: Source array to copy from.
            src_slice: Tuple of slices defining what to copy from source.
            dst_slice: Tuple of slices defining where to copy into output.
        """
        np.copyto(out[dst_slice], src[src_slice])
