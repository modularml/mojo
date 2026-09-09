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

from std.os import abort
from std.sys import size_of

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr
from std.memory import unsafe_memcpy
from std.hashlib._ahash import hash_seeded, hash_seeded_bytes


from sha256 import sha256


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    """Create a Python module with function bindings for `mojo_block_hasher`."""
    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_function[mojo_block_hasher](
            "mojo_block_hasher",
            docstring=(
                "Computes block hashes for a numpy array containing tokens"
            ),
        )
        b.def_function[mojo_block_hasher_sha256](
            "mojo_block_hasher_sha256",
            docstring=(
                "Chained SHA-256 block hasher. Writes (num_blocks, 32) uint8"
                " bytes into the supplied `out` numpy array. `parent_hash` is a"
                " 32-byte uint8 array used as the initial chain value."
            ),
        )
        b.def_function[mojo_sha256_oneshot](
            "mojo_sha256_oneshot",
            docstring=(
                "One-shot SHA-256 (FIPS 180-4): hashes `data` (uint8 numpy"
                " array) and writes the 32-byte digest into the supplied"
                " `out` numpy array. Exposed for known-answer-test"
                " validation of the underlying Mojo SHA-256 primitive."
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")


@fieldwise_init
struct PyArrayObject[dtype: DType](ImplicitlyCopyable):
    """
    Container for a numpy array.

    See: https://numpy.org/doc/2.1/reference/c-api/types-and-structures.html#c.PyArrayObject
    """

    var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var nd: Int

    var dimensions: Pointer[Int, MutUntrackedOrigin]

    var strides: Pointer[Int, MutUntrackedOrigin]
    var base: PyObjectPtr
    var descr: PyObjectPtr
    var flags: Int
    var weakreflist: PyObjectPtr

    # version dependent private members are omitted
    # ...

    def num_elts(self) -> Int:
        var num_elts = 1
        for i in range(self.nd):
            num_elts *= self.dimensions[unsafe_offset=i]
        return num_elts


@always_inline
def _mojo_block_hasher[
    dtype: DType,
    //,
](
    py_array_object_ptr: Pointer[PyArrayObject[dtype], _],
    block_size: Int,
    parent_hash: Int,
    seed: SIMD[.uint64, 4],
) -> PythonObject:
    # Compute number of hashes
    var num_elts: Int = py_array_object_ptr[].num_elts()
    var num_hashes: Int = num_elts // block_size

    ref cpython = Python().cpython()

    # Create a list of NULL elements with the size needed to store the hash
    # results.
    var result_py_list = cpython.PyList_New(num_hashes)

    # Performing hashing
    var prev_hash = parent_hash
    var num_bytes = block_size * size_of[dtype]()
    var hash_ptr_base = py_array_object_ptr[].data
    for block_idx in range(num_hashes):
        var hash_ptr_ints = hash_ptr_base.unsafe_offset(block_idx * block_size)
        var hash_ptr_bytes = hash_ptr_ints.unsafe_bitcast[Byte]()
        var token_hash = hash_seeded_bytes(hash_ptr_bytes, num_bytes, seed)
        var pair_to_hash = SIMD[.uint64, 2](UInt64(prev_hash), token_hash)
        var curr_hash = hash_seeded(pair_to_hash, seed)
        # Convert the hash result to a Python object and store it in our
        # uninitialized list.
        var curr_hash_obj = cpython.PyLong_FromSsize_t(Int(curr_hash))
        _ = cpython.PyList_SetItem(result_py_list, block_idx, curr_hash_obj)

        prev_hash = Int(curr_hash)

    return PythonObject(from_owned=result_py_list)


def mojo_block_hasher(
    py_array_object: PythonObject,
    block_size_obj: PythonObject,
    parent_hash_obj: PythonObject,
    seed_obj: PythonObject,
) raises -> PythonObject:
    # Parse np array tokens input
    var py_array_object_ptr = Pointer[PyArrayObject[.int32], _](
        unchecked_downcast_value=py_array_object
    )

    # Parse other arguments
    var block_size = Int(py=block_size_obj)
    var parent_hash = Int(py=parent_hash_obj)

    # Parse the 32-byte seed into 4 uint64 lanes.
    var seed_array_ptr = Pointer[PyArrayObject[.uint8], _](
        unchecked_downcast_value=seed_obj
    )
    var seed = (
        seed_array_ptr[].data.unsafe_bitcast[UInt64]().unsafe_load[width=4]()
    )

    # Performing hashing
    var results = _mojo_block_hasher(
        py_array_object_ptr, block_size, parent_hash, seed
    )

    return results^


@always_inline
def _mojo_block_hasher_sha256(
    tokens_ptr: Pointer[PyArrayObject[.int32], _],
    block_size: Int,
    parent_hash_ptr: Pointer[PyArrayObject[.uint8], _],
    out_ptr: Pointer[PyArrayObject[.uint8], _],
):
    """Chained block hashing using SHA-256. Writes (num_blocks, 32) bytes to out_ptr.
    Args:
        tokens_ptr: Pointer to the tokens array.
        block_size: The number of tokens per block.
        parent_hash_ptr: Pointer to the parent hash array.
        out_ptr: Pointer to the output array.
    """
    var num_token_elts = tokens_ptr[].num_elts()
    var num_blocks = num_token_elts // block_size
    var token_bytes_per_block = block_size * size_of[DType.int32]()

    var token_bytes_base = tokens_ptr[].data.unsafe_bitcast[Byte]()
    var parent_bytes = parent_hash_ptr[].data
    var out_bytes = out_ptr[].data

    var pair = Array[UInt8, 64](fill=UInt8(0))
    var pair_ptr: Pointer[UInt8, origin_of(pair)] = pair.unsafe_ptr()

    # Initialise prev with the caller-provided parent hash
    unsafe_memcpy(dest=pair_ptr.unsafe_offset(32), src=parent_bytes, count=32)

    for block_idx in range(num_blocks):
        # Local hash = SHA-256( token_bytes_for_this_block )
        var token_span = Span[Byte, _](
            unsafe_ptr=token_bytes_base.unsafe_offset(
                block_idx * token_bytes_per_block
            ),
            length=token_bytes_per_block,
        )
        var local_hash = sha256(token_span)

        # Pair = local_hash || prev_seq_hash; seq = SHA-256(pair)
        unsafe_memcpy(dest=pair_ptr, src=local_hash.unsafe_ptr(), count=32)
        var pair_span = Span[Byte, _](unsafe_ptr=pair_ptr, length=64)
        var seq_hash = sha256(pair_span)

        # Write to out[block_idx, :] and shift seq into pair[32:64] for newx iter
        unsafe_memcpy(
            dest=out_bytes.unsafe_offset(block_idx * 32),
            src=seq_hash.unsafe_ptr(),
            count=32,
        )
        unsafe_memcpy(
            dest=pair_ptr.unsafe_offset(32), src=seq_hash.unsafe_ptr(), count=32
        )


@always_inline
def _mojo_sha256_oneshot(
    data_ptr: Pointer[PyArrayObject[.uint8], _],
    out_ptr: Pointer[PyArrayObject[.uint8], _],
):
    """One-shot SHA-256 of a uint8 array; writes 32-byte digest to ``out``."""
    var n = data_ptr[].num_elts()
    var data_span = Span[Byte, _](unsafe_ptr=data_ptr[].data, length=n)
    var digest = sha256(data_span)
    var out_bytes = out_ptr[].data
    unsafe_memcpy(dest=out_bytes, src=digest.unsafe_ptr(), count=32)


def mojo_sha256_oneshot(
    data_obj: PythonObject,
    out_obj: PythonObject,
) raises -> PythonObject:
    var data_ptr = Pointer[PyArrayObject[.uint8], _](
        unchecked_downcast_value=data_obj
    )
    var out_ptr = Pointer[PyArrayObject[.uint8], _](
        unchecked_downcast_value=out_obj
    )
    _mojo_sha256_oneshot(data_ptr, out_ptr)
    return Python.none()


def mojo_block_hasher_sha256(
    tokens_obj: PythonObject,
    block_size_obj: PythonObject,
    parent_hash_obj: PythonObject,
    out_obj: PythonObject,
) raises -> PythonObject:
    var tokens_ptr = Pointer[PyArrayObject[.int32], _](
        unchecked_downcast_value=tokens_obj
    )
    var parent_ptr = Pointer[PyArrayObject[.uint8], _](
        unchecked_downcast_value=parent_hash_obj
    )
    var out_ptr = Pointer[PyArrayObject[.uint8], _](
        unchecked_downcast_value=out_obj
    )
    var block_size = Int(py=block_size_obj)

    _mojo_block_hasher_sha256(tokens_ptr, block_size, parent_ptr, out_ptr)

    return Python.none()
