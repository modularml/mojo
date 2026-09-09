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
from std.memory import OpaquePointer
from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
from nn.attention.gpu.mla_decode_dispatch_scalars import (
    mla_decode_dispatch_scalars,
)


@export
def PyInit_kv_cache_ops() abi("C") -> PythonObject:
    """Creates a Python module with KV-cache helper bindings."""
    try:
        var b = PythonModuleBuilder("kv_cache_ops")
        b.def_function[mha_decode_num_partitions](
            "mha_decode_num_partitions",
            docstring="Returns the MHA decode partition count.",
        )
        b.def_function[mla_dispatch_args_scalar](
            "mla_dispatch_args_scalar",
            docstring="Returns the MLA dispatch metadata scalars.",
        )
        return b.finalize()
    except e:
        abort(t"failed to create kv cache op bindings module: {e}")


def _make_int_list(values: Array[Int, 3]) -> PythonObject:
    ref cpython = Python().cpython()
    var result_py_list = cpython.PyList_New(len(values))
    for i in range(len(values)):
        _ = cpython.PyList_SetItem(
            result_py_list, i, cpython.PyLong_FromSsize_t(values[i])
        )
    return PythonObject(from_owned=result_py_list)


def _get_ctx(
    device_context_ptr: PythonObject,
) raises -> Optional[OpaquePointer[MutUntrackedOrigin]]:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        return None
    return OpaquePointer[MutUntrackedOrigin](unsafe_from_address=addr)


def mha_decode_num_partitions(
    batch_size_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
    n_kv_heads_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var max_cache_valid_length = Int(py=max_cache_valid_length_obj)
    var n_kv_heads = Int(py=n_kv_heads_obj)
    var ctx = _get_ctx(device_context_ptr)

    if not ctx:
        raise Error("Expected a non-null device context pointer.")
    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if n_kv_heads < 1:
        raise Error("n_kv_heads must be positive.")

    var device_ctx = DeviceContext(ctx.unsafe_value())
    var num_partitions = mha_decoding_num_partitions(
        batch_size,
        max_cache_valid_length,
        n_kv_heads,
        device_ctx,
    )

    ref cpython = Python().cpython()
    return PythonObject(from_owned=cpython.PyLong_FromSsize_t(num_partitions))


def mla_dispatch_args_scalar(
    batch_size_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
    q_max_seq_len_obj: PythonObject,
    num_heads_obj: PythonObject,
    is_fp8_kv_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var max_cache_valid_length = Int(py=max_cache_valid_length_obj)
    var q_max_seq_len = Int(py=q_max_seq_len_obj)
    var num_heads = Int(py=num_heads_obj)
    var is_fp8_kv = Int(py=is_fp8_kv_obj) != 0
    var ctx = _get_ctx(device_context_ptr)

    if not ctx:
        raise Error("Expected a non-null device context pointer.")
    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if q_max_seq_len < 1:
        raise Error("q_max_seq_len must be positive.")
    if num_heads < 1:
        raise Error("num_heads must be positive.")

    var device_ctx = DeviceContext(ctx.unsafe_value())

    var scalars = mla_decode_dispatch_scalars(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        device_ctx,
    )
    var result = Array[Int, 3](uninitialized=True)
    result[0] = scalars[0]
    result[1] = scalars[1]
    result[2] = scalars[2]
    return _make_int_list(result)
