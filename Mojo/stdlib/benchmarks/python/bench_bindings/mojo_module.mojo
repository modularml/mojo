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
"""Microbenchmark surface for Python -> Mojo FFI overhead.

This module exposes the same trivial calls (`noop`, `add`) through several
binding paths so that `bench.py` can attribute per-call overhead to specific
parts of the dispatch chain:

- `noop_def` / `add_def`: the high-level `def_function` path that most users
  will hit. Regression target.
- `noop_raw` / `add_raw`: hand-written wrappers registered via the lower-level
  `def_py_c_function` path. Lower bound for the current `METH_VARARGS`-based
  architecture; isolates the cost contributed by Mojo's generic dispatch
  wrapper from the cost of CPython's tuple-packing call convention itself.

See https://github.com/modular/modular/issues/6521.
"""

from std.os import abort

from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("mojo_module")

        # High-level `def_function` path (the regression target).
        b.def_function[noop_def]("noop_def")
        b.def_function[add_def]("add_def")

        # Low-level `def_py_c_function` path. Same observable behavior, but
        # bypasses the wrapper dispatch.
        # `*_raw` register under METH_VARARGS (hand-written PyCFunction
        # shape); `*_raw_fastcall` register under METH_FASTCALL
        # (hand-written PyCFunctionFast shape). The pair brackets the
        # call-convention cost so the def-vs-raw gap at a single
        # convention isolates Mojo's generic dispatch overhead.
        b.def_py_c_function(noop_raw, "noop_raw")
        b.def_py_c_function(add_raw, "add_raw")
        b.def_py_c_function(noop_raw_fastcall, "noop_raw_fastcall")
        b.def_py_c_function(add_raw_fastcall, "add_raw_fastcall")

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


# ===-----------------------------------------------------------------------===#
# High-level `def_function` path
# ===-----------------------------------------------------------------------===#


def noop_def(x: PythonObject) raises -> PythonObject:
    return x


def add_def(a: PythonObject, b: PythonObject) raises -> PythonObject:
    var ai = Int(py=a)
    var bi = Int(py=b)
    return PythonObject(ai + bi)


# ===-----------------------------------------------------------------------===#
# Low-level `def_py_c_function` path
# ===-----------------------------------------------------------------------===#


@export
def noop_raw(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    ref cpy = Python().cpython()
    # PyTuple_GetItem returns a borrowed reference; we must IncRef before
    # returning, since the caller expects a new (owned) reference.
    var item = cpy.PyTuple_GetItem(args, 0)
    return cpy.Py_NewRef(item)


@export
def add_raw(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    ref cpy = Python().cpython()
    var a = cpy.PyTuple_GetItem(args, 0)
    var b = cpy.PyTuple_GetItem(args, 1)
    var ai = cpy.PyLong_AsSsize_t(a)
    var bi = cpy.PyLong_AsSsize_t(b)
    return cpy.PyLong_FromSsize_t(ai + bi)


# ===-----------------------------------------------------------------------===#
# Low-level `def_py_c_function` path, METH_FASTCALL shape
# ===-----------------------------------------------------------------------===#


@export
def noop_raw_fastcall(
    py_self: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    ref cpy = Python().cpython()
    return cpy.Py_NewRef(args[])


@export
def add_raw_fastcall(
    py_self: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    ref cpy = Python().cpython()
    # Read directly from the borrowed `PyObject *const *` array, matching the
    # METH_FASTCALL convention CPython uses to invoke the callee.
    var ai = cpy.PyLong_AsSsize_t(args[])
    var bi = cpy.PyLong_AsSsize_t(args[unsafe_offset=1])
    return cpy.PyLong_FromSsize_t(ai + bi)
