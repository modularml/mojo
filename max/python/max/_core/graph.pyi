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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""MAX Graph Python bindings."""

import os
import pathlib
from collections.abc import Sequence

import max._core
import max._core.dialects.builtin
import max._core.dialects.m
import max._core.dialects.mo
import max._core.driver
import max._core.dtype
import max._mlir.ir

def load_modular_dialects(arg: max._mlir.ir.DialectRegistry, /) -> None:
    """Registers all Modular MLIR dialects into the given registry."""

def array_attr(
    arg0: max._core.driver.Buffer, arg1: max._core.dialects.mo.TensorType, /
) -> max._core.dialects.m.ArrayElementsAttr:
    """Creates an array attribute from a buffer and tensor type."""

def _buffer_from_constant_attr(
    attr: max._core.dialects.builtin.ElementsAttr,
    dtype: max._core.dtype.DType,
    shape: Sequence[int],
    device: max._core.driver.Device,
) -> max._core.driver.Buffer:
    """Creates a Buffer from an MLIR constant attribute."""

def next_operation(arg: max._mlir.ir.Operation, /) -> max._mlir.ir.Operation:
    """Returns the next operation in the parent block, or None."""

def prev_operation(arg: max._mlir.ir.Operation, /) -> max._mlir.ir.Operation:
    """Returns the previous operation in the parent block, or None."""

def last_operation(arg: max._mlir.ir.Block, /) -> max._mlir.ir.Operation:
    """Returns the last operation in the given block, or None."""

def dtype_to_type(arg: max._core.dtype.DType, /) -> max._core.Type:
    """Converts a MAX DType to the corresponding MLIR type."""

def type_to_dtype(arg: max._core.Type, /) -> max._core.dtype.DType:
    """Converts an MLIR type to the corresponding MAX DType."""

def frame_loc(
    arg0: max._mlir.ir.Context, arg1: object, /
) -> max._mlir.ir.Location:
    """Creates an opaque MLIR location containing a Python stack frame."""

def profile_scope_location(
    ctx: max._mlir.ir.Context,
    name: str,
    child: max._mlir.ir.Location,
    color: str | None = None,
) -> max._mlir.ir.Location:
    """
    Wraps `child` in a MOGG::ProfileScopeLocationAttr naming the active Graph.profile_scope(), with an optional color.
    """

def source_locations_bytecode(op: max._core.Operation) -> bytes:
    """
    Serializes the operation to MLIR bytecode with each op's Python source location materialized into printable form. Print it by reparsing through max.mlir. Does not mutate the operation.
    """

def _init_and_register_max_context(mlir_ctx: max._mlir.ir.Context) -> None:
    """
    Initializes a process-wide M::Context when none exists, then registers it with the given MLIR context so compiler code can use loadContext.

    Internal API; does not expose M::Context to Python. New contexts are
    created via Init::getOrCreateContext (same Init path the Engine uses before
    attaching devices).
    """

class KernelTorchInfo:
    """
    Info needed to register a Mojo kernel as a PyTorch custom op.

    Built by ``max.experimental.torch`` from the kernel's ``execute``
    signature via ``KernelDeclAdaptor``.
    """

    @property
    def num_dps_outputs(self) -> int:
        """Number of leading destination-passing-style output arguments."""

    @property
    def tensor_arg_names(self) -> list[str]:
        """
        Source names of tensor-like arguments in declaration order (DPS outputs first, then inputs).
        """

class Analysis:
    """
    Analyzes Mojo kernel libraries for custom operator integration.

    Loads one or more Mojo shared libraries and exposes their exported
    kernel operations for use in MAX graphs. Used internally to validate
    and register custom operations.
    """

    def __init__(
        self, context: object, paths: Sequence[str | os.PathLike]
    ) -> None:
        """
        Creates an analysis by loading the given Mojo libraries.

        Args:
            context: A Mojo library object or path.
            paths: Additional library search paths.
        """

    @property
    def symbol_names(self) -> list[str]:
        """Returns the list of exported kernel symbol names."""

    @property
    def library_paths(self) -> list[pathlib.Path]:
        """Returns the paths of all loaded libraries."""

    def kernel(self, arg: str, /) -> max._core.Operation:
        """Returns the MLIR operation for the named kernel symbol."""

    def verify_custom_op(self, arg: max._mlir.ir.Operation, /) -> None:
        """Verifies that an operation matches the expected kernel signature."""

    def add_path(self, arg: str | os.PathLike, /) -> None:
        """Adds a library search path to this analysis."""

    def seed_kernel_decls(self, arg: max._mlir.ir.Operation, /) -> None:
        """
        Copies kernel decl ops and the opaque-type mapping attribute into the given target module. Called by the compile path to seed a model module so the GC pipeline can skip mogg-import-packages.
        """

    def kernel_torch_info(self, name: str) -> KernelTorchInfo:
        """
        Returns the info ``max.experimental.torch`` needs to register the named kernel as a PyTorch custom op. Raises ``ValueError`` if the kernel is unknown or has an unsupported argument type.
        """

    def has_shape_function(self, name: str) -> bool:
        """Returns true if a shape function is registered for the named kernel."""
