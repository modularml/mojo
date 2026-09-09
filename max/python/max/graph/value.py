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

"""Defines symbolic value types used within a :class:`~max.graph.Graph`, including tensors and buffers."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from max._core import Type as _Type
from max._core import Value as _Value
from max._core.dialects import mo
from max.driver import DLPackArray
from max.dtype import DType

from . import ops
from .dim import Dim, DimLike
from .shape import Shape, ShapeLike
from .type import (
    BufferType,
    DeviceRef,
    FilterLayout,
    TensorType,
    Type,
    _ChainType,
    _OpaqueType,
)

MlirType = TypeVar("MlirType", bound=_Type)

_SliceIndex: TypeAlias = "TensorValue | int | slice | tuple[slice, DimLike]"
_SliceIndices: TypeAlias = "Sequence[_SliceIndex | builtins.ellipsis]"


class Value(Generic[MlirType]):
    """Represents a symbolic value within a :class:`~max.graph.Graph`.

    A :class:`Value` can represent the output of a node, the arguments of a
    :class:`~max.graph.Graph` (as seen from within its body), and more generally any symbolic
    value available within the :class:`~max.graph.Graph`. Other nodes receive :class:`Value`
    values as inputs to form a computation graph.

    A :class:`Value` may also refer to an existing input or output of a node,
    and you can change them, such as by swapping a new :class:`Value`.

    Conceptually, think of a :class:`Value` as an edge in the dataflow graph,
    with the other end being the user of that value.

    The following example shows how to work with Values in a graph to create a simple computation:

    .. code-block:: python

        from max.graph import DeviceRef, Graph, ops, Value
        from max.dtype import DType
        import numpy as np

        # Create a graph context
        with Graph("value_example") as graph:
            # Create input values
            a = ops.constant(np.array([1, 2, 3], dtype=np.float32), dtype=DType.float32, device=DeviceRef.CPU())
            b = ops.constant(np.array([4, 5, 6], dtype=np.float32), dtype=DType.float32, device=DeviceRef.CPU())

            # Use values to perform operations
            c = a + b  # c is a Value representing the addition

            # Demonstrate that the result is a Value
            print(f"Type of c: {type(c)}")
            print(f"Is c a Value? {isinstance(c, Value)}")

    Similar to a regular variable, a :class:`Value` has a data type.
    """

    _mlir_value: _Value[MlirType]

    def __init__(self) -> None:
        """Value is abstract, it shouldn't be constructed directly."""
        raise NotImplementedError

    @classmethod
    def from_mlir(cls, value: _Value[MlirType]) -> Value[Any]:
        """Creates a :class:`Value` from an MLIR value.

        Args:
            value: The MLIR value to wrap.
        """
        if isinstance(value.type, mo.TensorType):
            return TensorValue.from_mlir(cast(_Value[mo.TensorType], value))
        elif isinstance(value.type, mo.ChainType):
            return _ChainValue.from_mlir(cast(_Value[mo.ChainType], value))
        elif isinstance(value.type, mo.OpaqueType):
            return _OpaqueValue.from_mlir(cast(_Value[mo.OpaqueType], value))
        elif isinstance(value.type, mo.BufferType):
            return BufferValue.from_mlir(cast(_Value[mo.BufferType], value))
        raise TypeError(f"Invalid mlir value {value=}")

    def to_mlir(self) -> _Value[MlirType]:
        """Converts the :class:`Value` to an MLIR value."""
        return self._mlir_value

    def __repr__(self) -> str:
        """Returns a string representation of the :class:`Value`."""
        return str(self._mlir_value.type)

    @property
    def buffer(self) -> BufferValue:
        """Returns the Value as a :class:`BufferValue`.

        Raises an exception if the Value is not a BufferValue.
        """
        if isinstance(self, BufferValue):
            return self

        raise TypeError(
            f"Value is not a BufferValue, was '{type(self).__name__}'"
        )

    @property
    def tensor(self) -> TensorValue:
        """Returns the Value as a :class:`TensorValue`.

        Raises an exception if the Value is not a TensorValue.
        """
        if isinstance(self, TensorValue):
            return self

        raise TypeError(
            f"Value is not a TensorValue, was '{type(self).__name__}'"
        )

    @property
    def opaque(self) -> _OpaqueValue:
        """Returns the Value as an :class:`_OpaqueValue`.

        Raises an exception if the Value is not a _OpaqueValue.
        """
        if isinstance(self, _OpaqueValue):
            return self

        raise TypeError(
            f"Value is not an _OpaqueValue, was '{type(self).__name__}'"
        )

    @property
    def type(self) -> Type[MlirType]:
        """Returns the type of the :class:`Value` as a :class:`~max.graph.Type`."""
        raise NotImplementedError


class _ChainValue(Value[mo.ChainType]):
    """Represents a chain value used to sequence side-effecting ops.

    When creating a graph, a global sequence of chains is initialized to
    sequence side-effecting ops. Every side-effecting op, such as
    ``buffer_load()``, ``buffer_store()``, and
    ``buffer_store_slice()``, consumes the current chain and produces a new
    one. Each chain can be used at most once, which prevents data races.
    """

    def __init__(self, value: Value[Any] | _Value[mo.ChainType]) -> None:
        if isinstance(value, _Value):
            assert isinstance(value.type, mo.ChainType)
            self._mlir_value = value
        elif isinstance(value, _ChainValue):
            self._mlir_value = value._mlir_value
        else:
            raise TypeError(
                "_ChainValue() argument must be an mlir.Value of chain type "
                f"or a graph._ChainValue, not {type(value).__name__!r}"
            )

    @classmethod
    def from_mlir(cls, value: _Value[mo.ChainType]) -> _ChainValue:
        return cls(value)

    @property
    def type(self) -> _ChainType:
        """Returns the type of the :class:`_ChainValue` as a :class:`~max.graph.type._ChainType`."""
        return _ChainType.from_mlir(self._mlir_value.type)


class _OpaqueValue(Value[mo.OpaqueType]):
    """Represents an opaque value within a ``Graph``."""

    def __init__(self, value: Value[Any] | _Value[mo.OpaqueType]) -> None:
        if isinstance(value, _Value):
            assert isinstance(value.type, mo.OpaqueType)
            self._mlir_value = value
        elif isinstance(value, _OpaqueValue):
            self._mlir_value = value._mlir_value
        else:
            raise TypeError(
                "_OpaqueValue() argument must be an mlir.Value of opaque type "
                f"or a graph._OpaqueValue, not {type(value).__name__!r}"
            )

    @classmethod
    def from_mlir(cls, value: _Value[mo.OpaqueType]) -> _OpaqueValue:
        return cls(value)

    @property
    def type(self) -> _OpaqueType:
        """Returns the type of the :class:`_OpaqueValue` as a :class:`~max.graph.type._OpaqueType`."""
        return _OpaqueType.from_mlir(self._mlir_value.type)


class BufferValue(Value[mo.BufferType]):
    """Represents a mutable semantic tensor within a :class:`~max.graph.Graph`.

    Unlike a :class:`TensorValue`, which is value semantic and produces a new
    value from every operation, a ``BufferValue`` refers to memory you can
    update in place. Read it into a tensor with
    :func:`~max.graph.ops.buffer_load` and write a tensor back with
    :func:`~max.graph.ops.buffer_store`. This is how a graph carries state that
    must persist across executions, such as a key-value cache.

    The following example loads a buffer input, updates it, and stores the
    result back in place:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import BufferType, DeviceRef, Graph, ops

        buffer_type = BufferType(DType.float32, shape=[4], device=DeviceRef.CPU())

        with Graph("buffer_demo", input_types=[buffer_type]) as graph:
            state = graph.inputs[0].buffer

            # Read the current contents into a value-semantic tensor.
            current = ops.buffer_load(state)

            # Write an updated tensor back into the same memory.
            ops.buffer_store(state, current + 1)
            graph.output(current)

            print(f"dtype: {state.dtype}")  # Output: dtype: DType.float32
            print(f"shape: {state.shape}")  # Output: shape: [Dim(4)]
    """

    def __init__(
        self, value: Value[Any] | _Value[mo.BufferType] | HasBufferValue
    ) -> None:
        """Initializes a :class:`BufferValue` from another value.

        Args:
            value: The value to wrap, either an MLIR value of buffer type or another :class:`BufferValue`.
        """
        if isinstance(value, _Value):
            assert isinstance(value.type, mo.BufferType)
            self._mlir_value = value
        elif isinstance(value, BufferValue):
            self._mlir_value = value._mlir_value
        elif isinstance(value, HasBufferValue):
            self._mlir_value = value.__buffervalue__()._mlir_value
        else:
            raise TypeError(
                "BufferValue() argument must be an mlir.Value of buffer type "
                f"or a graph.BufferValue, not '{type(value).__name__}'"
            )

    @classmethod
    def from_mlir(cls, value: _Value[mo.BufferType]) -> BufferValue:
        """Creates a :class:`BufferValue` from an MLIR buffer value.

        Args:
            value: The MLIR buffer value to wrap.
        """
        return cls(value)

    @property
    def type(self) -> BufferType:
        """Returns the type of the :class:`BufferValue` as a :class:`~max.graph.BufferType`."""
        return BufferType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the BufferValue."""
        return self.type.shape

    @property
    def device(self) -> DeviceRef:
        """Returns the device of the BufferValue."""
        return self.type.device

    @property
    def dtype(self) -> DType:
        """Returns the tensor data type."""
        return self.type.dtype

    @property
    def rank(self) -> int:
        """Returns the rank (number of dims) of the buffer."""
        return self.type.rank

    def __repr__(self) -> str:
        """Returns a string representation of the :class:`BufferValue`."""
        dtype = self.dtype
        shape = self.shape
        device = self.device
        return f"{type(self).__name__}({dtype=}, {shape=}, {device=})"

    def __getitem__(
        self, index: builtins.ellipsis | int | slice | _SliceIndices
    ) -> TensorValue:
        """Loads data from the buffer at the specified index.

        Args:
            index: The index or slice to access. Can be an integer, slice, or tuple of indices.
        """
        x = ops.buffer_load(self)
        if index is Ellipsis:
            return x
        return ops.slice_tensor(
            x, index if isinstance(index, Sequence) else (index,)
        )

    def __setitem__(
        self,
        index: builtins.ellipsis | int | slice | _SliceIndices,
        val: TensorValue,
    ) -> None:
        """Stores data into the buffer at the specified index.

        Args:
            index: The index or slice to store at. Can be an integer, slice, or tuple of indices.
            val: The :class:`TensorValue` to store in the buffer.
        """
        if index is Ellipsis:
            return ops.buffer_store(self, val)
        return ops.buffer_store_slice(
            self, val, index if isinstance(index, Sequence) else (index,)
        )

    def print(self, label: str = "debug_buffer") -> None:
        """Prints detailed information about the buffer."""
        ops.print(self[...], label=label)


class TensorValue(Value[mo.TensorType]):
    """Represents a value semantic tensor within a :class:`~max.graph.Graph`.

    It provides various methods and properties to manipulate and query tensor
    attributes such as :obj:`shape`, data type (:obj:`dtype`), device placement
    (:obj:`device`), and more.

    The following example demonstrates how to create and manipulate tensor values in a graph:

    .. code-block:: python

        import numpy as np
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, ops

        # Create a sample matrix
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Create a Graph context to work with tensors
        with Graph("tensor_demo") as graph:
            # Create a constant tensor from the matrix
            tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

            # Access tensor properties
            print(f"Shape: {tensor.shape}")  # Output: [2, 2]
            print(f"Data type: {tensor.dtype}")  # Output: DType.float32

            # Perform operations on the tensor
            transposed = tensor.T
            doubled = tensor * 2

            print(f"Original shape: {tensor.shape}")  # Output: [2, 2]
            print(f"Transposed shape: {transposed.shape}")  # Output: [2, 2]
    """

    # Disallow special methods that would fall back to __getitem__ and hang.
    __contains__ = None
    __iter__ = None

    def __init__(self, value: TensorValueLike) -> None:
        """Initializes a :class:`TensorValue` from a tensor-like value.

        Args:
            value: The value to wrap. Can be an MLIR tensor value, another :class:`TensorValue`,
                a :class:`~max.graph.Dim`, or a :class:`~max.graph.Shape`.
        """
        if isinstance(value, HasTensorValue):
            self._mlir_value = value.__tensorvalue__()._mlir_value
        elif isinstance(value, _Value):
            assert isinstance(value.type, mo.TensorType)
            self._mlir_value = value
        elif isinstance(value, TensorValue):
            self._mlir_value = value._mlir_value
        elif isinstance(value, Dim):
            self._mlir_value = TensorValue._from_dim(value)._mlir_value
        elif isinstance(value, Shape):
            self._mlir_value = TensorValue._from_shape(value)._mlir_value
        elif isinstance(value, _numeric):
            raise TypeError(
                "TensorValue() can not be created directly from a"
                f" '{type(value).__name__}'. Use ops.constant to"
                " convert to a TensorValue with a specific dtype."
            )
        else:
            raise TypeError(
                "TensorValue() argument must be an mlir.Value of tensor type,"
                " a graph.TensorValue, or a graph.Weight, not"
                f" '{type(value).__name__}'"
            )

    @classmethod
    def from_mlir(cls, value: _Value[mo.TensorType]) -> TensorValue:
        """Creates a :class:`TensorValue` from an MLIR tensor value.

        Args:
            value: The MLIR tensor value to wrap.
        """
        return cls(value)

    @staticmethod
    def _from_dim(dim: DimLike) -> TensorValue:
        """Creates a new tensor based on provided MLIR dimension type.

        Args:
            dim: The dimension value.
        """
        result = ops.shape_to_tensor([dim])
        result.type.device = DeviceRef.CPU()
        return result.reshape(())

    @staticmethod
    def _from_shape(shape: ShapeLike) -> TensorValue:
        """Creates a new tensor with the specified shape.

        Args:
            shape: An iterable of integers or symbolic dimensions.
        """
        result = ops.shape_to_tensor(shape)
        result.type.device = DeviceRef.CPU()
        return result

    def __repr__(self) -> str:
        """Returns a string representation of the :class:`TensorValue`."""
        dtype = self.dtype
        shape = self.shape
        device = self.device
        return f"{type(self).__name__}({dtype=}, {shape=}, {device=})"

    @cached_property
    def type(self) -> TensorType:
        """Returns the type of the :class:`TensorValue` as a :class:`~max.graph.TensorType`."""
        return TensorType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the :class:`TensorValue`.

        The following example demonstrates how to access the shape of a tensor:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x2 matrix
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("shape_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Access tensor shape
                print(f"Shape: {tensor.shape}")  # Shape: [Dim(2), Dim(2)]
        """
        return self.type.shape

    @property
    def device(self) -> DeviceRef:
        """Returns the device of the TensorValue."""
        return self.type.device

    @property
    def dtype(self) -> DType:
        """Returns the tensor data type.

        The following example demonstrates how to access the data type of a tensor:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a matrix with float32 values
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("dtype_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Access tensor data type
                print(f"Data type: {tensor.dtype}")  # Output: DType.float32
        """
        return self.type.dtype

    @property
    def rank(self) -> int:
        """Returns the rank (number of dims) of the buffer.

        The following example demonstrates how to access the rank of a tensor:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x2 matrix (2-dimensional array)
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("rank_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Access tensor rank (number of dimensions)
                print(f"Rank: {tensor.rank}")  # Output: 2
        """
        return self.type.rank

    def print(self, label: str = "debug_tensor") -> None:
        """Prints detailed information about the tensor.

        Args:
            label: A string label for the printed output. Defaults to ``debug_tensor``.
        """
        ops.print(self, label=label)

    def reshape(self, shape: ShapeLike) -> TensorValue:
        """Creates a new tensor with the same data but reshaped.

        The following example demonstrates how to reshape a tensor to change its dimensions:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x2 matrix
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("reshape_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Reshape tensor to a 1x4 matrix
                reshaped_tensor = tensor.reshape((1, 4))

                print(f"Original shape: {tensor.shape}")  # Output: [2, 2]
                print(f"Reshaped shape: {reshaped_tensor.shape}")  # Output: [1, 4]

        Args:
            shape: The new shape as an iterable of integers or symbolic dimensions.

        Returns:
            A new :class:`TensorValue` with the reshaped dimensions.
        """
        return ops.reshape(self, shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> TensorValue:
        """Flattens the specified dims of a symbolic tensor.

        The number and order of the elements in the tensor is unchanged.
        All dimensions from ``start_dim`` to ``end_dim`` (inclusive) are merged into a single output dim.

        The following example demonstrates how to flatten a multi-dimensional tensor:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x2 matrix
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("flatten_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Flatten the tensor to a 1D array
                flattened_tensor = tensor.flatten()

                print(f"Original shape: {tensor.shape}")  # Output: [2, 2]
                print(f"Flattened shape: {flattened_tensor.shape}")  # Output: [4]

        Args:
            start_dim: The starting dimension to flatten. Defaults to ``0``.
            end_dim: The ending dimension to flatten. Defaults to ``-1``.

        Returns:
            A new :class:`TensorValue` with the flattened dimensions.
        """
        return ops.flatten(self, start_dim, end_dim)

    def broadcast_to(self, shape: ShapeLike) -> TensorValue:
        """Broadcasts the tensor to a new shape.

        The following example demonstrates how to broadcast a tensor to a larger shape:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x2 matrix
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("broadcast_to_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Broadcast tensor to a 3x2x2 tensor (add a new dimension of size 3)
                broadcasted_tensor = tensor.broadcast_to((3, 2, 2))

                print(f"Original shape: {tensor.shape}")  # Output: [2, 2]
                print(f"Broadcasted shape: {broadcasted_tensor.shape}")  # Output: [3, 2, 2]

        Args:
            shape: An iterable of integers or symbolic dimensions.

        Returns:
            A new :class:`TensorValue` with the broadcasted shape.
        """
        return ops.broadcast_to(self, shape)

    def cast(self, dtype: DType) -> TensorValue:
        """Casts a symbolic tensor to a different data type.

        The following example demonstrates how to cast a tensor from one data type to another:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a matrix with float32 values
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            # Create a Graph context to work with tensors
            with Graph("cast_demo") as graph:
                # Create a constant tensor from the matrix
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Cast tensor to integer type
                casted_tensor = tensor.cast(DType.int32)

                print(f"Original dtype: {tensor.dtype}")  # Output: DType.float32
                print(f"Casted dtype: {casted_tensor.dtype}")  # Output: DType.int32

        Args:
            dtype: The target data type (for example, ``DType.int32``, ``DType.float64``).

        Returns:
            A new :class:`TensorValue` with the casted data type.
        """
        return ops.cast(self, dtype)

    def rebind(self, shape: ShapeLike, message: str = "") -> TensorValue:
        """Asserts that the tensor has the specified shape at runtime.

        This method adds a runtime assertion that the tensor's shape matches
        the given ``shape``. It does not modify the tensor's data or actual
        shape; instead, it verifies the shape constraint and propagates
        the asserted shape information to subsequent operations in the graph.

        Use ``rebind()`` when you need to:

        - Constrain a tensor with dynamic or unknown dimensions to specific
          fixed sizes required by a downstream operation.
        - Assert shape invariants that the compiler cannot infer statically.
        - Provide shape hints to enable compiler optimizations.

        If the assertion fails at runtime (the actual shape does not match
        ``shape``), an error is raised with the optional ``message``.

        Args:
            shape: The expected shape as an iterable of integers or symbolic
                dimensions (:class:`~max.graph.Dim`). The rank must match the
                tensor's current rank.
            message: An optional message to include in the error if the
                assertion fails at runtime.

        Returns:
            A new :class:`TensorValue` with the same data but with the
            symbolic shape set to the asserted ``shape``.

        Raises:
            ValueError: If the rank of ``shape`` does not match the tensor's
                rank.
        """
        return ops.rebind(self, shape, message)

    def _with_layout(self, layout: FilterLayout) -> TensorValue:
        """Rebinds the tensor with a known layout for convolution filters.

        Args:
            layout: The layout value.

        Returns:
            A new :class:`TensorValue` with the known layout.
        """
        return ops.rebind(self, shape=self.shape, layout=layout)

    def permute(self, dims: list[int]) -> TensorValue:
        """Permutes the tensor's dimensions based on provided indices.

        Args:
            dims: A list of integers specifying the new order of dimensions.

        Returns:
            A new :class:`TensorValue` with permuted dimensions.
        """
        return ops.permute(self, dims)

    def transpose(self, dim_1: int, dim_2: int) -> TensorValue:
        """Swaps two dimensions of the tensor.

        The following example demonstrates how to transpose a tensor by swapping its dimensions:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops

            # Create a 2x3 matrix
            matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

            with Graph("transpose_demo") as graph:
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Transpose the tensor (swap dimensions 0 and 1)
                transposed_tensor = tensor.transpose(dim_1=0, dim_2=1)

                print(f"Original shape: {tensor.shape}")  # Output: [2, 3]
                print(f"Transposed shape: {transposed_tensor.shape}")  # Output: [3, 2]

        Args:
            dim_1: The first dimension to swap.
            dim_2: The second dimension to swap.

        Returns:
            A new :class:`TensorValue` with swapped dimensions.
        """
        return ops.transpose(self, dim_1, dim_2)

    def to(self, device: DeviceRef) -> TensorValue:
        """Inserts a graph-level transfer to ``device`` into the compiled graph.

        This is a **graph execution-time** operation: it records a transfer
        node during graph tracing that moves this symbolic tensor to ``device``
        when the compiled graph runs. It is equivalent to calling
        :func:`~max.graph.ops.transfer_to` and is typically used inside
        :meth:`forward` to route activation tensors between devices.

        This is distinct from :meth:`~max.experimental.nn.Module.to`, which is a
        **pre-compilation** host-side operation that moves stored weight
        tensors before the graph is built. If you want to place a module's
        weights and computation on a device, use ``Module.to(device)`` before
        calling :meth:`~max.experimental.nn.Module.compile`.

        The following example demonstrates how to move a tensor from one device to another:

        .. code-block:: python

            import numpy as np
            from max.dtype import DType
            from max.graph import Graph, ops, DeviceRef

            # Create a 2x2 matrix
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)

            with Graph("to_device_example") as graph:
                # Create a tensor on the default device
                tensor = ops.constant(matrix, dtype=DType.float32, device=DeviceRef.CPU())

                # Move the tensor to a GPU device
                gpu_tensor = tensor.to(DeviceRef.GPU())

                print(f"Original device: {tensor.device}")  # Output depends on default device
                print(f"New device: {gpu_tensor.device}")  # Output: gpu:0

        Args:
            device: A :class:`~max.graph.DeviceRef` object specifying the target device.

        Returns:
            A new :class:`TensorValue` on the specified device.
        """
        return ops.transfer_to(self, device)

    def argmax(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using an argmax operation along ``axis``.

        When the result is ambiguous ie. there are multiple maxima,
        selects one index arbitrarily.

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("argmax_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Argmax along axis 1 (last dimension of each row)
                indices = x.argmax(axis=1)

                print(f"Input shape: {x.shape}")       # [2, 3]
                print(f"Argmax shape: {indices.shape}")  # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` of dtype ``DType.int64`` with the same rank as the input,
            and the same shape except along ``axis``, which will have size 1.
        """
        return ops.argmax(self, axis=axis)

    def max(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using a max operation along ``axis``.

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("max_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Max along axis 1 (last dimension of each row)
                m = x.max(axis=1)

                print(f"Input shape: {x.shape}")  # [2, 3]
                print(f"Max shape: {m.shape}")    # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` with the same rank as the input and the same
            shape except along ``axis``, which will have size 1.
        """
        return ops.max(self, axis=axis)

    def mean(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using a mean operation along ``axis``.

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("mean_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Mean along axis 1 (last dimension of each row)
                mu = x.mean(axis=1)

                print(f"Input shape: {x.shape}")  # [2, 3]
                print(f"Mean shape: {mu.shape}")  # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` with the same rank as the input and the same
            shape except along ``axis``, which will have size 1.
        """
        return ops.mean(self, axis=axis)

    def min(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using a min operation along ``axis``.

        .. code-block:: python

            from max.dtype import DType

            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("min_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Min along axis 1 (last dimension of each row)
                mn = x.min(axis=1)

                print(f"Input shape: {x.shape}")  # [2, 3]
                print(f"Min shape: {mn.shape}")   # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` with the same rank as the input and the same
            shape except along ``axis``, which will have size 1.
        """
        return ops.min(self, axis=axis)

    def stdev(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using a standard deviation operation along ``axis``.

        The standard deviation is computed as the square root of the population
        variance along the specified axis.

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("stdev_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Standard deviation along axis 1 (last dimension of each row)
                sd = x.stdev(axis=1)

                print(f"Input shape: {x.shape}")    # [2, 3]
                print(f"Stdev shape: {sd.shape}")  # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` with the same rank as the input and the same
            shape except along ``axis``, which will have size 1.
        """
        return ops.sqrt(self.var(axis=axis))

    def var(self, axis: int = -1) -> TensorValue:
        """Reduces the tensor using a variance operation along ``axis``.

        The variance is computed as the mean of squared deviations from the mean
        (population variance, i.e., without Bessel's correction) along the specified axis.

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, TensorType, DeviceRef

            # Define a 2x3 float32 input tensor for the graph
            input_type = TensorType(DType.float32, (2, 3), device=DeviceRef.CPU())
            with Graph("var_demo", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor

                # Variance along axis 1 (last dimension of each row)
                vr = x.var(axis=1)

                print(f"Input shape: {x.shape}")  # [2, 3]
                print(f"Var shape: {vr.shape}")  # [2, 1]

        Args:
            axis: The axis along which to compute the reduction. If negative,
                indexes from the last dimension (for example, ``-1`` is the last dimension).

        Returns:
            A :class:`TensorValue` with the same rank as the input and the same
            shape except along ``axis``, which will have size 1.
        """
        if self.dtype is DType.bool:
            raise TypeError("Variance undefined for boolean values.")
        return ((self - self.mean(axis=axis)) ** 2).mean(axis=axis)

    @property
    def T(self) -> TensorValue:
        """Returns the transposed tensor.

        ``T`` is the shorthand notation for transposing.
        For more information, see :meth:`transpose`.

        Returns:
            A new :class:`TensorValue` with swapped dimensions.
        """
        return self.transpose(-1, -2)

    def __getitem__(self, index: Any) -> TensorValue:
        """Extracts a slice or subset of the tensor.

        Args:
            index: The index or slice to access. Can be an integer, slice, ellipsis, or tuple of indices.
        """
        if isinstance(index, TensorValue) or not isinstance(index, Iterable):
            index = (index,)
        return ops.slice_tensor(self, index)

    def __eq__(self, rhs: object) -> TensorValue:  # type: ignore[override]
        """Performs element-wise equality comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        if _is_tensor_value_like(rhs):
            return ops.equal(self, rhs)
        else:
            raise TypeError(
                "'==' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __neg__(self) -> TensorValue:
        """Performs element-wise negation."""
        return ops.negate(self)

    def __round__(self) -> TensorValue:
        """Rounds to the elementwise nearest integer, with ties going towards the nearest even number."""
        return ops.round(self)

    def __ne__(self, rhs: object) -> TensorValue:  # type: ignore[override]
        """Performs element-wise inequality comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        if _is_tensor_value_like(rhs):
            return ops.not_equal(self, rhs)
        else:
            raise TypeError(
                "'!=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __ge__(self, rhs: Any) -> TensorValue:
        """Performs element-wise greater-than-or-equal comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        if _is_tensor_value_like(rhs):
            return ops.greater_equal(self, rhs)
        else:
            raise TypeError(
                "'>=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __gt__(self, rhs: Any) -> TensorValue:
        """Performs element-wise greater-than comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        if _is_tensor_value_like(rhs):
            return ops.greater(self, rhs)
        else:
            raise TypeError(
                f"'>' not supported between instance of '{type(self).__name__}'"
                f" and '{type(rhs).__name__}'"
            )

    def __lt__(self, rhs: Any) -> TensorValue:
        """Performs element-wise less-than comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        return ops.logical_not(self >= rhs)

    def __le__(self, rhs: Any) -> TensorValue:
        """Performs element-wise less-than-or-equal comparison.

        Args:
            rhs: The right-hand side operand for comparison. Must be tensor-like.
        """
        return ops.logical_not(self > rhs)

    def __add__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise addition.

        Args:
            rhs: The right-hand side operand for addition. Must be tensor-like.
        """
        return ops.add(self, rhs)

    def __radd__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise addition with reversed operands.

        Args:
            lhs: The left-hand side operand for addition. Must be tensor-like.
        """
        return ops.add(lhs, self)

    def __sub__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise subtraction.

        Args:
            rhs: The right-hand side operand for subtraction. Must be tensor-like.
        """
        return ops.sub(self, rhs)

    def __rsub__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise subtraction with reversed operands.

        Args:
            lhs: The left-hand side operand for subtraction. Must be tensor-like.
        """
        return ops.sub(lhs, self)

    def __mul__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise multiplication.

        Args:
            rhs: The right-hand side operand for multiplication. Must be tensor-like.
        """
        return ops.mul(self, rhs)

    def __rmul__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise multiplication with reversed operands.

        Args:
            lhs: The left-hand side operand for multiplication. Must be tensor-like.
        """
        return ops.mul(lhs, self)

    def __truediv__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise division.

        Args:
            rhs: The right-hand side operand for division. Must be tensor-like.
        """
        return ops.div(self, rhs)

    def __rtruediv__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise division with reversed operands.

        Args:
            lhs: The left-hand side operand for division. Must be tensor-like.
        """
        return ops.div(lhs, self)

    def __floordiv__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise floor division.

        Args:
            rhs: The right-hand side operand for floor division. Must be tensor-like.
        """
        return ops.floor(ops.div(self, rhs))

    def __rfloordiv__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise floor division with reversed operands.

        Args:
            lhs: The left-hand side operand for floor division. Must be tensor-like.
        """
        return ops.floor(ops.div(lhs, self))

    def __mod__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise modulo operation.

        Args:
            rhs: The right-hand side operand for modulo. Must be tensor-like.
        """
        return ops.mod(self, rhs)

    def __rmod__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise modulo operation with reversed operands.

        Args:
            lhs: The left-hand side operand for modulo. Must be tensor-like.
        """
        return ops.mod(lhs, self)

    def __divmod__(
        self, rhs: TensorValueLike
    ) -> tuple[TensorValue, TensorValue]:
        """Performs element-wise division and modulo operation simultaneously.

        Args:
            rhs: The right-hand side operand for divmod. Must be tensor-like.
        """
        return (self // rhs, self % rhs)

    def __rdivmod__(
        self, lhs: TensorValueLike
    ) -> tuple[TensorValue, TensorValue]:
        """Performs element-wise division and modulo operation with reversed operands.

        Args:
            lhs: The left-hand side operand for divmod. Must be tensor-like.
        """
        return (lhs // self, lhs % self)

    def __matmul__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs matrix multiplication.

        Args:
            rhs: The right-hand side operand for matrix multiplication. Must be tensor-like.
        """
        return ops.matmul(self, rhs)

    def __rmatmul__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs matrix multiplication with reversed operands.

        Args:
            lhs: The left-hand side operand for matrix multiplication. Must be tensor-like.
        """
        return ops.matmul(lhs, self)

    def __pow__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise exponentiation.

        Args:
            rhs: The right-hand side operand for exponentiation. Must be tensor-like.
        """
        return ops.pow(self, rhs)

    def __rpow__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise exponentiation with reversed operands.

        Args:
            lhs: The left-hand side operand for exponentiation. Must be tensor-like.
        """
        return ops.pow(lhs, self)

    def __and__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical AND operation.

        Args:
            rhs: The right-hand side operand for logical AND. Must be tensor-like.
        """
        return ops.logical_and(self, rhs)

    def __rand__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical AND operation with reversed operands.

        Args:
            lhs: The left-hand side operand for logical AND. Must be tensor-like.
        """
        return ops.logical_and(lhs, self)

    def __or__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical OR operation.

        Args:
            rhs: The right-hand side operand for logical OR. Must be tensor-like.
        """
        return ops.logical_or(self, rhs)

    def __ror__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical OR operation with reversed operands.

        Args:
            lhs: The left-hand side operand for logical OR. Must be tensor-like.
        """
        return ops.logical_or(lhs, self)

    def __xor__(self, rhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical XOR operation.

        Args:
            rhs: The right-hand side operand for logical XOR. Must be tensor-like.
        """
        return ops.logical_xor(self, rhs)

    def __rxor__(self, lhs: TensorValueLike) -> TensorValue:
        """Performs element-wise logical XOR operation with reversed operands.

        Args:
            lhs: The left-hand side operand for logical XOR. Must be tensor-like.
        """
        return ops.logical_xor(lhs, self)

    def __invert__(self) -> TensorValue:
        """Performs element-wise logical NOT operation."""
        return ops.logical_not(self)


@runtime_checkable
class HasTensorValue(Protocol):
    """Protocol for objects convertible to a :class:`TensorValue`."""

    def __tensorvalue__(self) -> TensorValue: ...


@runtime_checkable
class HasBufferValue(Protocol):
    """Protocol for objects convertible to a :class:`BufferValue`."""

    def __buffervalue__(self) -> BufferValue: ...


Numeric = int | float | np.integer[Any] | np.floating[Any] | DLPackArray
Scalar = int | float | np.integer[Any] | np.floating[Any] | Dim
StrongTensorValueLike = (
    _Value[mo.TensorType] | TensorValue | Shape | Dim | HasTensorValue
)

TensorValueLike = StrongTensorValueLike | Numeric
BufferValueLike = BufferValue | HasBufferValue

# This is needed for python 3.9 compatibility.
# `isinstance` only works with tuples and not unions in 3.9.
_numeric = (int, float, np.integer, np.floating, np.ndarray)
_scalar = (int, float, np.integer, np.floating, Dim)
_strong_tensor_value_like = (_Value[mo.TensorType], TensorValue, Shape, Dim)
_tensor_value_like = _strong_tensor_value_like + _numeric


def _is_numeric(obj: Any) -> TypeGuard[Numeric]:
    return isinstance(obj, _numeric)


def _is_scalar(obj: Any) -> TypeGuard[Scalar]:
    return isinstance(obj, _scalar)


def _is_strong_tensor_value_like(obj: Any) -> TypeGuard[StrongTensorValueLike]:
    return isinstance(obj, TensorValue | Shape | Dim | HasTensorValue) or (
        isinstance(obj, _Value) and isinstance(obj.type, mo.TensorType)
    )


def _is_tensor_value_like(obj: Any) -> TypeGuard[TensorValueLike]:
    return _is_strong_tensor_value_like(obj) or isinstance(obj, _numeric)
