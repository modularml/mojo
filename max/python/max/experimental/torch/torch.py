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

"""Provides PyTorch integration utilities and custom operator support for MAX."""

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from concurrent import futures
from functools import partial
from pathlib import Path
from typing import Any, overload

from max._core import Operation
from max._mlir_context import (
    MLIRThreadPoolExecutor,
    call_with_default_mlir_context,
)
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    KernelLibrary,
    Shape,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)

try:
    import torch  # type: ignore
    from torch._library.custom_ops import CustomOpDef  # type: ignore
except ImportError:
    raise ImportError(  # noqa: B904
        "torch not found - install `max[torch]` (if using pip/uv) or max-conda (if using magic/conda)"
    )


# Torch dtype conversion mappings
_MAX_DTYPE_TO_TORCH = {
    DType.bool: torch.bool,
    DType.int8: torch.int8,
    DType.int16: torch.int16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint8: torch.uint8,
    DType.uint16: torch.uint16,
    DType.uint32: torch.uint32,
    DType.uint64: torch.uint64,
    DType.float16: torch.float16,
    DType.float32: torch.float32,
    DType.float64: torch.float64,
    DType.bfloat16: torch.bfloat16,
    DType.float8_e8m0fnu: torch.float8_e8m0fnu,
    DType.float8_e4m3fn: torch.float8_e4m3fn,
    DType.float8_e4m3fnuz: torch.float8_e4m3fnuz,
    DType.float8_e5m2: torch.float8_e5m2,
    DType.float8_e5m2fnuz: torch.float8_e5m2fnuz,
}

_TORCH_TO_MAX_DTYPE = {v: k for k, v in _MAX_DTYPE_TO_TORCH.items()}


def max_dtype_to_torch(dtype: DType) -> torch.dtype:
    """Converts a MAX DType to the corresponding torch dtype.

    Args:
        dtype: The MAX DType to convert.

    Returns:
        The corresponding torch dtype object.

    Raises:
        ValueError: If the dtype is not supported.
    """
    if torch_dtype := _MAX_DTYPE_TO_TORCH.get(dtype):
        return torch_dtype
    raise ValueError(f"unsupported MAX DType to convert to torch: {dtype}")


def torch_dtype_to_max(dtype: torch.dtype) -> DType:
    """Converts a torch dtype to the corresponding MAX DType.

    Args:
        dtype: The torch dtype to convert.

    Returns:
        The corresponding MAX DType enum value.

    Raises:
        ValueError: If the input dtype is not supported.
    """
    if max_dtype := _TORCH_TO_MAX_DTYPE.get(dtype):
        return max_dtype
    raise ValueError(f"unsupported torch dtype to convert to MAX: {dtype}")


class CustomOpLibrary:
    """A PyTorch interface to custom operations implemented in Mojo.

    This API allows for easy passing of PyTorch data as
    ``torch.Tensor`` values to the corresponding custom op. ``CustomOpLibrary``
    handles the compilation of the Mojo custom ops and marshalling of data between
    PyTorch and the executable Mojo code.

    For example, consider a grayscale operation implemented in Mojo:

    .. code-block:: mojo
       :caption: my_library/grayscale.mojo

        @register("grayscale")
        struct Grayscale:
            @staticmethod
            fn execute[
                # The kind of device this is running on: "cpu" or "gpu"
                target: StaticString,
            ](
                img_out: OutputTensor[dtype = DType.uint8, rank=2],
                img_in: InputTensor[dtype = DType.uint8, rank=3],
                ctx: DeviceContext,
            ) raises:
                ...

    You can then use ``CustomOpLibrary`` to invoke the Mojo operation like so:

    .. code-block:: python

        import torch
        from max.experimental.torch import CustomOpLibrary

        op_library = CustomOpLibrary("my_library")
        grayscale_op = op_library.grayscale

        def grayscale(pic: torch.Tensor) -> torch.Tensor:
            result = pic.new_empty(pic.shape[:-1])
            grayscale_op(result, pic)
            return result

        img = (torch.rand(64, 64, 3) * 255).to(torch.uint8)
        result = grayscale(img)

    The custom operation produced by ``op_library.<opname>`` will have the
    same interface as the backing Mojo operation. Each ``InputTensor`` or
    ``OutputTensor`` argument corresponds to a
    :code_link:`https://docs.pytorch.org/docs/stable/tensors.html#tensor-class-reference|torch.Tensor`
    value in Python. Each argument corresponding to an ``OutputTensor`` in the
    Mojo operation will be modified in-place.

    For more information, see the [custom ops for PyTorch](/develop/custom-kernels-pytorch) tutorial.

    Args:
        kernel_library: The path to a ``.mojo`` file or a ``.mojoc`` with
            your custom op kernels, or the corresponding library object.
    """

    _kernel_library: KernelLibrary
    _session: InferenceSession
    _ops: dict[str, CustomOp]
    _deferred_errors: dict[str, Exception]

    def __init__(self, kernel_library: Path | KernelLibrary) -> None:
        devices = [Accelerator(i) for i in range(accelerator_count())]

        if isinstance(kernel_library, KernelLibrary):
            self._kernel_library = kernel_library
        else:
            self._kernel_library = KernelLibrary()
            self._kernel_library.load_paths([kernel_library])

        self._session = InferenceSession(devices=[*devices])

        # Eagerly register all ops so that __getattr__ is a trivial
        # dict lookup.  This is required for torch.compile with
        # fullgraph=True: Dynamo traces into __getattr__ and cannot
        # handle threading.Lock context managers or complex init
        # logic.  Ops that fail to register (e.g. unsupported arg
        # types) have their exceptions stored and re-raised on access.
        self._ops: dict[str, CustomOp] = {}
        self._deferred_errors: dict[str, Exception] = {}
        for name in self._kernel_library:
            try:
                self._ops[name] = CustomOp(self, name)
            except Exception as e:
                # The kernel library iterator yields internal ops
                # (MOGG primitives, EP ops, etc.) that are not meant
                # for PyTorch registration and fail in various ways:
                # ValueError (unsupported arg types), RuntimeError
                # (schema parsing), KeyError (missing MLIR attrs).
                self._deferred_errors[name] = e

    def __getattr__(self, attr: str) -> CustomOp:
        """Get a custom op from the library registered with the given name."""
        if attr in self._ops:
            return self._ops[attr]
        if attr in self._deferred_errors:
            raise self._deferred_errors[attr]
        raise AttributeError(
            f"custom library does not register operation {attr}"
        )


ParametersDict = Mapping[str, bool | int | str | DType]
ParameterKey = tuple[str, bool | int | str | DType]

CompiledModelKey = tuple[Hashable, ...]


class CustomOp:
    """A Mojo-implemented custom op registered for use in MAX graphs."""

    library: CustomOpLibrary
    name: str
    parameters: ParametersDict | None

    _custom_op_def: CustomOpDef
    _parameter_specializations: dict[tuple[ParameterKey, ...], CustomOp]

    def __init__(
        self,
        library: CustomOpLibrary,
        name: str,
        parameters: ParametersDict | None = None,
    ) -> None:
        self.library = library
        self.name = name
        self.parameters = parameters

        # Decode the kernel's execute signature once via KernelDeclAdaptor on
        # the C++ side. Raises ValueError if the kernel is unknown or has an
        # unsupported argument type (caught by CustomOpLibrary.__init__ for
        # internal ops / primitives that aren't meant for PyTorch).
        self._torch_info = library._kernel_library._analysis.kernel_torch_info(
            name
        )

        if parameters:
            suffix = "".join(
                f"{key}_{value}" for key, value in sorted(parameters.items())
            )
            name = f"{name}_{suffix}"
        op = MaxOp(self, name, self.library, num_outputs=self.num_outputs)
        self._custom_op_def = op.custom_op_def()

        self._parameter_specializations = {}

    def __getitem__(self, parameters: ParametersDict) -> CustomOp:
        """Parameterizes the custom op Mojo function with the given parameters."""
        if self.parameters is not None:
            raise TypeError("Parameters already specified")

        normalized = tuple(sorted(parameters.items()))

        if not (result := self._parameter_specializations.get(normalized)):
            result = CustomOp(self.library, self.name, parameters)
            self._parameter_specializations[normalized] = result

        return result

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> None:
        """Calls the PyTorch custom op with the given arguments."""
        return self._custom_op_def(*args, **kwargs)

    @property
    def kernel(self) -> Operation:
        """Retrieves the op definition MLIR from the custom op library."""
        analysis = self.library._kernel_library._analysis
        return analysis.kernel(self.name)

    @property
    def num_outputs(self) -> int:
        """Returns the number of outputs of the custom op."""
        # TODO(GEX-2219): support non-dps outputs
        return self._torch_info.num_dps_outputs

    def op(self, *args: TensorValue, result_types: Sequence[TensorType]):  # noqa: ANN201
        """Builds a MAX graph custom op with the given inputs and output types.

        The device is inferred from the input or result types, or defaults to
        CPU.

        Args:
            *args: The input graph tensor values to the custom op.
            result_types: The types of the op's output tensors.

        Returns:
            The graph value(s) produced by the custom op.
        """
        # Infer custom op device from inputs
        device = next(
            itertools.chain(
                (arg.type.device for arg in args),
                (type.device for type in result_types),
                (DeviceRef.CPU(0),),
            )
        )
        return ops.custom(
            self.name,
            device,
            args,
            out_types=result_types,
            parameters=self.parameters,
        )

    @property
    def torch_signature(self) -> inspect.Signature:
        """Compute the Python-level signature of the provided custom op.

        The computed signature is derived from the kernel's stored ``execute``
        signature, decoded on the C++ side by ``KernelDeclAdaptor``. The
        synthesised signature has one ``torch.Tensor`` parameter per
        tensor-like argument (DPS outputs first, then inputs); device contexts
        are supplied implicitly by the graph compiler and don't appear here.

        Returns:
            inspect.Signature: The Python-level signature for the custom op.
        """
        args = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for name in self._torch_info.tensor_arg_names
        ]
        return inspect.Signature(args, return_annotation=None)


class MaxOp:
    """A MAX graph operation registered as a callable for use in PyTorch."""

    fn: Callable[..., Iterable[Value[Any]] | Value[Any] | None] | CustomOp
    name: str
    library: CustomOpLibrary
    # Specify explicit input types to eg. compile with symbolic dims
    input_types: tuple[TensorType, ...] | None = None
    output_types: tuple[TensorType, ...] | None = None
    num_outputs: int

    def __init__(
        self,
        fn: Callable[..., Iterable[Value[Any]] | Value[Any] | None] | CustomOp,
        name: str,
        library: CustomOpLibrary,
        # Specify explicit input types to eg. compile with symbolic dims
        input_types: Sequence[TensorType] | None = None,
        output_types: Sequence[TensorType] | None = None,
        num_outputs: int | None = None,
    ) -> None:
        if num_outputs is None:
            self.num_outputs = 1 if output_types is None else len(output_types)
        elif output_types is not None and len(output_types) != num_outputs:
            raise ValueError(f"{num_outputs=} does not match {output_types=}")
        else:
            self.num_outputs = num_outputs

        self.fn = fn
        self.name = name
        self.library = library
        self.input_types = None if input_types is None else tuple(input_types)
        self.output_types = (
            None if output_types is None else tuple(output_types)
        )

    @property
    def torch_signature(self) -> inspect.Signature:
        """Computes the signature of the generated PyTorch custom op."""
        if isinstance(self.fn, CustomOp):
            return self.fn.torch_signature

        base_signature = inspect.signature(self.fn)
        dps_args = [
            inspect.Parameter(
                f"__out{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for i in range(self.num_outputs)
        ]
        args = [
            param.replace(annotation=torch.Tensor)
            for param in base_signature.parameters.values()
        ]
        return inspect.Signature((*dps_args, *args), return_annotation=None)

    def graph(
        self,
        input_types: Sequence[TensorType],
        result_types: Sequence[TensorType],
    ) -> Graph:
        """Builds a MAX graph with the given input and output types.

        The created graph will be destination-passing-style; the output
        types will be converted to buffer types so that inputs may be
        passed as mutable arguments to the compiled model.

        Args:
            input_types: The input types of the graph.
            result_types: The output types of the graph.
                Required for destination-passing-style.

        Returns:
            A MAX graph that executes the op.
        """
        output_types = [t.as_buffer() for t in result_types]
        graph_types: list[Type[Any]] = [*output_types, *input_types]

        with Graph(
            self.name,
            input_types=graph_types,
            kernel_library=self.library._kernel_library,
        ) as graph:
            # Awkward design, there's probably a better way here
            fn: Callable[..., Iterable[Value[Any]] | Value[Any] | None] = (
                partial(self.fn.op, result_types=result_types)
                if isinstance(self.fn, CustomOp)
                else self.fn
            )
            args = (input.tensor for input in graph.inputs[self.num_outputs :])
            results = fn(*args)
            iterable_results: Iterable[Value[Any]]
            if isinstance(results, Value):
                iterable_results = [results]
            elif results is None:
                iterable_results = []
            else:
                iterable_results = results
            for input, result in zip(
                graph.inputs, iterable_results, strict=False
            ):
                input.buffer[...] = result.tensor

            graph.output()

        return graph

    def custom_op_def(self) -> CustomOpDef:
        """Builds and registers a PyTorch custom operation with this MAX graph."""
        # This will hold the compiled model once the registered fake tensor function
        # is invoked for the first time.
        model_cache: dict[str, futures.Future[Model]] = {}
        model_cache_lock = threading.Lock()
        executor = MLIRThreadPoolExecutor()

        signature: inspect.Signature = self.torch_signature
        mutated_args = list(signature.parameters.keys())[: self.num_outputs]

        # Compile the model if it has not been compiled already.
        def compiled_model(args: tuple[torch.Tensor, ...]) -> Model:
            # args are destination-passing-style
            output_key = self.output_types or args[: self.num_outputs]
            input_key = self.input_types or args[self.num_outputs :]

            key = ",".join(
                f"{tensor.dtype} {tensor.shape} {tensor.device}"
                for tensor in (output_key + input_key)
            )

            # Only one thread can schedule model compilation for a given key
            with model_cache_lock:
                if not (model_future := model_cache.get(key)):
                    arg_types = tuple(max_tensor_type(arg) for arg in args)
                    # args are destination-passing-style
                    output_types = (
                        self.output_types or arg_types[: self.num_outputs]
                    )
                    input_types = (
                        self.input_types or arg_types[self.num_outputs :]
                    )

                    graph = self.graph(input_types, output_types)

                    model_cache[key] = model_future = executor.submit(
                        self.library._session.load, graph
                    )
            return model_future.result()

        # Torch `__dlpack__(stream=...)` has substantial overhead.
        # - Manually retrieving and syncing the stream drops dlpack marshalling
        #   from ~60us per tensor to ~15us per tensor.
        # - Further optimizations are possible. Moving more of this behavior
        #   into a single C++ ffi call can drop overhead to ~2us.
        # - Generally users shouldn't be putting this marshalling into their
        #   inner loop. Gains are much more substantial for larger graphs
        #   which can take advantage of MAX's automatic kernel fusion.
        def fast_from_dlpack(t: torch.Tensor):  # noqa: ANN202
            if t.device.type == "cuda":
                stream = torch.cuda.current_stream(t.device).cuda_stream
                device = max_device(t.device)
                data = t.__dlpack__()
                try:
                    return Buffer._from_dlpack(data, device, stream)
                except Exception:
                    # This approach fails when passing the tensor across threads.
                    # Fall back to letting torch slowly sync streams.
                    return Buffer.from_dlpack(t)
            return Buffer.from_dlpack(t)

        # ops always have no return type! they assign their results to mutable buffers
        def callable(*args: torch.Tensor) -> None:
            # In eager mode, the fake_tensor function will not be called,
            # so we call it here.
            # registered_fake with real inputs will create buffers for the outputs
            def execute() -> None:
                model = compiled_model(args)
                converted = [fast_from_dlpack(arg) for arg in args]
                model.execute(*converted)

            call_with_default_mlir_context(execute)

        name = f"max::torch.{self.name}"
        callable.__signature__ = signature  # type: ignore
        return torch.library.custom_op(
            name, callable, mutates_args=mutated_args
        )


@overload
def graph_op(
    fn: Callable[..., Value[Any] | None],
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> CustomOpDef: ...


@overload
def graph_op(
    fn: None = None,
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> Callable[
    [Callable[..., Iterable[Value[Any]] | Value[Any] | None]], CustomOpDef
]: ...


def graph_op(
    fn: Callable[..., Iterable[Value[Any]] | Value[Any] | None] | None = None,
    name: str | None = None,
    kernel_library: Path | KernelLibrary | None = None,
    input_types: Sequence[TensorType] | None = None,
    output_types: Sequence[TensorType] | None = None,
    num_outputs: int | None = None,
) -> (
    CustomOpDef
    | Callable[
        [Callable[..., Iterable[Value[Any]] | Value[Any] | None]], CustomOpDef
    ]
):
    """A decorator to create PyTorch custom operations using MAX graph operations.

    This decorator allows you to define larger graphs using the functions in
    :mod:`max.graph.ops` or :mod:`max.nn` modules and
    call them with PyTorch tensors, or integrate them into PyTorch modules.
    These custom ops can be called eagerly, and support compilation with
    ``torch.compile`` and the Inductor backend.

    The resulting custom operation uses destination-passing style, where output
    tensors are passed as the first arguments and modified in-place. This
    allows PyTorch to manage the memory and streams of the output tensors.
    Tensors internal to the computation are managed via MAX's graph compiler
    and memory planning.

    The default behavior is to JIT-compile for the specific input and output
    shapes needed. If you are passing variable-sized inputs, for instance a
    batch size or sequence length which may take on many different values
    between calls, you should specify this dimension as a symbolic dimension
    through :obj:`input_types` and :obj:`output_types`. Otherwise you will
    end up compiling specialized graphs for each possible variation of
    inputs, which may use a lot of memory.

    If neither `output_types` nor `num_outputs` is specified, default to 1
    output.

    For example to create a functional-style PyTorch op backed by MAX:

    .. code-block:: python
        :caption: grayscale.py

        import torch
        import numpy as np
        import max.experimental.torch
        from max.dtype import DType
        from max.graph import ops

        @max.experimental.torch.graph_op
        def max_grayscale(pic: max.graph.TensorValue):
            scaled = pic.cast(DType.float32) * np.array([0.21, 0.71, 0.07])
            grayscaled = ops.sum(scaled, axis=-1).cast(pic.dtype)
            # max reductions don't remove the dimension, need to squeeze
            return ops.squeeze(grayscaled, axis=-1)

        @torch.compile
        def grayscale(pic: torch.Tensor):
            output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
            max_grayscale(output, pic)  # Call as destination-passing style
            return output

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = (torch.rand(64, 64, 3, device=device) * 255).to(torch.uint8)
        result = grayscale(img)
        print(f"Input shape: {img.shape}")
        print(f"Output shape: {result.shape}")
        print("Grayscale conversion completed successfully!")

    Args:
        fn: The function to decorate. If None, returns a decorator.
        name: Optional name for the custom operation. Defaults to the function name.
        kernel_library: Optional kernel library to use for compilation. Useful
            for creating graphs with custom Mojo ops.
        input_types: Optional sequence of input tensor types for compilation.
            If None, types are inferred from runtime arguments.
        output_types: Optional sequence of output tensor types for compilation.
            If None, types are inferred from runtime arguments.
        num_outputs: The number of outputs of the graph. We need to know this ahead
            of time to register with PyTorch before we've compiled the final kernels.

    Returns:
        A PyTorch custom operation that can be called with torch.Tensor arguments.
    """

    def decorator(
        fn: Callable[..., Iterable[Value[Any]] | Value[Any] | None],
    ) -> CustomOpDef:
        library = kernel_library or KernelLibrary()
        op = MaxOp(
            fn,
            name or fn.__name__,
            CustomOpLibrary(library),
            input_types=input_types,
            output_types=output_types,
            num_outputs=num_outputs,
        )
        return op.custom_op_def()

    return decorator if fn is None else decorator(fn)


###############################################################################

# Convert torch.Tensor to a TensorType

###############################################################################


def max_shape(shape: torch.Size) -> Shape:
    """Returns the equivalent MAX shape for a PyTorch shape."""
    return Shape([int(dim) for dim in shape])


def max_device_ref(device: torch.device) -> DeviceRef:
    """Returns the equivalent MAX graph device for a PyTorch device."""
    type = device.type
    index = device.index or 0
    if type == "cpu":
        return DeviceRef.CPU(index)
    elif type == "cuda":
        return DeviceRef.GPU(index)
    else:
        raise TypeError(f"Unable to convert {type} to a MAX device type.")


def max_device(device: torch.device) -> Device:
    """Returns the equivalent MAX device for a PyTorch device."""
    DeviceType = {"cuda": Accelerator, "cpu": CPU}[device.type]
    return DeviceType(device.index)


def max_tensor_type(tensor: torch.Tensor) -> TensorType:
    """Returns the equivalent MAX tensor type for a PyTorch tensor."""
    dtype = torch_dtype_to_max(tensor.dtype)
    shape = max_shape(tensor.shape)
    device = max_device_ref(tensor.device)
    return TensorType(dtype, shape, device=device)
