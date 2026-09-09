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

"""Modular framework Python bindings."""

import enum
import inspect
import os
import pathlib
import types
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeAlias, overload

import max._core.driver
import max._core.dtype
import max._core.mlrt
from max._core.driver import Buffer
from max._core.mlrt import AsyncValue
from max._core_types.driver import DLPackArray

InputType: TypeAlias = DLPackArray | Buffer | int | float | bool

class TensorSpec:
    """
    Defines the properties of a tensor, including its name, shape and data type.

    For usage examples, see :obj:`Model.input_metadata`.
    """

    @property
    def dtype(self) -> max._core.dtype.DType:
        """A tensor data type."""

    @property
    def name(self) -> str:
        """A tensor name."""

    @property
    def shape(self) -> list[int | None] | None:
        """
        The shape of the tensor as a list of integers.

        If a dimension size is unknown/dynamic (such as the batch size), its
        value is ``None``.
        """

class ModelMetadata:
    """Input and output metadata for a compiled model function."""

    @property
    def name(self) -> str: ...
    @property
    def input_metadata(self) -> list[TensorSpec]: ...
    @property
    def output_metadata(self) -> list[TensorSpec]: ...

class CompiledModels:
    """A compiled model artifact containing one or more submodels."""

    def __len__(self) -> int: ...
    def __getitem__(self, arg: int, /) -> ModelMetadata: ...
    def __iter__(self) -> Iterator[ModelMetadata]: ...
    @property
    def names(self) -> list[str]: ...
    def export_mef(self, path: str) -> None:
        """
        Exports the compiled model as MEF bytes to the given file.

        Args:
            path: Filesystem path to write the MEF to.
        """

def get_config_value(key: str) -> str:
    """
    Reads a config value.

    Global overrides take priority, then the ``MODULAR_<KEY>``
    environment variable, then ``modular.cfg``.

    Raises:
        KeyError: If the key is not set in any source.
        RuntimeError: If the config fails to open.
    """

def get_global_value(key: str) -> str | None:
    """
    Returns the process-wide override for ``key``, or ``None`` if unset.

    Ignores environment variables and ``modular.cfg``.
    """

def set_global_value(key: str, value: str) -> None:
    """
    Sets a process-wide config override.

    Overrides take priority over environment variables and
    ``modular.cfg`` for every consumer in the process. They are not
    inherited by subprocesses.
    """

def unset_global_value(key: str) -> None:
    """Removes an override set by :func:`set_global_value`."""

def max_cache_dir() -> pathlib.Path | None:
    """
    Returns the directory the engine caches compiled models (``.mef``) in.

    Resolved by the compiler itself.

    Returns:
        pathlib.Path | None: the cache directory, or None if unresolvable.
    """

@overload
def read(path: str | os.PathLike) -> CompiledModels:
    """
    Reads a compiled-model artifact (``.mef``) from a file path.

    Returns:
        CompiledModels: the artifact, ready to be initialized on any
        session via :meth:`InferenceSession._load_all`.

    Raises:
        RuntimeError: if the file is missing or is not a valid MEF
        for this engine build.
    """

@overload
def read(data: bytes) -> CompiledModels:
    """
    Reads a compiled-model artifact (``.mef``) from bytes.

    Returns:
        CompiledModels: the artifact, ready to be initialized on any
        session via :meth:`InferenceSession._load_all`.

    Raises:
        RuntimeError: if the bytes are not a valid MEF for this
        engine build.
    """

class Model:
    """
    A loaded model that you can execute.

    Do not instantiate this class directly. Instead, create it with
    :meth:`InferenceSession.load` or :meth:`InferenceSession.init`.

    A :class:`Model` is callable. Calling it directly (``model(inputs...)``)
    accepts tensors as positional or keyword arguments and dispatches
    to :meth:`execute`. You can also call :meth:`execute` directly, which
    accepts positional arguments only.

    When using keyword arguments, the names must match the model's input
    metadata (see :attr:`input_metadata`). Calling the model raises
    ``TypeError`` if a keyword argument doesn't match a model input, if a
    positional and keyword argument refer to the same parameter, or if the
    number of inputs doesn't match.

    For supported input types and execution errors, see :meth:`execute`.
    """

    @property
    def devices(self) -> list[max._core.driver.Device]:
        """Returns the device objects used in the Model."""

    @property
    def input_devices(self) -> list[max._core.driver.Device]:
        """
        Devices of the model's input tensors, as a list of :obj:`Device` objects.
        """

    @property
    def input_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's input tensors, as a list of :obj:`TensorSpec` objects.

        For example, you can print the input tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.input_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """

    @property
    def output_devices(self) -> list[max._core.driver.Device]:
        """
        Devices of the model's output tensors, as a list of :obj:`Device` objects.
        """

    @property
    def output_metadata(self) -> list[TensorSpec]:
        """
        Metadata about the model's output tensors, as a list of :obj:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            for tensor in model.output_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
        """

    @property
    def kernel_summaries(self) -> list[str]:
        """
        Kernel fusion summaries from the compiled model.

        Returns a list of strings, one per ``mgp.generic.execute`` kernel in
        the compiled graph.  Each string describes the fused kernel composition,
        e.g. ``"Epilogue(custom__kv_rope, custom__kv_cache_store)"``.
        """

    @property
    def name(self) -> str:
        """
        The symbol name of this model.

        Mirrors the ``sym_name`` of the model's ``mo.graph`` op, preserved
        through MEF serialization. Used by
        :meth:`InferenceSession.load_all` to key the returned dict by graph
        name.
        """

    @property
    def signature(self) -> inspect.Signature:
        """Get input signature for model."""

    def execute(self, *args: InputType) -> list[Buffer]:
        """
        Executes the model with the provided input and returns the outputs.

        For example, if the model has one input tensor:

        .. code-block:: python

            input_tensor = np.random.rand(1, 224, 224, 3)
            model.execute(input_tensor)

        Args:
            args: A list of input tensors. The following input types are
              supported:

              * Any tensors implementing the DLPack protocol, such as
                :obj:`np.ndarray` or :obj:`torch.Tensor`.
              * Max Driver buffers, such as :obj:`max.driver.Buffer`.
              * Scalar inputs, such as :obj:`bool`, :obj:`float`, :obj:`int`,
                or :obj:`np.generic`.

              All inputs are copied to the device that the model is resident on
              prior to executing.

        Returns:
            A list of output tensors. The output tensors are resident on the
            execution device.

        Raises:
            RuntimeError: If the given input tensors' shapes don't match what
              the model expects.

            TypeError: If the given input tensors' dtype can't be cast to what
              the model expects.

            ValueError: If positional inputs aren't one of the supported
              types.
        """

    def __call__(self, *args: InputType, **kwargs: InputType) -> list[Buffer]:
        """Executes the model. See :class:`Model` for details."""

    def capture(
        self, graph_keys: int | Sequence[int], *inputs: Buffer
    ) -> list[Buffer]:
        """
        Capture execution into a device graph for caller-provided key.

        Capture is best-effort and model-dependent. It records the current execution
        path; models that perform unsupported operations during capture (for example,
        host-device synchronization) will fail to capture. Callers should decide which
        phases are safe to capture (e.g. decode-only in serving).
        """

    def replay(self, graph_keys: int | Sequence[int], *inputs: Buffer) -> None:
        """Replay the captured device graph for the provided key."""

    def debug_verify_replay(
        self, graph_keys: int | Sequence[int], *inputs: Buffer
    ) -> None:
        """
        Execute eagerly and verify the launch trace matches the captured graph.

        This method validates that graph capture correctly represents eager execution
        by running the model and comparing kernel launch traces. Useful for debugging
        graph capture issues.

        Args:
            graph_keys: One graph key per participating device stream.
            inputs: Input buffers matching the captured input signature.

        Raises:
            RuntimeError: If no graph captured or trace verification fails.
        """

    def release_captured_graph(self, graph_keys: int | Sequence[int]) -> None:
        """
        Release a previously captured device graph.

        Drops the device-side graph and its working memory once the last reference
        held by the runtime is released. Releasing a key that was never captured
        is a no-op.

        Args:
            graph_keys: Caller-provided graph key (or per-device keys) identifying
                the captured graph to release.
        """

    def _execute_device_tensors(
        self, tensors: Sequence[max._core.driver.Buffer]
    ) -> list[max._core.driver.Buffer]: ...
    def _capture(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> list[max._core.driver.Buffer]:
        """Capture execution into a device graph."""

    def _replay(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> None:
        """Replay the captured device graph."""

    def _debug_verify_replay(
        self,
        graph_keys: Sequence[int],
        inputs: Sequence[max._core.driver.Buffer],
    ) -> None:
        """Debug verify replay against captured graph."""

    def _await_device_graphs(self) -> None:
        """Await all pending device graph instantiations."""

    def _release_captured_graph(self, graph_keys: Sequence[int]) -> None:
        """Release captured device graphs for the given keys."""

    def _export_mef(self, path: str) -> None:
        """
        Exports the compiled model as a mef to a file.

        Args:
          path: The filename where the mef is exported to.
        """

    def reload(self, weights_registry: Mapping[str, Any]) -> None: ...
    def release_weights(self) -> None:
        """
        Drops the host-side weight references held by this model.

        Releases the weights registry and the owning references, so the host
        weight memory can be freed once the caller drops its own references.
        Safe only when every weight was copied to its execution device during
        model init: reading a host weight after this call is undefined
        behavior. ``reload`` remains usable afterwards.
        """

class DebugConfig:
    """
    Unified debug configuration for MAX inference.

    ``DebugConfig`` is a process-wide singleton accessed through
    :attr:`InferenceSession.debug`. It controls model debugging features
    such as ``NaN`` checks, synchronous GPU execution, stack traces, and IR
    dumping.

    There are two ways to configure debugging options:

    * Set the ``MODULAR_DEBUG`` environment variable to a list of
      kebab-case property names separated by commas. Boolean properties
      can be enabled with just the name; others use ``name=value`` form.
      For example: ``MODULAR_DEBUG=nan-check,assert-level=all``.
    * Set properties directly with the Python API, for example
      ``InferenceSession.debug.<property> = <value>``. Options are
      class-level on :class:`InferenceSession` because they affect
      globally shared infrastructure.

    For the environment variable and config file, the name ``sensible``
    enables a curated default set defined in :attr:`sensible_mode`.
    """

    @property
    def nan_check(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to insert runtime checks after each compiled op that abort if any output contains ``NaN``. Takes effect at model build time.
        """

    @nan_check.setter
    def nan_check(self, arg: bool, /) -> None: ...
    @property
    def uninitialized_read_check(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to instrument buffer reads to detect reads of uninitialized memory. Takes effect at model build time.
        """

    @uninitialized_read_check.setter
    def uninitialized_read_check(self, arg: bool, /) -> None: ...
    @property
    def device_sync_mode(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to force synchronous GPU execution so every device operation waits for completion. This surfaces async errors at their call site but serializes the pipeline. Takes effect at run time.
        """

    @device_sync_mode.setter
    def device_sync_mode(self, arg: bool, /) -> None: ...
    @property
    def stack_trace_on_error(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to print a C++ stack trace whenever a runtime error is raised. Takes effect at run time.
        """

    @stack_trace_on_error.setter
    def stack_trace_on_error(self, arg: bool, /) -> None: ...
    @property
    def stack_trace_on_crash(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to print a C++ stack trace on fatal signals such as ``SIGSEGV`` or ``SIGABRT``. Takes effect at run time.
        """

    @stack_trace_on_crash.setter
    def stack_trace_on_crash(self, arg: bool, /) -> None: ...
    @property
    def source_tracebacks(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to capture Python source locations during graph construction so runtime errors can be traced back to user code. Takes effect at graph build time and is typically set using ``Graph.debug.source_tracebacks``.
        """

    @source_tracebacks.setter
    def source_tracebacks(self, arg: bool, /) -> None: ...
    @property
    def op_log_level(self) -> str:
        r"""
        A string that sets the log level for per-op tracing. One of ``\'\'``, ``'notset'``, ``'trace'``, ``'debug'``, ``'info'``, ``'warning'``, ``'error'``, ``'critical'``. Takes effect at model build time.
        """

    @op_log_level.setter
    def op_log_level(self, arg: str, /) -> None: ...
    @property
    def assert_level(self) -> str:
        r"""
        A string that sets the Mojo assertion level for compiled kernels. One of ``\'\'``, ``'none'``, ``'warn'``, ``'safe'``, ``'all'``. Higher levels enable more runtime checks (e.g. LayoutTensor bounds) at a performance cost. Takes effect at model build time.
        """

    @assert_level.setter
    def assert_level(self, arg: str, /) -> None: ...
    @property
    def print_style(self) -> PrintStyle:
        """
        A :obj:`PrintStyle` value that sets the format for tensor debug printing. Takes effect at run time.
        """

    @print_style.setter
    def print_style(self, arg: PrintStyle, /) -> None: ...
    @property
    def ir_output_dir(self) -> str:
        """
        A string path to the directory into which MAX dumps intermediate compiler IR for inspection. Empty string disables dumping. Takes effect at model build time.
        """

    @ir_output_dir.setter
    def ir_output_dir(self, arg: str, /) -> None: ...
    @property
    def sensible_mode(self) -> bool:
        """
        A boolean that, when ``True``, triggers MAX to enable a curated default debugging set, including ``nan_check``, ``assert_level='all'``, ``device_sync_mode``, ``stack_trace_on_error``, ``stack_trace_on_crash``, and ``source_tracebacks``. You can override the defaults using individual properties.
        """

    @sensible_mode.setter
    def sensible_mode(self, arg: bool, /) -> None: ...
    def reset(self) -> None:
        """Reset all debug options to their defaults."""

class InferenceSession:
    """
    Manages compilation and execution of MAX models.

    An inference session holds device configuration and compiles graphs
    into executable :class:`Model` objects. It also manages custom
    extensions and debug options.

    .. code-block:: python

        from max._core.engine import InferenceSession
        from max import driver

        devices = [driver.CPU()]
        session = InferenceSession(devices, custom_extensions=[])
        model = session.compile_from_path("model.mef", [])
    """

    def __init__(
        self,
        devices: Sequence[max._core.driver.Device],
        custom_extensions: Sequence[str | os.PathLike],
        num_threads: int = 0,
    ) -> None:
        """
        Creates an inference session for model compilation and execution.

        Args:
            devices: List of devices used for compilation and execution.
            custom_extensions: Paths to custom Mojo extension libraries.
            num_threads (int): Number of execution threads. Defaults to 0,
                which lets the runtime choose automatically.
        """

    def _load_all(
        self,
        compiled: AsyncValue[CompiledModels],
        weights_registry: Mapping[str, Any],
    ) -> list[Model]: ...
    @overload
    def compile(
        self,
        model_path: str | os.PathLike,
        custom_extension_paths: Sequence[str | os.PathLike],
    ) -> max._core.mlrt.AsyncValue[CompiledModels]:
        """
        Compiles a model from a file path.

        Args:
            model_path: Path to the compiled model file (for example, a ``.mef`` file).
            custom_extension_paths: Paths to custom Mojo extension libraries.

        Returns:
            AsyncValue[CompiledModels]: handle to the compiled artifact,
            ready to be initialized with weights via :meth:`_load_all`.
        """

    @overload
    def compile(
        self,
        model: types.CapsuleType,
        custom_extensions: Sequence[str | os.PathLike],
        pipeline_name: str,
        tile_based_fusion: bool = False,
    ) -> max._core.mlrt.AsyncValue[CompiledModels]:
        """
        Compiles a model from an in-memory capsule object.

        Args:
            model: A capsule containing the compiled model object.
            custom_extensions: Paths to custom Mojo extension libraries.
            pipeline_name: Name identifier for the compiled pipeline.
            tile_based_fusion: When ``True``, compile the graph under the
                tile-based programming model. Defaults to ``False``.

        Returns:
            CompiledModels: The compiled artifact, ready to be initialized
            with weights via :meth:`_load_all`.
        """

    def _wrap_compiled(
        self, models: CompiledModels
    ) -> max._core.mlrt.AsyncValue[CompiledModels]:
        """
        Wraps an already-read ``CompiledModels`` in a resolved async handle.

        Consumes ``models``. The handle is allocated on this session's
        runtime and can be passed to :meth:`_load_all`.
        """

    def set_debug_print_options(
        self, style: PrintStyle, precision: int, directory: str
    ) -> None:
        """
        Sets debug output options for tensor printing during execution.

        Args:
            style (PrintStyle): The output format style.
            precision (int): Number of decimal places for floating-point values.
            directory (str): Directory path for binary output files.
        """

    @overload
    def set_mojo_define(self, key: str, value: bool) -> None:
        """
        Sets a compile-time Mojo define to a boolean value.

        Args:
            key (str): The define name.
            value (bool): The boolean value to assign.
        """

    @overload
    def set_mojo_define(self, key: str, value: int) -> None:
        """
        Sets a compile-time Mojo define to an integer value.

        Args:
            key (str): The define name.
            value (int): The integer value to assign.
        """

    @overload
    def set_mojo_define(self, key: str, value: str) -> None:
        """
        Sets a compile-time Mojo define to a string value.

        Args:
            key (str): The define name.
            value (str): The string value to assign.
        """

    @property
    def devices(self) -> list[max._core.driver.Device]:
        """Returns the list of devices used by this session."""

    debug: DebugConfig

class PrintStyle(enum.Enum):
    """
    Controls the output format for debug tensor printing.

    Pass one of these values to :meth:`InferenceSession.set_debug_print_options`
    to configure how tensors are printed during execution.
    """

    COMPACT = 0
    """Compact human-readable format."""

    FULL = 1
    """Full verbose format with all tensor details."""

    BINARY = 2
    """Raw binary format."""

    BINARY_MAX_CHECKPOINT = 4
    """Binary checkpoint format compatible with MAX."""

    NONE = 3
    """Disables debug output."""
