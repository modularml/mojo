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

"""Graph-compiler cast model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-1 cast graph.
- **Precompile sweep (``=1``).** The batched sweep compiles the full matrix at
  import; a :func:`cast_model` miss is then a hard error.

Cast is any->any dtype, so its cache key is ``(device, src_dtype, dst_dtype)``,
an N x N matrix, unlike the single dtype the other elementwise families key on.
``src`` and ``dst`` are both drawn from :func:`_cast_dtypes`. ``ops.cast``
elides a same-dtype cast (it emits no ``mo.CastOp``), so the matrix excludes
the identity ``src == dst`` and is ``|dtypes| x (|dtypes| - 1)`` per device.
Models serve the eager handler via :func:`cast_model`. Must not import from
``handlers.py``.
"""

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops


def _cast_dtypes(device: Device) -> list[DType]:
    """The dtypes cast sweeps on *device* (both endpoints drawn from this set).

    - Floats: the shared conservative set (:func:`gc_compile.float_dtypes`),
      f32/f64 on CPU and f16/f32/bf16 on GPU.
    - Every integer dtype and bool, on both.
    """
    return (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
        + [DType.bool]
    )


def _graph_name(device: Device, src: DType, dst: DType) -> str:
    return f"cast_{device.label}_{device.id}_{src.name}_{dst.name}"


canonical_shape = gc_compile.canonical_shape_rank1


def _cast_graph(module: Module, device: Device, src: DType, dst: DType) -> None:
    """Adds one fully-symbolic rank-1 cast graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(src, ["n"], device=device_ref)
    graph = Graph(
        _graph_name(device, src, dst),
        input_types=[in_type],
        module=module,
    )
    with graph:
        (x,) = graph.inputs
        graph.output(ops.cast(x.tensor, dst))


def _is_supported(device: Device, src: DType, dst: DType) -> bool:
    """Whether (device, src, dst) is in the swept cast matrix.

    Single source of truth for the matrix (the sweep and :func:`cast_model`'s
    guard both route through it): both endpoints in the device's set, and
    ``src != dst``, since ``ops.cast`` elides a same-dtype cast, so no identity
    ``mo.CastOp`` ever reaches the interpreter.
    """
    dtypes = _cast_dtypes(device)
    return src != dst and src in dtypes and dst in dtypes


class _CastFamily(gc_compile.GCFamilySpec):
    name = "cast"

    def build_module(self) -> Module:
        """Builds every (device, src, dst) graph across CPU + all accelerators
        into one module, shared by the warm producer and the batched sweep."""
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Builds every (src, dst) graph for one device slot. Per-slot so the
        warm is device-count-independent: a k-GPU consumer adopts slots
        ``0..k-1``.
        """
        if module is None:
            module = Module()
        dtypes = _cast_dtypes(device)
        for src in dtypes:
            for dst in dtypes:
                if _is_supported(device, src, dst):
                    _cast_graph(module, device, src, dst)
        return module


_FAMILY = gc_compile.GCOpFamily(_CastFamily())
gc_compile.register_family(_FAMILY)


def cast_model(device: Device, src: DType, dst: DType) -> engine.Model:
    """Returns the cast :class:`~max.engine.Model` for *device* / *src* / *dst*.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss an available warm cache is adopted whole rather
    than compiling each target singly.

    Args:
        device: The realized input's device (cast does not move devices).
        src: The input dtype.
        dst: The target dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If (device, src, dst) is outside the swept matrix (e.g. a
            16-bit-float endpoint on CPU); or, with ``MAX_EAGER_OP_PRECOMPILE=1``,
            if a supported target was not swept.
    """
    key = _graph_name(device, src, dst)
    # Cache-check before building the closures below: this runs on every eager
    # cast dispatch, so a hit must not pay for closures it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(device, src, dst):
            return None
        return (
            f"Unsupported cast device/src/dst for key {key!r}."
            f"  Supported dtypes for this device: {_cast_dtypes(device)}"
        )

    def build(module: Module) -> None:
        _cast_graph(module, device, src, dst)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
