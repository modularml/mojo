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

from __future__ import annotations

import difflib
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from inspect import signature
from typing import Any, Protocol, runtime_checkable

import numpy as np
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    Shape,
    ShapeLike,
    ShardingStrategy,
    StaticDim,
    Type,
    Value,
    Weight,
)
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from typing_extensions import Self

from .._identity import IdentitySet


@runtime_checkable
class Shardable(Protocol):
    """Protocol for objects that support sharding across multiple devices.

    This protocol defines the interface that all shardable components
    (like Linear layers and Weight objects) must implement to participate
    in distributed computation.
    """

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the weight sharding strategy."""
        ...

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the weight sharding strategy.

        Args:
            strategy: A ShardingStrategy that defines how to shard the weight.
        """
        ...

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Self]:
        """Creates a sharded view of this object for a specific device.

        Args:
            device: The devices where this shard should reside.

        Returns:
            A sequence of sharded instances of this object.
        """
        ...


@runtime_checkable
class FlattenableGraphInput(Protocol):
    """A structured graph input that can cross a subgraph boundary.

    Objects implementing this protocol (for example
    :class:`~max.nn.kv_cache.PagedCacheValues`) know how to serialize
    themselves into a flat list of graph :class:`~max.graph.Value` objects and
    reconstruct themselves from a flat iterator, letting them be passed as a
    single logical argument through :meth:`Module.build_subgraph` /
    :func:`~max.graph.ops.call` without manual field-by-field decomposition.
    """

    def flatten(self) -> list[Value[Any]]:
        """Serializes this object into a flat list of graph values."""
        ...

    def unflatten(self, it: Iterator[Any]) -> Any:
        """Reconstructs this object by consuming values from ``it``."""
        ...


# A single argument in a subgraph's input pytree: a leaf ``Value``, a
# (possibly nested) list/tuple of such arguments, or a flattenable structured
# node. ``Any`` is used for the recursive/leaf cases to keep the annotation
# usable at call sites.
SubgraphInput = Value[Any] | Sequence[Any] | FlattenableGraphInput


def _flatten_graph_inputs(node: Any) -> list[Value[Any]]:
    """Flattens a pytree of subgraph inputs into a flat list of values.

    A leaf is anything that is neither a ``list``/``tuple`` nor a
    :class:`FlattenableGraphInput`. Flattenable nodes are expanded via their
    own ``flatten`` method, keeping their field order.
    """
    if isinstance(node, list | tuple):
        result: list[Value[Any]] = []
        for item in node:
            result.extend(_flatten_graph_inputs(item))
        return result
    if isinstance(node, FlattenableGraphInput):
        return list(node.flatten())
    return [node]


def _rebuild_graph_inputs(node: Any, it: Iterator[Value[Any]]) -> Any:
    """Rebuilds ``node``'s structure, drawing fresh leaves from ``it``.

    Mirrors :func:`_flatten_graph_inputs`: it must consume values from ``it`` in
    the exact order that function produced them.
    """
    if isinstance(node, list | tuple):
        return [_rebuild_graph_inputs(item, it) for item in node]
    if isinstance(node, FlattenableGraphInput):
        return node.unflatten(it)
    return next(it)


class Layer:
    """.. deprecated:: 25.2

    Base class for neural network components.
    Use :class:`Module` instead.

    Provides functionality for adding hooks to the call function of
    each layer to support testing, debugging or profiling.
    """

    def __init_subclass__(cls):
        if cls.__name__ == "Module":
            # Module subclasses Layer, but we don't want to apply
            # _call_with_hooks to it.
            return
        # Check `__dict__` instead of `hasattr` because `hasattr` passes on
        # subclasses that don't implement the method.
        if "__call__" in cls.__dict__:
            setattr(cls, "__call__", _call_with_hooks(cls.__dict__["__call__"]))  # noqa: B010

    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        `max.graph.TensorValue`. Non-`TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


class Module(Layer, ABC):
    """Base class for model components with weight management.

    Provides functionality to create custom layers and construct networks with automatic weight tracking.

    The following example uses the :class:`Module` class to create custom layers and build a neural network:

    .. code-block:: python

        from max import nn
        from max.dtype import DType
        from max.graph import Weight, ops, DeviceRef

        class Linear(nn.Module):
            def __init__(self, in_dims, out_dims):
                super().__init__()
                self.weight = Weight("weight", DType.float32, (in_dims, out_dims), DeviceRef.CPU())

            def __call__(self, x):
                return x @ self.weight.T

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = Linear(5, 10)
                self.gate = Linear(5, 10)
                self.down = Linear(10, 5)

            def __call__(self, x):
                return self.down(ops.silu(self.gate(x)) + self.up(x))

        model = MLP()
        print(model.state_dict())  # {"up.weight": Buffer([5, 10]), ...}

    Constructing a graph without :class:`Module` can result in name collisions
    with the weights (in this example, there would be three weights with the
    name ``Weight``). With :class:`Module`, you can use :meth:`state_dict` or
    :meth:`load_state_dict` to initialize or set the weights values, and finalize
    the weight names to be unique within the model.
    """

    @property
    def _omit_module_attr_name(self) -> bool:
        """Whether this module's attribute name is dropped from descendant FQNs.

        When ``True``, this module's attribute name is skipped when building
        weight FQNs in :meth:`raw_state_dict` / :meth:`load_state_dict`, so
        children appear in the namespace as if they were attached directly
        to the parent.

        Example: by default, a child Linear's weight under
        ``self.qkv_proj`` is exposed as ``self_attn.qkv_proj.q_proj.weight``.
        If ``qkv_proj`` opts into ``_omit_module_attr_name``, the same
        weight is exposed as ``self_attn.q_proj.weight`` instead.

        Defaults to ``False``. Subclasses opt in by overriding this
        property (see :class:`~max.nn.StackedLinear`, which returns
        ``not self._stacked``).
        """
        return False

    def __init__(self) -> None:
        # `__init__` may be called if `__setattr__` is called before
        # `super().__init__()`. So, to avoid resetting the values, first
        # check to see if the layer has been initialized before.
        if not hasattr(self, "_sublayers"):
            self._sublayers: dict[str, Module] = {}
            self._layer_weights: dict[str, Weight] = {}
            self._weight_values: dict[str, DLPackArray] = {}
            self._shared_weights: dict[str, Weight] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            if isinstance(value, Module):
                self._sublayers[name] = value
            elif isinstance(value, Weight):
                if existing_weight := getattr(self, name, None):
                    if isinstance(existing_weight, Weight):
                        # If the attribute being set is a weight, remove the existing weight from
                        # the layer weights. This is to avoid having irrelevant weights returned by
                        # `raw_state_dict()`. This is particularly important for subgraphs as Weight
                        # instances are materialized by accessing their `_mlir_value` attribute, and
                        # doing so on an weight that's meant to be sharded will raise an error because
                        # we'll end up trying to add a weight that's already been added to the graph.
                        del self._layer_weights[existing_weight.name]
                self._layer_weights[value.name] = value
        except AttributeError:
            # The layer didn't call `super().__init__()` first thing.
            Module.__init__(self)
            self.__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        # TODO: Make this pretty
        return f"{type(self).__name__}({len(self.sublayers)} layers, {len(self.layer_weights)} weights)"

    @property
    def layer_weights(self) -> dict[str, Weight]:
        """Returns a mapping from weight name to :class:`~max.graph.Weight` for this layer."""
        return self._layer_weights

    def __delattr__(self, name: str) -> None:
        self._sublayers.pop(name, None)
        self._layer_weights.pop(name, None)
        self._shared_weights.pop(name, None)
        super().__delattr__(name)

    def set_shared_weight(self, name: str, weight: Weight) -> None:
        """Registers a :class:`~max.graph.Weight` as shared on this layer.

        Sets ``name`` as an attribute on this layer and marks the weight as
        shared so that :meth:`raw_state_dict` and :meth:`load_state_dict` skip
        it when iterating over owned weights.

        Args:
            name: The attribute name under which the weight is registered.
            weight: The :class:`~max.graph.Weight` to share.
        """
        setattr(self, name, weight)
        self._shared_weights[name] = weight

    def build_subgraph(
        self,
        name: str,
        inputs: Sequence[SubgraphInput],
        weight_prefix: str = "",
    ) -> Graph:
        """Builds a subgraph encapsulating this layer's computation.

        Call this method once on a representative layer, then call the returned
        subgraph once per layer using :func:`~max.graph.ops.call` with a unique
        ``prefix``. This pattern lets the compiler process the layer definition
        once rather than once per repetition, which significantly reduces compile
        time for models with many identical layers.

        Examples:

            Define a small layer, build a subgraph from a representative input,
            then call it once per repetition with a layer-specific weight
            ``prefix``:

            .. code-block:: python

                from max.dtype import DType
                from max.graph import DeviceRef, Graph, TensorType, Weight, ops
                from max.nn.layer import Module

                class Linear(Module):
                    def __init__(self, in_dims, out_dims):
                        super().__init__()
                        self.weight = Weight(
                            "weight",
                            DType.float32,
                            (in_dims, out_dims),
                            DeviceRef.CPU(),
                        )

                    def __call__(self, x):
                        return x @ self.weight.T

                num_layers = 3
                layer = Linear(4, 4)
                with Graph(
                    "build_subgraph_example",
                    input_types=(
                        TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
                    ),
                ) as graph:
                    h = graph.inputs[0].tensor

                    # Build the subgraph once from a representative input.
                    subgraph = layer.build_subgraph(
                        "linear_block",
                        inputs=[h],
                        weight_prefix="layers.0.",
                    )

                    # Call it once per layer with the correct weight prefix.
                    for idx in range(num_layers):
                        outputs = ops.call(subgraph, h, prefix=f"layers.{idx}.")
                        h = outputs[0].tensor

                    graph.output(h)

        Args:
            name: The name of the subgraph. Must be unique within the containing
                graph.
            inputs: Representative input values for the subgraph, one per
                positional argument of the layer's ``__call__``. Each argument
                may be a single :class:`~max.graph.Value`, a (possibly nested)
                list/tuple of values, or a structured
                :class:`FlattenableGraphInput` such as
                :class:`~max.nn.kv_cache.PagedCacheValues`. The subgraph's
                signature is derived from the flattened leaves' types, and the
                same structure is rebuilt from the subgraph's inputs before the
                layer is invoked.
            weight_prefix: A prefix string to strip from weight names before
                registering them as placeholder weights. At call time, the caller
                supplies the same prefix via the ``prefix`` argument of
                :func:`~max.graph.ops.call` to re-resolve each weight to the
                correct entry in the weights registry.

        Returns:
            A :class:`~max.graph.Graph` instance representing the subgraph.

        Notes:
            Weights with names that start with ``weight_prefix`` are marked as
            placeholders. Any :func:`~max.graph.ops.call` invocation for this
            subgraph must supply a matching ``prefix``. Pass the flattened
            leaves (via :func:`~max.graph.ops.call`) in the same order that
            ``inputs`` flattens to.
        """
        layer_weights = list(self.raw_state_dict().values())

        flat_leaves: list[Value[Any]] = []
        for arg in inputs:
            flat_leaves.extend(_flatten_graph_inputs(arg))
        subgraph_input_types: list[Type[Any]] = [
            leaf.type for leaf in flat_leaves
        ]

        with Graph.current.add_subgraph(
            name,
            input_types=subgraph_input_types,
            devices=list(Graph.current.device_chains.keys()),
        ) as subgraph:
            fresh_inputs = iter(subgraph.inputs)
            subgraph_inputs = [
                _rebuild_graph_inputs(arg, fresh_inputs) for arg in inputs
            ]

            if weight_prefix:
                for weight in filter(
                    lambda w: w.name.startswith(weight_prefix), layer_weights
                ):
                    weight._placeholder = True
                    weight.name = weight.name.removeprefix(weight_prefix)

            result = self(*subgraph_inputs)
            if isinstance(result, list | tuple):
                subgraph.output(*result)
            else:
                subgraph.output(result)

        return subgraph

    @property
    def sublayers(self) -> dict[str, Module]:
        return self._sublayers

    def replace_module(self, name: str, module: Module) -> None:
        """Replaces the child module registered at ``name``.

        Args:
            name: The attribute name of an existing child module.
            module: The replacement module.

        Raises:
            KeyError: If ``name`` is not a registered child module. Unlike a
                bare ``setattr``, this surfaces a wrong name instead of
                silently creating a new, unreferenced child.
        """
        if name not in self._sublayers:
            raise KeyError(
                f"{type(self).__name__!r} has no child module {name!r}"
            )
        setattr(self, name, module)

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackArray | WeightData],
        *,
        override_quantization_encoding: bool = False,
        weight_alignment: int | None = None,
        strict: bool = True,
    ) -> None:
        """Sets the values of all weights in this model.

        The keys in ``state_dict`` must match the fully-qualified weight
        names used internally by the `Module`. Those names normally
        follow the attribute hierarchy (e.g.
        ``model.layers.0.self_attn.qkv_proj.weight``), but a sublayer
        whose :attr:`_omit_module_attr_name` is ``True`` is *omitted*
        from its descendants' FQNs. The canonical example is
        :class:`~max.nn.StackedLinear` in unfused mode, where
        ``self.qkv_proj = StackedLinear(names=["q_proj", "k_proj", "v_proj"], stacked=False)``
        exposes weights at ``self_attn.q_proj.weight`` /
        ``self_attn.k_proj.weight`` / ``self_attn.v_proj.weight`` rather
        than nested under ``self_attn.qkv_proj.``. Use
        :meth:`raw_state_dict` to inspect the exact keys this method
        expects for a given module.

        Args:
            state_dict: A map from weight name to a numpy array or
                :class:`~max.driver.Buffer`.
            override_quantization_encoding: Whether to override the weight
                quantization based on the loaded value.
            weight_alignment: If specified, overrides the alignment for each
                weight in the `Module`. If left as `None`, each value in
                state_dict must be aligned by the default dtype alignment.
            strict: If True, raises an error if any weights required by the
                `Module` are missing from `state_dict`, or if any keys in
                `state_dict` were not used by the `Module`. If False, both
                missing and unexpected keys are tolerated and reported only
                via return values/logging by callers.

        Raises:
            ValueError: If `strict` is True and any required weight is missing
                from `state_dict`, or if `state_dict` contains keys not used by
                the `Module`.
        """
        loaded_keys = set()
        missing_keys = set()

        for full_weight_name, weight, _layer in self._iter_named_weights():
            if (data := state_dict.get(full_weight_name)) is not None:
                loaded_keys.add(full_weight_name)
                if isinstance(data, WeightData):
                    data = _array_from_weight_loader(
                        weight,
                        data,
                        override_quantization_encoding,
                        full_weight_name,
                    ).data
                else:
                    _validate_weight_value(weight, data, full_weight_name)

                if weight_alignment:
                    weight.align = weight_alignment

                _check_alignment(
                    data,
                    weight.align or weight.dtype.align,
                    full_weight_name,
                )
                self._weight_values[full_weight_name] = data
                weight.name = full_weight_name
            else:
                missing_keys.add(full_weight_name)

        # After the loop, check for all errors at once if in strict mode.
        unused_keys = state_dict.keys() - loaded_keys

        if strict and (missing_keys or unused_keys):
            parts = []
            if missing_keys:
                sorted_missing = sorted(missing_keys)
                parts.append(
                    f"Missing required weights: {', '.join(sorted_missing)}"
                )

                # Add a helpful "Did you mean?" suggestion for the first missing key.
                first_missing_key = sorted_missing[0]
                if possible_match := difflib.get_close_matches(
                    first_missing_key, state_dict.keys(), n=1
                ):
                    parts.append(
                        f"For '{first_missing_key}', did you mean '{possible_match[0]}'?"
                    )

            if unused_keys:
                parts.append(
                    f"Unexpected keys in state_dict: {', '.join(sorted(unused_keys))}"
                )

            raise ValueError(
                "load_state_dict() strict=True validation failed. "
                + "; ".join(parts)
            )

    def state_dict(
        self, auto_initialize: bool = True
    ) -> dict[str, DLPackArray]:
        """Returns values of all weights in the model.

        The values returned are the same as the values set in :meth:`load_state_dict`.
        If :meth:`load_state_dict` has not been called and none of the weights have
        values, then they are initialized to zero.

        Keys follow the same FQN convention as :meth:`load_state_dict`:
        attribute paths through the module tree, with any sublayer that
        sets :attr:`_omit_module_attr_name` (e.g.
        :class:`~max.nn.StackedLinear` in unfused mode) skipped in the
        prefix.

        Args:
            auto_initialize: Determines whether to initialize weights to zero if
                the weight value has not been loaded. If this is False, a
                ValueError is raised if an uninitialized weight is found.

        Returns:
            Map from weight name to the weight value (can be numpy array or
            :class:`~max.driver.Buffer`).
        """
        state_dict = {}
        for full_weight_name, weight in self.raw_state_dict().items():
            if (data := self._weight_values.get(full_weight_name)) is None:
                if not auto_initialize:
                    raise ValueError(
                        f"Weight '{full_weight_name}' was not initialized."
                    )
                # Contents of weights should be filled with zeros.
                data = self._weight_values[full_weight_name] = Buffer.zeros(
                    shape=weight.shape.static_dims, dtype=weight.dtype
                )
            state_dict[full_weight_name] = data
            weight.name = full_weight_name
        return state_dict

    def raw_state_dict(self) -> dict[str, Weight]:
        """Returns all weights objects in the model.
        Unlike :meth:`state_dict`, this returns :class:`~max.graph.Weight` objects instead of
        the assigned values. Some parameters inside the :class:`~max.graph.Weight` can be
        configured before a graph is built. Do not change these attributes after
        building a graph:

        - :obj:`~max.graph.Weight.align`
        - :obj:`~max.graph.Weight.dtype`
        - :obj:`~max.graph.Weight.quantization_encoding`
        - :obj:`~max.graph.Weight.shape`

        Keys follow the same FQN convention as :meth:`load_state_dict`:
        attribute paths through the module tree, with any sublayer that
        sets :attr:`_omit_module_attr_name` skipped in the prefix.

        Returns:
            Map from weight name to the :class:`~max.graph.Weight` object.
        """
        return {
            name: weight for name, weight, _layer in self._iter_named_weights()
        }

    def _iter_named_weights(
        self,
    ) -> Iterable[tuple[str, Weight, Module]]:
        """Yields ``(fully-qualified name, weight, owning layer)`` triples.

        This is the single chokepoint that turns the module tree into a
        flat namespace of weights. It is consumed by :meth:`raw_state_dict`,
        :meth:`state_dict`, and :meth:`load_state_dict` so they all agree
        on naming and on collision detection.

        Raises:
            ValueError: If two distinct attribute paths produce the same
                fully-qualified weight name. This can happen when a
                module with ``_omit_module_attr_name`` (e.g.
                :class:`~max.nn.StackedLinear`) flattens a child name into
                the parent's namespace and a sibling already claims that
                name.
        """
        # We carry the weight prefix explicitly through the walk rather
        # than re-deriving it from each child's display name; otherwise a
        # name-omitting child's dropped attribute name would be silently
        # reintroduced when descending past it.
        seen: dict[str, str] = {}
        # Queue entries are ``(display_name, weight_prefix, layer)``.
        # ``weight_prefix`` already accounts for any name-omitting ancestors
        # above ``layer``; ``display_name`` is only used in error messages.
        queue: deque[tuple[str, str, Module]] = deque()
        queue.append(("", "", self))
        visited = IdentitySet[Module]()

        while queue:
            display_name, weight_prefix, layer = queue.popleft()
            if layer in visited:
                continue
            visited.add(layer)

            for weight_name, weight in layer.layer_weights.items():
                if weight_name in layer._shared_weights:
                    continue
                full_name = f"{weight_prefix}{weight_name}"
                if (existing_owner := seen.get(full_name)) is not None:
                    raise ValueError(
                        f"Duplicate weight FQN '{full_name}' produced by "
                        f"two distinct attribute paths: '{existing_owner}' "
                        f"and '{display_name or '<root>'}'. Two weights "
                        "cannot share the same fully-qualified name in a "
                        "module's state_dict.\n\n"
                        "This most often happens because one of the "
                        "owning modules sets `_omit_module_attr_name = "
                        "True` (notably `StackedLinear` in unfused mode, "
                        "i.e. `stacked=False`), which drops the module's "
                        "own attribute name from its descendants' FQNs. "
                        "When such a module flattens a child name "
                        "(e.g. `qkv_proj.q_proj` -> `q_proj`) into a "
                        "namespace where a sibling already uses that "
                        "name, the two collide here.\n\n"
                        "To fix: rename one of the colliding attributes "
                        "(e.g. pass different `names=[...]` to the "
                        "name-omitting module, or rename the conflicting "
                        "sibling), or set `_omit_module_attr_name = "
                        "False` on the offending module if its attribute "
                        "name should be preserved in the FQN."
                    )
                seen[full_name] = display_name or "<root>"
                yield full_name, weight, layer

            # Enqueue children. A child with ``_omit_module_attr_name``
            # contributes nothing to the weight prefix; otherwise the
            # child appends its own attribute name.
            for local_name, child in layer.sublayers.items():
                if child._omit_module_attr_name:
                    child_weight_prefix = weight_prefix
                else:
                    child_weight_prefix = f"{weight_prefix}{local_name}."
                # display_name still includes the attribute name even for
                # a name-omitting child, so error messages can point at
                # the actual attribute path used in the source.
                if display_name:
                    child_display = f"{display_name}.{local_name}"
                else:
                    child_display = local_name
                queue.append((child_display, child_weight_prefix, child))

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Defines the forward function of this layer.

        Subclasses must override this function. There is no exact signature that a
        call function must follow, but inputs/outputs should generally be
        :class:`~max.graph.TensorValue`. Non-:class:`~max.graph.TensorValue` inputs are fine, but
        cannot be updated once the graph is built.
        """


def _array_from_weight_loader(
    weight: Weight,
    data: WeightData,
    override_quantization_encoding: bool,
    name: str,
) -> WeightData:
    """Processes and validates the data from WeightData."""
    if weight.quantization_encoding == QuantizationEncoding.GPTQ:
        # Load all weights with GPTQ quantization as uint8.
        # Store the original shape and dtype of the weight (used in layers like
        # GPTLinear).
        weight.original_dtype_and_shape = (data.dtype, data.shape)
        data.data = new_data = Buffer.from_dlpack(data.data).view(DType.uint8)
        data.dtype = DType.uint8
        data.shape = Shape(new_data.shape)
        weight._shape = Shape(new_data.shape)

    if weight.quantization_encoding:
        # TODO: Set the quantized weight shape correctly when initializing the
        # weight. For now, we trust that the value loaded from the checkpoint
        # has the correct shape.
        weight._shape = data.shape
    elif (weight.shape == [] and data.shape == [1]) or (
        weight.shape == [1] and data.shape == []
    ):
        # These shapes are actually the same.
        # Treat the data as if it has the correct shape.
        data.shape = Shape(weight._shape)
    elif weight.shape != data.shape:
        if weight.dtype == DType.uint8 and data.dtype == DType.uint8:
            weight._shape = data.shape
            data.shape = Shape(weight._shape)
        else:
            raise ValueError(
                f"Value provided to weight '{name}' had different shape"
                f" (expected={weight.shape}, actual={data.shape})"
            )

    if weight.quantization_encoding != data.quantization_encoding:
        if (
            override_quantization_encoding
            and data.quantization_encoding is not None
        ):
            weight.quantization_encoding = data.quantization_encoding
        # We don't raise an error if `override_quantization_encoding` is `False`
        # because in some cases the data is not aware of its own quantization
        # type (e.g. data loaded from GPTQ Safetensors do not have a
        # quantization label)

    if weight.dtype != data.dtype:
        raise ValueError(
            f"Value provided to weight '{name}' had different dtype"
            f" (expected={weight.dtype}, actual={data.dtype})"
        )

    return data


def _get_value_shape_dtype(value: DLPackArray) -> tuple[ShapeLike, DType]:
    if isinstance(value, Buffer):
        shape = value.shape
        dtype = value.dtype
    elif isinstance(value, np.ndarray):
        shape = value.shape
        dtype = DType.from_numpy(value.dtype)
    else:
        # `from_dlpack` does not copy the data.
        value_buffer = Buffer.from_dlpack(value)
        shape = value_buffer.shape
        dtype = value_buffer.dtype

    return shape, dtype


def _check_alignment(value: DLPackArray, align: int, name: str) -> None:
    # Fast path for ndarray.
    # The use of Buffer.from_dlpack always copies if the numpy array is not
    # writeable, which is very common for weight values.
    #
    # This logic special cases the two code paths that potentially could copy
    # and performs the alignment check manually.
    if isinstance(value, np.ndarray):
        data = value.ctypes.data
        if data % align == 0:
            return
    elif isinstance(value, Buffer):
        if value._aligned(align):
            return
    else:
        buffer = Buffer.from_dlpack(value)
        if buffer._aligned(align):
            return

    raise ValueError(
        f"Found unaligned weight '{name}' (expected alignment={align})."
        "If you are using a Safetensor checkpoint, it is recommended that "
        "you copy the weight to correct the alignment, or pass "
        "`weight_alignment=1` to `Module.load_state_dict()`."
    )


def _validate_weight_value(
    weight: Weight, value: DLPackArray, name: str
) -> None:
    if not isinstance(value, DLPackArray):
        raise ValueError(
            f"The class type of '{name}' value ({type(value)}) is not an array "
            "type that we understand. Please use a numpy array or max.driver.Buffer."
        )

    shape, dtype = _get_value_shape_dtype(value)

    diffs = []

    # Check if weight has symbolic dimensions
    # Convert weight.shape to list to ensure it's sized
    weight_shape_dims = list(weight.shape)
    weight_has_symbolic_dims = len(weight_shape_dims) != len(
        weight.shape.static_dims
    )

    # Convert shape to tuple to ensure it's sized
    shape_tuple = tuple(shape)

    if weight_has_symbolic_dims:
        # For weights with symbolic dimensions, validate by comparing static dimensions
        # at their correct positions, allowing symbolic dimensions to vary
        if len(shape_tuple) != len(weight_shape_dims):
            # Shape rank must match
            diffs.append(
                f"shape rank (expected={len(weight_shape_dims)}, actual={len(shape_tuple)})"
            )
        else:
            # Check each dimension: static dims must match, symbolic dims can vary
            mismatches = []
            for i, (weight_dim, value_dim) in enumerate(
                zip(weight_shape_dims, shape_tuple, strict=True)
            ):
                if isinstance(weight_dim, StaticDim):
                    # This is a static dimension - must match exactly
                    if int(weight_dim) != value_dim:
                        mismatches.append(
                            f"dim[{i}]: expected {int(weight_dim)}, got {value_dim}"
                        )
                # Symbolic dimensions are allowed to vary, so no check needed

            if mismatches:
                diffs.append(f"shape ({', '.join(mismatches)})")
    else:
        # For fully static weights, use the original validation
        weight_shape = tuple(weight.shape.static_dims)
        if shape_tuple != weight_shape:
            diffs.append(
                f"shape (expected={weight_shape}, actual={shape_tuple})"
            )

    if dtype != weight.dtype:
        diffs.append(f"dtype (expected={weight.dtype}, actual={dtype})")
    if diffs:
        diff_str = " and ".join(diffs)
        raise ValueError(
            f"Value provided to weight '{name}' had different {diff_str}."
        )


def recursive_named_layers(
    parent: Module, prefix: str = ""
) -> Iterable[tuple[str, Module]]:
    """Recursively walks through the layers and generates names.

    A sublayer marked with ``_omit_module_attr_name = True`` does not
    contribute its attribute name to the prefix passed to its descendants;
    the children appear in the namespace as if they were directly attached
    to the name-omitting module's parent. The name-omitting layer itself
    is still yielded (callers walking the module tree may need to reach
    it via its full attribute path), but its descendants' names skip the
    omitted segment.

    .. note::
       The names yielded here are *layer* names, not weight names. For a
       name-omitting layer they include the omitted segment (so callers
       can still reach the layer in the source tree), which means they
       do **not** match the FQNs that appear in :meth:`Module.state_dict`
       for that layer's descendants. If you need state-dict-aligned
       names, walk the weights via :meth:`Module.raw_state_dict` /
       ``_iter_named_weights`` instead.

    The ``parent`` passed to this function must not itself set
    ``_omit_module_attr_name``: such a root has no enclosing namespace to
    flatten into, so the request is ambiguous. Pass the enclosing owner
    instead.
    """
    # See the docstring above: a name-omitting root is not meaningful
    # here. Catching it as an assertion (rather than silently producing
    # names under ``prefix``) makes the misuse loud at the call site.
    assert not parent._omit_module_attr_name, (
        "recursive_named_layers() called with a "
        "`_omit_module_attr_name` root module; pass the enclosing parent "
        "instead."
    )

    seen = IdentitySet[Module]()
    # Queue entries are ``(name, prefix_for_children, layer)``:
    # - ``name`` is the FQN this layer is yielded under (the full attribute
    #   path, including any omitted segments).
    # - ``prefix_for_children`` is the FQN prefix used to *build* the
    #   children's ``name``. For a name-omitting layer this skips its own
    #   attribute name; for a regular layer it is ``f"{name}."``.
    queue: deque[tuple[str, str, Module]] = deque()
    root_child_prefix = f"{prefix}." if prefix else ""
    queue.append((prefix, root_child_prefix, parent))

    while queue:
        name, child_prefix, layer = queue.popleft()
        if layer in seen:
            continue
        seen.add(layer)

        yield (name, layer)

        for local_name, child in layer.sublayers.items():
            child_name = f"{child_prefix}{local_name}"
            if child._omit_module_attr_name:
                # The child's own attribute name is dropped from its
                # descendants' prefixes.
                grandchild_prefix = child_prefix
            else:
                grandchild_prefix = f"{child_prefix}{local_name}."
            # ``child_name`` keeps the full attribute path even for
            # name-omitting children so callers can reach them; only the
            # *grandchildren* see the omission.
            queue.append((child_name, grandchild_prefix, child))


_LOCAL = threading.local()
_LAYER_HOOKS = _LOCAL._layer_hooks = []


def add_layer_hook(
    fn: Callable[[Layer, tuple[Any, ...], dict[str, Any], Any], Any],
) -> None:
    """Adds a hook to call a function after each layer's ``__call__``.

    The function will be passed four inputs:
    - layer
    - input_args
    - input_kwargs
    - outputs

    The function can either return `None` or new
    outputs that will replace the layer returned outputs.

    Note that input and outputs contain graph Values, which show limited
    information (like :obj:`~max.graph.TensorValue.shape` and :obj:`~max.graph.TensorValue.dtype`). You can still see the computed values
    if you include the Value in the :obj:`graph.ops.output` op, or call :obj:`graph.ops.print`.

    Example of printing debug inputs:

    .. code-block:: python

        from max.nn.layer import add_layer_hook

        def print_info(layer, args, kwargs, outputs):
            print("Layer:", type(layer).__name__)
            print("Input args:", args)
            print("Input kwargs:", kwargs)
            print("Outputs:", outputs)
            return outputs

        add_layer_hook(print_info)
    """
    _LAYER_HOOKS.append(fn)


def clear_hooks() -> None:
    """Remove all hooks."""
    _LAYER_HOOKS.clear()


def _call_with_hooks(call_fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(call_fn)
    def __call_with_hooks(layer: Layer, *args, **kwargs) -> Any:
        # Hide this wrapper from rich traceback.
        _rich_traceback_omit = True

        outputs = call_fn(layer, *args, **kwargs)
        # Use the inspect lib to ensure that args and kwargs are passed
        # to the hook as defined in the function signature.
        bound_args = signature(call_fn).bind(layer, *args, **kwargs)
        for hook in _LAYER_HOOKS:
            # Call the hook. Note that the first argument in `bound_args.args`
            # is the layer, so it is skipped.
            hook_outputs = hook(
                layer, bound_args.args[1:], bound_args.kwargs, outputs
            )
            if hook_outputs is not None:
                outputs = hook_outputs
        return outputs

    return __call_with_hooks
