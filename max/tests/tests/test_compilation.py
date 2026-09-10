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
"""Tests for max.experimental.compilation.

Every transform takes the callable and is then called with the callable's own
arguments, so each test spells the call once for the transform and once for the
run, and they must agree.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import compilation
from max.experimental import functional as F
from max.experimental.compilation import (
    _SPEC_TYPES,
    StagedGraph,
    as_layout,
    as_subgraph,
    compile,
    stage,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor
from max.experimental.tree_utils import paths as tree_paths
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import MHAKVCacheParams

_F32 = DType.float32


def _spec(*shape: int) -> TensorLayout:
    mesh = DeviceMesh.single(CPU())
    return TensorLayout(
        _F32, list(shape), PlacementMapping(mesh, (Replicated(),))
    )


def _dyn(*shape: int | str) -> TensorType:
    return TensorType(_F32, list(shape), device=DeviceRef.CPU())


def _mesh2() -> DeviceMesh:
    return DeviceMesh(
        devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("x",)
    )


def _sharded_spec(*shape: int) -> TensorLayout:
    return TensorLayout(
        _F32, list(shape), PlacementMapping(_mesh2(), (Sharded(0),))
    )


def _tensor(*values: float) -> Tensor:
    return Tensor.from_dlpack(np.array(values, dtype=np.float32))


def _sharded_tensor(*per_device: float) -> Tensor:
    return Tensor._from_shards(
        tuple(_tensor(v).driver_tensor for v in per_device),
        _mesh2(),
        (Sharded(0),),
    )


def _replicated_spec(*shape: int) -> TensorLayout:
    return TensorLayout(
        _F32, list(shape), PlacementMapping(_mesh2(), (Replicated(),))
    )


def _replicated_tensor(*values: float) -> Tensor:
    return Tensor._from_shards(
        (_tensor(*values).driver_tensor,) * 2, _mesh2(), (Replicated(),)
    )


def _calls(graph: object, name: str) -> int:
    return len(re.findall(rf"mo\.call @{name}\b", str(graph)))


def _bodies(staged: object, name: str) -> int:
    return len(re.findall(rf"mo\.graph @{name}\b", str(staged)))


class TestAsLayout:
    def test_a_layout_passes_through(self) -> None:
        spec = _spec(4)
        assert as_layout(spec) is spec

    def test_a_tensor_type_becomes_a_one_device_layout(self) -> None:
        layout = as_layout(_dyn(4))
        assert isinstance(layout, TensorLayout)
        assert layout.mesh.num_devices == 1
        assert not layout.mesh.num_devices > 1

    def test_a_tensor_reads_as_its_own_layout(self) -> None:
        layout = as_layout(_tensor(1.0, 2.0, 3.0))
        assert isinstance(layout, TensorLayout)
        assert layout.dtype == _F32
        assert [int(d) for d in layout.shape] == [3]

    def test_a_sharded_tensor_keeps_its_distribution(self) -> None:
        layout = as_layout(_sharded_tensor(1.0, 2.0))
        assert isinstance(layout, TensorLayout)
        assert layout.mesh.num_devices == 2

    def test_anything_else_is_rejected(self) -> None:
        # ``Spec`` rules this out statically; the guard answers untyped callers.
        not_a_spec: Any = 3.0
        with pytest.raises(TypeError, match=re.escape("got float: 3.0")):
            as_layout(not_a_spec)


class TestStage:
    def test_a_call_is_traced_as_written(self) -> None:
        staged = stage(lambda x: x * 2)(_spec(4))
        assert staged.graph.name == "lambda"
        assert staged._signature.out_structure.num_leaves == 1
        assert staged._signal_device_ids == ()

    def test_an_example_tensor_serves_as_a_spec(self) -> None:
        staged = stage(lambda x: x * 2)(_tensor(1.0, 2.0))
        assert len(staged.graph.inputs) == 1

    def test_keyword_arguments_are_traced(self) -> None:
        staged = stage(lambda x, *, y: x + y)(_spec(2), y=_spec(2))
        assert len(staged.graph.inputs) == 2

    def test_a_non_tensor_argument_bakes_in(self) -> None:
        staged = stage(lambda x, alpha: x * alpha)(_spec(2), 3.0)
        assert len(staged.graph.inputs) == 1, "alpha is not a graph input"

    def test_a_nested_container_argument_flattens(self) -> None:
        staged = stage(lambda kv: kv["a"] + kv["b"])(
            {"a": _spec(2), "b": _spec(2)}
        )
        assert len(staged.graph.inputs) == 2

    def test_a_distributed_spec_threads_one_input_per_device(self) -> None:
        staged = stage(lambda x: x * 2)(_sharded_spec(4))
        assert len(staged.graph.inputs) == 2
        assert staged._signature.out_structure.num_leaves == 2

    def test_options_bind_to_the_transform_not_the_call(self) -> None:
        staged = stage(lambda x: x * 2, name="explicit")(_spec(2))
        assert staged.graph.name == "explicit"

    def test_an_option_name_may_also_be_a_parameter_name(self) -> None:
        staged = stage(lambda name: name * 2, name="g")(_spec(2))
        assert staged.graph.name == "g"
        assert len(staged.graph.inputs) == 1


class TestCompiled:
    def test_round_trip(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        np.testing.assert_allclose(run(_tensor(1.0, 2.0)).to_numpy(), [2, 4])

    def test_keyword_arguments_round_trip(self) -> None:
        run = compile(lambda x, *, y: x + y)(_spec(2), y=_spec(2))
        out = run(_tensor(1.0, 2.0), y=_tensor(3.0, 4.0))
        np.testing.assert_allclose(out.to_numpy(), [4, 6])

    def test_a_static_argument_must_match_what_was_traced(self) -> None:
        run = compile(lambda x, alpha: x * alpha)(_spec(2), 3.0)
        np.testing.assert_allclose(
            run(_tensor(1.0, 2.0), 3.0).to_numpy(), [3, 6]
        )
        with pytest.raises(ValueError, match="tree structure mismatch"):
            run(_tensor(1.0, 2.0), 4.0)

    def test_a_nested_container_result_round_trips(self) -> None:
        run = compile(lambda x: {"a": x * 2, "b": [x * 3]})(_spec(1))
        out = run(_tensor(2.0))
        np.testing.assert_allclose(out["a"].to_numpy(), [4.0])
        np.testing.assert_allclose(out["b"][0].to_numpy(), [6.0])

    def test_a_non_tensor_result_leaf_is_frozen(self) -> None:
        run = compile(lambda x: (x * 2, 7))(_spec(1))
        assert run(_tensor(1.0))[1] == 7

    def test_a_symbolic_dimension_accepts_any_size(self) -> None:
        run = compile(lambda x: x * 2)(_dyn("n"))
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [2.0])
        np.testing.assert_allclose(run(_tensor(1.0, 2.0)).to_numpy(), [2, 4])

    def test_a_wrong_shape_names_the_argument(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        with pytest.raises(ValueError, match=r"argument 0: expected"):
            run(_tensor(1.0, 2.0, 3.0))

    def test_a_wrong_keyword_shape_names_the_keyword(self) -> None:
        run = compile(lambda x, *, y: x + y)(_spec(2), y=_spec(2))
        with pytest.raises(ValueError, match=r"argument y: expected"):
            run(_tensor(1.0, 2.0), y=_tensor(1.0))

    @pytest.mark.skipif(not accelerator_count(), reason="needs a second device")
    def test_a_wrong_device_names_the_argument(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        with pytest.raises(ValueError, match=r"argument 0: expected"):
            run(_tensor(1.0, 2.0).to(Accelerator()))

    def test_a_buffer_where_a_tensor_belongs_is_rejected(self) -> None:
        """A buffer is named as a wrong argument type, not a wrong structure.

        Under one leaf policy a non-spec value flattens as static structure
        rather than as a leaf, so the naive comparison reports a tree mismatch.
        The call re-pairs the values to say which argument is wrong instead.
        """
        run = compile(lambda x: x * 2)(_spec(2))
        with pytest.raises(TypeError, match="argument 0: expected a Tensor"):
            run(_tensor(1.0, 2.0).driver_tensor)

    def test_a_spec_where_a_tensor_belongs_is_rejected(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        with pytest.raises(TypeError, match="expected a Tensor"):
            run(_spec(2))

    def test_execute_raw_matches_call(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        x = _tensor(1.0, 2.0)
        raw = run.execute_raw(x.driver_tensor)
        np.testing.assert_allclose(
            Tensor(storage=raw[0]).to_numpy(), run(x).to_numpy()
        )

    def test_export_mef(self, tmp_path) -> None:  # noqa: ANN001
        run = compile(lambda x: x * 2)(_spec(2))
        path = tmp_path / "model.mef"
        run.export_mef(path)
        assert path.stat().st_size > 0
        assert "_engine_model" not in vars(run), "exporting must not initialize"

    def test_tracing_yields_a_graph_that_cannot_run(self) -> None:
        staged = stage(lambda x: x * 2)(_spec(2))
        assert isinstance(staged, StagedGraph)
        assert not callable(staged), "a graph is not an executable"

    def test_a_traced_graph_compiles_to_the_same_executable(self) -> None:
        run = stage(lambda x: x * 2)(_spec(2)).compile()
        np.testing.assert_allclose(run(_tensor(1.0, 2.0)).to_numpy(), [2, 4])

    def test_compiling_does_not_initialize(self) -> None:
        run = compile(lambda x: x * 2)(_spec(2))
        assert "_engine_model" not in vars(run)
        run(_tensor(1.0, 2.0))
        assert "_engine_model" in vars(run), "the first call initializes"

    def test_a_distributed_call_round_trips(self) -> None:
        run = compile(lambda x: x * 2)(_sharded_spec(2))
        out = run(_sharded_tensor(1.0, 2.0))
        assert out.is_distributed
        np.testing.assert_allclose(out.to_numpy(), [2.0, 4.0])


class TestWeights:
    def test_a_weight_is_external_and_computes(self) -> None:
        def layer(x: Tensor) -> Tensor:
            return x * F.constant_external(
                "w", TensorType(_F32, [2], DeviceRef.CPU())
            )

        assert "mo.constant.external" in str(stage(layer)(_spec(2)))
        run = compile(layer, weights={"w": _tensor(2.0, 3.0)})(_spec(2))
        np.testing.assert_allclose(
            run(_tensor(1.0, 1.0)).to_numpy(), [2.0, 3.0]
        )


class TestSubgraphable:
    def test_identical_calls_share_one_body(self) -> None:
        @as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        staged = stage(lambda x: block(block(block(x))))(_spec(2))
        assert _calls(staged.graph, "block") == 3

    def test_a_keyed_body_is_traced_once(self) -> None:
        body_runs = []

        def block(x: Tensor) -> Tensor:
            body_runs.append(1)
            return x * 2

        shared = as_subgraph(block, key="block")
        staged = stage(lambda x: shared(shared(shared(x))))(_spec(2))
        assert _calls(staged.graph, "block") == 3
        assert len(body_runs) == 1, (
            "a keyed second call must reuse, not retrace"
        )

    def test_a_closure_is_not_keyed_by_its_name(self) -> None:
        def scaled(alpha: float) -> Callable[[Tensor], Tensor]:
            @as_subgraph
            def block(x: Tensor) -> Tensor:
                return x * alpha

            return block

        staged = stage(lambda x: scaled(3.0)(scaled(2.0)(x)))(_spec(2))
        assert _calls(staged.graph, "block") == 1
        assert _calls(staged.graph, "block_1") == 1

    def test_a_differing_static_argument_yields_two_bodies(self) -> None:
        @as_subgraph
        def block(x: Tensor, alpha: float) -> Tensor:
            return x * alpha

        staged = stage(lambda x: block(block(x, 2.0), 3.0))(_spec(2))
        assert _calls(staged.graph, "block") == 1
        assert _calls(staged.graph, "block_1") == 1

    def test_keyword_tensor_arguments_become_operands(self) -> None:
        @as_subgraph
        def block(x: Tensor, *, w: Tensor) -> Tensor:
            return x * w

        w0, w1 = _tensor(2.0), _tensor(3.0)
        staged = stage(lambda x: block(block(x, w=w0), w=w1))(_spec(1))
        assert _calls(staged.graph, "block") == 2

    def test_a_shared_body_computes_correctly(self) -> None:
        @as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        run = compile(lambda x: block(block(x)))(_spec(1))
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [4.0])

    def test_allow_subgraphs_false_inlines(self) -> None:
        @as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        staged = stage(lambda x: block(block(x)), allow_subgraphs=False)(
            _spec(2)
        )
        assert _calls(staged.graph, "block") == 0

    def test_outside_a_capture_raises(self) -> None:
        @as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        with pytest.raises(TypeError, match="needs a capture"):
            block(_tensor(1.0))

    def test_a_placeholder_prefix_resolves_per_call_site(self) -> None:
        weights = {"l0.w": _tensor(2.0), "l1.w": _tensor(10.0)}

        def body(x: Tensor) -> Tensor:
            return x * F.constant_external(
                "w",
                TensorType(_F32, [1], DeviceRef.CPU()),
                is_placeholder=True,
            )

        def model(x: Tensor) -> Tensor:
            for name in ("l0.", "l1."):
                x = as_subgraph(body, name="block", prefix=name)(x)
            return x

        staged = stage(model)(_spec(1))
        assert _calls(staged.graph, "block") == 2
        run = staged.compile(weights=weights)
        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [20.0])

    def test_declaring_in_full_inside_a_body_overrides_the_prefix(self) -> None:
        """``is_placeholder=False`` under a prefix: one shared name, not one each.

        The default follows the prefix, which is what a stack of layers wants.
        Overridden, every call site reads the same checkpoint entry -- a tied
        weight -- and the name in the body is already complete.
        """

        def body(x: Tensor) -> Tensor:
            return x * F.constant_external(
                "shared.w",
                TensorType(_F32, [1], DeviceRef.CPU()),
                is_placeholder=False,
            )

        def model(x: Tensor) -> Tensor:
            for name in ("l0.", "l1."):
                x = as_subgraph(body, name="block", prefix=name)(x)
            return x

        staged = stage(model)(_spec(1))
        whole = str(staged)
        assert _calls(staged.graph, "block") == 2, "still one body, two calls"
        run = staged.compile(weights={"shared.w": _tensor(3.0)})
        assert 'name = "shared.w"' in whole
        assert "isPlaceholder = true" not in whole

        np.testing.assert_allclose(run(_tensor(1.0)).to_numpy(), [9.0])


def _gnarly(x, /, pair, scale=2.0, *rest, bias, flag=True, **extras):  # noqa: ANN001, ANN202
    """Every parameter kind Python has, with tensors and statics interleaved.

    Positional-only, positional-or-keyword holding a container, a defaulted
    positional that stays static, var-positional tensors, a required
    keyword-only tensor, a defaulted keyword-only static, and var-keyword
    tensors whose iteration order is observable.
    """
    out = x * scale + pair["lo"] - pair["hi"]
    for tensor in rest:
        out = out + tensor
    out = out + bias if flag else out - bias
    for name in sorted(extras):
        out = out * extras[name] + 1.0
    return {"out": out, "arity": len(rest), "tags": tuple(sorted(extras))}


def _gnarly_specs(spec):  # noqa: ANN001, ANN202
    return (
        (spec(2), {"lo": spec(2), "hi": spec(1)}, 3.0, spec(2), spec(2)),
        {"bias": spec(2), "flag": False, "alpha": spec(2), "beta": spec(2)},
    )


def _gnarly_call(make):  # noqa: ANN001, ANN202
    return (
        (
            make(1.0, 2.0),
            {"lo": make(10.0, 10.0), "hi": make(1.0)},
            3.0,
            make(100.0, 100.0),
            make(1000.0, 1000.0),
        ),
        {
            "bias": make(2.0, 2.0),
            "flag": False,
            "alpha": make(2.0, 2.0),
            "beta": make(0.5, 0.5),
        },
    )


_GNARLY_EXPECTED = [1111.5, 1114.5]

# Flatten order is not call order: `pair`'s dict sorts "hi" before "lo", and the
# keyword block sorts alphabetically, so `bias` trails the var-keyword tensors.
_GNARLY_ROUTES = [
    "0.0",
    "0.1.hi",
    "0.1.lo",
    "0.3",
    "0.4",
    "1.alpha",
    "1.beta",
    "1.bias",
]


#: Appended to once per stage of the bodies below, so a test can tell a body
#: that was staged from one that was answered from the cache.
_TRACED: list[str] = []


def _plain_block(x: Tensor) -> Tensor:
    """A body with no closure, so its name and arguments determine its IR."""
    _TRACED.append("plain")
    return x * 2.0


def _indexed_block(x: Tensor, idx: int) -> Tensor:
    """A body whose per-layer index is baked in, the ``layer_idx`` shape."""
    _TRACED.append(f"indexed:{idx}")
    return x * float(idx)


def _closing_block(scale: float) -> Callable[[Tensor], Tensor]:
    """Returns a body that captures ``scale``, which no key can see."""

    def block(x: Tensor) -> Tensor:
        _TRACED.append(f"closed:{scale}")
        return x * scale

    return block


class TestSubgraphDedupIsFastByDefault:
    """A repeat call should cost a ``mo.call``, not another stage.

    Deduplicating on the staged IR is the only way to tell two bodies apart
    when nothing else can, but it means every repeat is staged and thrown
    away -- sixty-one times over for a transformer's layers. A body whose IR
    follows from its own code and its arguments does not need that, and both
    of those are already known before staging.
    """

    def test_an_identical_repeat_is_answered_without_tracing_again(
        self,
    ) -> None:
        _TRACED.clear()

        def stack(x: Tensor) -> Tensor:
            for _ in range(3):
                x = as_subgraph(_plain_block)(x)
            return x

        staged = stage(stack)(_spec(2))

        assert _TRACED == ["plain"], "the body was staged more than once"
        assert _calls(staged.graph, "plain_block") == 3

    def test_a_differing_baked_index_still_splits_the_bodies(self) -> None:
        """The ``layer_idx`` case: same function, different constant per layer.

        The index is an argument rather than a captured value, so it reaches
        the key through the argument structure and the layers are told apart
        without staging them to find out.
        """
        _TRACED.clear()

        def stack(x: Tensor) -> Tensor:
            for i in range(3):
                x = as_subgraph(_indexed_block)(x, i)
            return x

        staged = stage(stack)(_spec(2))

        assert _TRACED == ["indexed:0", "indexed:1", "indexed:2"]
        # Three bodies, so three symbols, each called once.
        assert _calls(staged.graph, "indexed_block") == 1
        assert _calls(staged.graph, "indexed_block_1") == 1
        assert _calls(staged.graph, "indexed_block_2") == 1

    def test_a_captured_difference_falls_back_to_comparing_the_ir(
        self,
    ) -> None:
        """What the fast key cannot see, the slow path still catches.

        A closure's body depends on what it captured, and no key derived from
        the callable can tell two of them apart -- so these keep being staged
        and compared, which is the point of not deriving a key for them.
        """
        _TRACED.clear()

        def stack(x: Tensor) -> Tensor:
            for scale in (2.0, 3.0, 2.0):
                x = as_subgraph(_closing_block(scale), name="closing")(x)
            return x

        staged = stage(stack)(_spec(2))

        # All three staged, since only the IR can separate them...
        assert _TRACED == ["closed:2.0", "closed:3.0", "closed:2.0"]
        # ...but the two that match still share one body.
        assert _calls(staged.graph, "closing") == 2
        assert _calls(staged.graph, "closing_1") == 1

    def test_a_default_argument_also_falls_back_to_comparing_the_ir(
        self,
    ) -> None:
        """A default is captured state, so no key is derived from the callable.

        Two factory-made bodies share a qualname and have no closure; only the
        default separates them, and it never crosses the call boundary, so a
        derived key would let the second silently reuse the first's IR.
        """
        _TRACED.clear()

        def defaulted_block(scale: float) -> Callable[[Tensor], Tensor]:
            def block(x: Tensor, s: float = scale) -> Tensor:
                _TRACED.append(f"default:{s}")
                return x * s

            return block

        def stack(x: Tensor) -> Tensor:
            for scale in (2.0, 3.0):
                x = as_subgraph(defaulted_block(scale), name="defaulted")(x)
            return x

        staged = stage(stack)(_spec(2))

        assert _TRACED == ["default:2.0", "default:3.0"]
        assert _calls(staged.graph, "defaulted") == 1
        assert _calls(staged.graph, "defaulted_1") == 1

    def test_an_explicit_none_still_compares_the_ir(self) -> None:
        """``key=None`` is a request, not an absence.

        A module holding real data passes it deliberately: its constants are
        inlined, so two bodies differ in ways only the IR shows.
        """
        _TRACED.clear()

        def stack(x: Tensor) -> Tensor:
            for _ in range(3):
                x = as_subgraph(_plain_block, key=None)(x)
            return x

        str(stage(stack)(_spec(2)))

        assert _TRACED == ["plain"] * 3, "key=None should have staged each time"


class TestTheHardestSignature:
    def test_every_tensor_slot_becomes_an_input_and_the_rest_bakes_in(
        self,
    ) -> None:
        args, kwargs = _gnarly_specs(_spec)
        staged = stage(_gnarly)(*args, **kwargs)
        assert len(staged.graph.inputs) == 8
        assert staged._signature.out_structure.num_leaves == 1, (
            "arity and tags are static"
        )

    def test_the_route_to_every_tensor_is_addressable(self) -> None:
        args, kwargs = _gnarly_specs(_spec)
        staged = stage(_gnarly)(*args, **kwargs)
        assert (
            list(tree_paths(staged._signature.in_specs, leaf=_SPEC_TYPES))
            == _GNARLY_ROUTES
        )

    def test_it_round_trips_compiled(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        out = run(*args, **kwargs)
        np.testing.assert_allclose(out["out"].to_numpy(), _GNARLY_EXPECTED)
        assert out["arity"] == 2
        assert out["tags"] == ("alpha", "beta")

    def test_example_tensors_serve_as_specs(self) -> None:
        args, kwargs = _gnarly_call(_tensor)
        run = compile(_gnarly)(*args, **kwargs)
        np.testing.assert_allclose(
            run(*args, **kwargs)["out"].to_numpy(), _GNARLY_EXPECTED
        )

    def test_a_var_positional_tensor_is_named_by_its_index(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        args = (*args[:4], _tensor(1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match=r"argument 4: expected"):
            run(*args, **kwargs)

    def test_a_nested_positional_tensor_is_named_by_its_route(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        args = (args[0], {**args[1], "lo": _tensor(1.0)}, *args[2:])
        with pytest.raises(ValueError, match=r"argument 1\.lo: expected"):
            run(*args, **kwargs)

    def test_a_var_keyword_tensor_is_named_by_its_keyword(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        with pytest.raises(ValueError, match=r"argument beta: expected"):
            run(*args, **{**kwargs, "beta": _tensor(1.0, 2.0, 3.0)})

    def test_a_differing_keyword_only_static_is_a_different_signature(
        self,
    ) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        with pytest.raises(ValueError, match="tree structure mismatch"):
            run(*args, **{**kwargs, "flag": True})

    def test_an_extra_var_keyword_tensor_is_a_different_signature(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        with pytest.raises(ValueError, match=r"got \[.*'gamma'\]"):
            run(*args, **kwargs, gamma=_tensor(1.0, 1.0))

    def test_dropping_a_positional_shifts_the_rest_and_is_rejected(
        self,
    ) -> None:
        """`scale` is positional-or-keyword, so a tensor would slide into its slot.

        Counting the positionals is what stops it: the slide is only visible as
        one argument too few, and saying so beats reporting the type error the
        shifted tensor would go on to cause in ``scale``'s place.
        """
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args, **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        with pytest.raises(
            ValueError, match=r"expected 5 positional argument\(s\), got 4"
        ):
            run(args[0], args[1], *args[3:], **kwargs)

    def test_the_default_itself_traces_as_a_static(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_spec)
        run = compile(_gnarly)(*spec_args[:2], **spec_kwargs)
        args, kwargs = _gnarly_call(_tensor)
        out = run(*args[:2], **kwargs)
        # scale defaults to 2.0, and `rest` is empty.
        np.testing.assert_allclose(out["out"].to_numpy(), [10.5, 12.5])
        assert out["arity"] == 0


_STACK_WEIGHTS = {"blk.w": _tensor(2.0, 2.0)}


@as_subgraph
def _block(x, /, pair, *rest, gain, **extras):  # noqa: ANN001, ANN202
    out = x + pair[0] - pair[1]
    for tensor in rest:
        out = out * gain + tensor
    for name in sorted(extras):
        out = out + extras[name]
    return {"y": out, "depth": len(rest)}


def _stack(x, /, table, *, scale, **tails):  # noqa: ANN001, ANN202
    out = x * F.constant_external(
        "blk.w", TensorType(_F32, [2], DeviceRef.CPU())
    )
    for name in sorted(tails):
        out = _block(
            out,
            (table["lo"], table["hi"]),
            tails[name],
            gain=scale,
            bump=table["hi"],
        )["y"]
    return {"out": out, "kinds": tuple(sorted(tails))}


def _stack_specs(spec):  # noqa: ANN001, ANN202
    return (
        (spec(2), {"lo": spec(2), "hi": spec(1)}),
        {"scale": 3.0, "p": spec(2), "q": spec(2)},
    )


def _stack_call(make):  # noqa: ANN001, ANN202
    return (
        (make(1.0, 2.0), {"lo": make(10.0, 10.0), "hi": make(1.0)}),
        {"scale": 3.0, "p": make(100.0, 100.0), "q": make(1000.0, 1000.0)},
    )


_STACK_EXPECTED = [1430.0, 1448.0]


class TestEverythingMixed:
    def test_the_ultimate_stress_test(self) -> None:
        spec_args, spec_kwargs = _stack_specs(_spec)
        staged = stage(_stack)(*spec_args, **spec_kwargs)
        assert _calls(staged.graph, "block") == 2, "one body, two call sites"
        assert "mo.constant.external" in str(staged)
        run = staged.compile(weights=_STACK_WEIGHTS)
        args, kwargs = _stack_call(_tensor)
        out = run(*args, **kwargs)
        np.testing.assert_allclose(out["out"].to_numpy(), _STACK_EXPECTED)
        assert out["kinds"] == ("p", "q")

    def test_the_hardest_signature_across_a_replicated_mesh(self) -> None:
        spec_args, spec_kwargs = _gnarly_specs(_replicated_spec)
        staged = stage(_gnarly)(*spec_args, **spec_kwargs)
        assert len(staged.graph.inputs) == 16, "two shards per tensor"
        run = staged.compile()
        args, kwargs = _gnarly_call(_replicated_tensor)
        out = run(*args, **kwargs)
        assert out["out"].is_distributed
        np.testing.assert_allclose(out["out"].to_numpy(), _GNARLY_EXPECTED)
        assert out["tags"] == ("alpha", "beta")


@dataclass(frozen=True)
class Projections:
    """A record of tensors, as every layer in ``nn/functional`` holds.

    Spelled out rather than generated, since the node protocol is what is
    under test here as much as the round trip.
    """

    a: Tensor
    b: Tensor
    bias: Tensor | None = None

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Projections:
        del meta
        return cls(**children)


class TestARecordArgumentSurvivesTheRoundTrip:
    """A pytree record as an argument, all the way through execution.

    Tracing one was already covered; *calling* the result was not, and the two
    walks are different -- specs flatten under ``leaf=_SPEC_TYPES`` and a call
    flattens under ``_one_slot``. When those disagreed about a record the
    symptom was not a mismatch error: the record became one static slot, and
    checking it against its own spec ran ``Tensor.__eq__`` against a
    ``TensorLayout``, which stages an ``equal`` op instead of comparing.
    """

    def test_a_record_of_tensors_round_trips_compiled(self) -> None:
        run = compile(lambda p: p.a * p.b)(
            Projections(a=_tensor(0.0, 0.0), b=_tensor(0.0, 0.0))
        )

        out = run(Projections(a=_tensor(2.0, 3.0), b=_tensor(4.0, 5.0)))

        np.testing.assert_allclose(out.to_numpy(), [8.0, 15.0])

    def test_each_of_a_records_tensors_is_its_own_input(self) -> None:
        staged = stage(lambda p: p.a * p.b)(
            Projections(a=_tensor(0.0, 0.0), b=_tensor(0.0, 0.0))
        )

        assert len(staged.graph.inputs) == 2, "not one slot for the record"

    def test_a_wrong_shape_inside_a_record_names_its_field(self) -> None:
        run = compile(lambda p: p.a * p.b)(
            Projections(a=_tensor(0.0, 0.0), b=_tensor(0.0, 0.0))
        )

        with pytest.raises(ValueError, match=r"argument 0\.a: expected"):
            run(Projections(a=_tensor(1.0), b=_tensor(4.0, 5.0)))


class TestBufferSpecs:
    """A buffer input is how mutable state crosses a graph boundary.

    A layout describes where a value *sits*; a buffer exists so a kernel can
    write *through* it, which no layout says. Without buffer specs a paged KV
    cache cannot be a graph input at all: it is neither a leaf nor a container
    to the spec walk, so it contributes no input and the staged function
    receives the type object where it expected a tensor -- silently, with no
    error anywhere.
    """

    def test_a_buffer_spec_becomes_a_graph_input(self) -> None:
        spec = BufferType(DType.float32, [4, 8], device=DeviceRef.CPU())

        staged = compilation.stage(lambda b: b)(spec)

        assert len(staged.graph.inputs) == 1
        assert isinstance(staged.graph.inputs[0], BufferValue)

    def test_the_traced_function_receives_a_tensor(self) -> None:
        """What arrives is a :class:`Tensor` backed by the buffer, so the same
        model code works whether its cache is a graph input or an eager one."""
        seen: list[object] = []

        def keep(b: Tensor) -> TensorValue:
            seen.append(type(b))
            return ops.buffer_load(b.__buffervalue__())

        compilation.stage(keep)(
            BufferType(DType.float32, [4, 8], device=DeviceRef.CPU())
        )

        assert seen == [Tensor]

    def test_a_distributed_buffer_contributes_one_input_per_device(
        self,
    ) -> None:
        """The half that was never wired: a cache spread over a mesh."""
        mesh = DeviceMesh(
            devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",)
        )
        spec = DistributedBufferType(DType.float32, [4, 8], mesh, (Sharded(0),))

        def read_first(b: Tensor) -> TensorValue:
            return ops.buffer_load(b.local_shards[0].__buffervalue__())

        staged = compilation.stage(read_first)(spec)

        assert len(staged.graph.inputs) == 2
        assert all(isinstance(i, BufferValue) for i in staged.graph.inputs)

    def test_a_compiled_graph_writes_through_its_buffer_input(self) -> None:
        """The whole point: the caller's own memory holds what the graph wrote.

        A tensor input cannot show this. It is copied in, and a mutation of it
        dies with the execution. This is the property a KV cache needs and the
        only one that distinguishes a buffer input from a tensor one.
        """

        def fill(cache: Tensor, x: Tensor) -> Tensor:
            F.buffer_store(cache, x)
            return x * 2

        run = compile(fill)(
            BufferType(_F32, [3], device=DeviceRef.CPU()), _spec(3)
        )
        cache = _tensor(0.0, 0.0, 0.0)

        out = run(cache, _tensor(1.0, 2.0, 3.0))

        np.testing.assert_allclose(out.to_numpy(), [2, 4, 6])
        np.testing.assert_allclose(cache.to_numpy(), [1, 2, 3])

    def test_a_compiled_graph_writes_through_every_shard(self) -> None:
        """Sharded, so each device writes its own slice and no other."""
        mesh = _mesh2()

        def fill(cache: Tensor, x: Tensor) -> Tensor:
            F.buffer_store(cache, x)
            return x * 2

        run = compile(fill)(
            DistributedBufferType(_F32, [4], mesh, (Sharded(0),)),
            TensorLayout(_F32, [4], PlacementMapping(mesh, (Sharded(0),))),
        )
        cache = Tensor._from_shards(
            (
                _tensor(0.0, 0.0).driver_tensor,
                _tensor(0.0, 0.0).driver_tensor,
            ),
            mesh,
            (Sharded(0),),
        )
        source = Tensor._from_shards(
            (
                _tensor(1.0, 2.0).driver_tensor,
                _tensor(3.0, 4.0).driver_tensor,
            ),
            mesh,
            (Sharded(0),),
        )

        run(cache, source)

        np.testing.assert_allclose(cache.local_shards[0].to_numpy(), [1.0, 2.0])
        np.testing.assert_allclose(cache.local_shards[1].to_numpy(), [3.0, 4.0])


class TestExplicitMutability:
    """Which tensor a store may promote, and where a caller has to say so.

    A store into a tensor backed by a ``TensorValue`` creates a
    ``BufferValue`` and copies the value in. That is sound where the buffer it
    made is the one the reader will read, and wrong at a boundary, where the
    type is already fixed and the buffer is local to the graph that made it.
    """

    def test_writing_a_tensor_staged_argument_is_refused(self) -> None:
        """Promotion would store into a graph-local buffer the caller never
        sees, so it is an error rather than a lost write."""

        def writes_its_argument(cache: Tensor, x: Tensor) -> Tensor:
            cache[0:1] = x[0:1]
            return x

        with pytest.raises(TypeError, match="staged as a tensor"):
            stage(writes_its_argument)(_spec(2), _spec(2))

    def test_a_value_the_graph_computed_is_promoted_by_a_store(self) -> None:
        """Nothing outside the graph names it, so create-and-store is sound."""

        def writes_a_local(x: Tensor) -> Tensor:
            scratch = x * 1.0
            scratch[1:2] = x[1:2] * 2.0
            return scratch[...]

        assert "mo.buffer.create" in str(stage(writes_a_local)(_spec(2)))

    def test_a_body_writes_through_a_buffer_the_caller_promoted(self) -> None:
        """An operand's type is the body's argument type, so a value a body
        writes has to be a buffer before the call. A store in the caller
        promotes it; promoting inside the body would create a buffer the
        caller never sees and the store would be lost."""

        def block(scratch: Tensor, x: Tensor) -> Tensor:
            scratch[0:1] = x[0:1]
            return scratch[...]

        layer = as_subgraph(block, name="block")

        def model(x: Tensor) -> Tensor:
            scratch = x * 0.0
            scratch[0:1] = scratch[0:1]  # promotes, before the call
            body = layer(scratch, x)
            # Read back through the caller's own handle, not the call result.
            return body + scratch[...]

        staged = stage(model)(_spec(2))
        body = str(staged).split("mo.graph @block")[1]
        assert "!mo.buffer" in body.split("{")[0], "the body takes a buffer"
        assert "mo.buffer.create" not in body, "so it allocates none of its own"

    def test_without_that_store_the_body_is_refused(self) -> None:
        """The same model unpromoted: the operand is a tensor, so the body's
        argument is one, and the write that used to be lost is an error."""

        def block(scratch: Tensor, x: Tensor) -> Tensor:
            scratch[0:1] = x[0:1]
            return scratch[...]

        layer = as_subgraph(block, name="block")

        def model(x: Tensor) -> Tensor:
            scratch = x * 0.0
            return layer(scratch, x) + scratch[...]

        with pytest.raises(TypeError, match="staged as a tensor"):
            stage(model)(_spec(2))

    def test_one_pool_argument_serves_every_layer_call(self) -> None:
        """The production shape: a buffer declared at ``compile()``, threaded
        into one shared body, called once per layer, each writing its own
        slot. This is a paged KV cache with the kernel taken out."""

        def layer(pool: Tensor, x: Tensor, slot: Tensor) -> Tensor:
            pool[slot] = x
            return x * 2

        block = as_subgraph(layer, name="block")

        def model(pool: Tensor, x: Tensor) -> Tensor:
            for i in range(3):
                x = block(pool, x, F.constant(i, DType.int64, device=CPU()))
            return x

        staged = stage(model)(
            BufferType(_F32, [3, 2], device=DeviceRef.CPU()), _spec(2)
        )
        assert _bodies(staged, "block") == 1, "one body for every layer"
        assert _calls(staged.graph, "block") == 3
        body = str(staged).split("mo.graph @block")[1]
        assert "!mo.buffer" in body.split("{")[0], "the pool stays a buffer"
        assert "mo.buffer.create" not in body, "so none is allocated inside"

        run = staged.compile()
        pool = Tensor.from_dlpack(np.zeros((3, 2), dtype=np.float32))
        out = run(pool, _tensor(1.0, 2.0))

        np.testing.assert_allclose(out.to_numpy(), [8.0, 16.0])
        np.testing.assert_allclose(
            pool.to_numpy(), [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]
        )


@dataclass(frozen=True)
class PagedSpec:
    """How a :class:`PagedCacheValues` describes its boundary.

    Mirrors every field, so the tree matches and unflattening hands the body
    the production class. Only ``kv_blocks`` is a buffer.
    """

    kv_blocks: BufferType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_prompt_length: TensorType
    max_cache_length: TensorType
    kv_scales: None = None
    attention_dispatch_metadata: TensorType | None = None
    mla_num_partitions: None = None

    def __tree_flatten__(self) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        names = tuple(f.name for f in fields(self))
        return tuple(getattr(self, name) for name in names), names

    @classmethod
    def __tree_unflatten__(
        cls, aux: tuple[str, ...], children: Sequence[Any]
    ) -> PagedCacheValues:
        return PagedCacheValues(**dict(zip(aux, children, strict=True)))


class TestAPagedKVCache:
    """The production paged cache through a shared body, which is the point.

    :meth:`KVCacheParams.get_symbolic_inputs` already types one device's
    boundary as a ``BufferType`` for the page pool and a ``TensorType`` per
    piece of metadata, and :class:`PagedCacheValues` already carries the
    pytree protocol. So a real attention layer reaches this with nothing
    added to either, which is the test: whether a ModuleV3-style paged
    pipeline can be written against this API unchanged.

    Staged rather than run: the page contents a kernel reads are the
    integration suite's business, while the boundary is this file's.
    """

    LAYERS = 2

    def _boundary(self) -> tuple[Any, ...]:
        """The production description of a paged cache, converted nowhere.

        ``get_symbolic_inputs`` already types the pool as a ``BufferType`` and
        each piece of metadata as a ``TensorType``, and ``PagedCacheValues``
        already carries the pytree protocol, so a pipeline reaches this with
        nothing added to either.
        """
        heads, head_dim = 4, 16
        hidden = heads * head_dim
        kv_params = MHAKVCacheParams(
            dtype=_F32,
            n_kv_heads=1,
            head_dim=head_dim,
            num_layers=self.LAYERS,
            page_size=128,
            devices=[DeviceRef.CPU()],
        )
        declared = kv_params.get_symbolic_inputs().inputs[0]
        cache_spec = PagedSpec(
            kv_blocks=declared.kv_blocks,
            cache_lengths=declared.cache_lengths,
            lookup_table=declared.lookup_table,
            max_prompt_length=declared.max_prompt_length,
            max_cache_length=declared.max_cache_length,
            attention_dispatch_metadata=declared.attention_dispatch_metadata,
        )
        x_spec = TensorType(
            _F32, ["total_seq_len", hidden], device=DeviceRef.CPU()
        )
        rows_spec = TensorType(DType.uint32, ["rows"], device=DeviceRef.CPU())
        return hidden, cache_spec, x_spec, rows_spec

    def _transformer(self) -> tuple[Any, ...]:
        (
            hidden,
            cache_spec,
            x_spec,
            rows_spec,
        ) = self._boundary()

        def projection(name: str) -> Tensor:
            return F.constant_external(
                name,
                TensorType(_F32, [hidden, hidden], device=DeviceRef.CPU()),
                is_placeholder=True,
            )

        def block(
            x: Tensor,
            cache: PagedCacheValues,
            layer_idx: Tensor,
            input_row_offsets: Tensor,
        ) -> Tensor:
            # Where attention would read and write the pool. A kernel is not
            # this file's business; that the pool is a buffer here is.
            cache.kv_blocks[...] = cache.kv_blocks[...] * 1.0
            return (x @ projection("attn.qkv")) @ projection("attn.o")

        def transformer(
            x: Tensor, cache: PagedCacheValues, input_row_offsets: Tensor
        ) -> Tensor:
            for i in range(self.LAYERS):
                # The index crosses as an operand. Baked in as a constant, the
                # way a layer holding its own does, it would be one body per
                # layer.
                x = as_subgraph(block, name="block", prefix=f"layers.{i}.")(
                    x,
                    cache,
                    F.constant(i, DType.uint32, device=CPU()),
                    input_row_offsets,
                )
            return x

        return transformer, cache_spec, x_spec, rows_spec

    def test_it_shares_one_body_over_one_pool_with_weights_by_prefix(
        self,
    ) -> None:
        """Staged once, because every claim below is about the same graph."""
        transformer, cache_spec, x_spec, rows_spec = self._transformer()

        staged = stage(transformer)(x_spec, cache_spec, rows_spec)

        assert _bodies(staged, "block") == 1, "one body, not one per layer"
        assert _calls(staged.graph, "block") == self.LAYERS
        buffers = [i for i in staged.graph.inputs if isinstance(i, BufferValue)]
        assert len(buffers) == 1, "one page pool, shared by every layer"

        body = str(staged).split("mo.graph @block")[1]
        signature = body.split("{")[0]
        assert "!mo.buffer" in signature, "the pool is a buffer in the body"
        assert "total_num_pages" in signature, "and stays symbolic"
        assert "mo.buffer.create" not in body, "so nothing is copied in"

        assert 'name = "attn.qkv"' in body, "weights are named relatively"
        assert "isPlaceholder = true" in body
        for i in range(self.LAYERS):
            assert f'prefix = "layers.{i}."' in str(staged.graph)
