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
"""Tests attribute factories."""

import functools

import pytest
from max import mlir
from max._core import (
    Block,
    InsertPoint,
    NamedAttribute,
    OpBuilder,
    Operation,
    Pass,
    Type,
    lower,
)
from max._core.dialects import builtin, kgen, m, mo, mosh, rmo
from max._core.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_mo_attr(mlir_context) -> None:  # noqa: ANN001
    attr = mo.DTypeAttr(DType.bool)
    assert attr.dtype == DType.bool
    assert attr == mo.DTypeAttr(DType.bool)
    assert attr != mo.DTypeAttr(DType.int8)


def test_mosh(mlir_context) -> None:  # noqa: ANN001
    shape_type = mosh.ShapeType()
    assert isinstance(shape_type, mosh.ShapeType)
    assert isinstance(shape_type, Type)
    assert shape_type == mosh.ShapeType()


def test_mosh_shapeattr(mlir_context) -> None:  # noqa: ANN001
    shape_type = mosh.ShapeType()
    attr = mosh.ShapeAttr([1, 2, 3], shape_type)
    dims = [kgen.CastToBuiltinAttr(d) for d in attr.values]
    index_type = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
    uint8_type = builtin.IntegerType(8, builtin.SignednessSemantics.unsigned)
    Index = functools.partial(builtin.IntegerAttr, index_type)
    UInt8 = functools.partial(builtin.IntegerAttr, uint8_type)
    expected = [Index(1), Index(2), Index(3)]
    assert dims == expected
    assert dims != []
    assert dims != [Index(1), Index(1), Index(1)]
    assert dims != [UInt8(1), UInt8(2), UInt8(3)]


def test_mosh_shapeattr_empty(mlir_context) -> None:  # noqa: ANN001
    shape_type = mosh.ShapeType()
    attr = mosh.ShapeAttr([], shape_type)
    assert list(attr.values) == []


def test_context_always_active() -> None:
    assert mlir.Context.current
    shape_type = mosh.ShapeType()
    assert shape_type


def test_builtin_integerattr(mlir_context) -> None:  # noqa: ANN001
    int_type = builtin.IntegerType(1, builtin.SignednessSemantics.unsigned)
    int_attr = builtin.IntegerAttr(int_type, 1)
    assert int_attr.type == int_type


def test_builtin_moduleop(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    builtin.ModuleOp(loc)


def test_mo_graph_op(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [], is_subgraph=False)

    assert graph.sym_name == "hello"
    assert list(graph.input_parameters) == []
    assert graph.function_type == builtin.FunctionType([], [])


@pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
def test_operation_bytecode_round_trips_across_contexts(
    mlir_context,  # noqa: ANN001
    dtype: DType,
) -> None:
    """``Operation.bytecode`` / ``from_bytecode`` move IR between MLIRContexts
    losslessly. f64 is the load-bearing case: the textual round-trip
    (``parse_module``) mangles f64 float literals, so bytecode must not.
    """
    with Graph(
        "round_trip",
        input_types=[TensorType(dtype, [2], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = graph.inputs
        graph.output(ops.add(x, ops.constant(2.0, dtype, DeviceRef.CPU())))
    module = graph._module

    data = module.bytecode
    assert isinstance(data, bytes)
    assert data

    # Re-serializing the module parsed into a fresh context yields byte-
    # identical bytecode: the cross-context round-trip is lossless. The target
    # context must outlive the parsed module, so bind it to a local.
    target = mlir.Context()
    restored = Operation.from_bytecode(data, target)
    assert restored.bytecode == data


def test_infer_type_op_adaptor() -> None:
    input_type = TensorType(DType.float32, [1], DeviceRef.GPU())
    with Graph("empty", input_types=[input_type, input_type]) as graph:
        with mlir.Location.unknown() as location:
            assert isinstance(location, mlir.Location)
            builder = OpBuilder(Block._from_cmlir(graph._current_block).end)
            x, y = graph.inputs
            params = kgen.ParamDeclArrayAttr([])
            op = rmo.AddOp(builder, location, x.to_mlir(), y.to_mlir(), params)
            op.verify()


def test_regions_and_blocks(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [], is_subgraph=False)

    block = graph.regions[0].front
    assert isinstance(block, Block)

    del builder
    del graph
    del module

    # check that we can still safely access the block
    ip = block.end
    assert isinstance(ip, InsertPoint)


def test_free_standing_block_allocation(mlir_context: mlir.Context) -> None:
    """A `Block(arg_types, arg_locs)` allocated via the constructor is owned
    by the Python wrapper. Dropping the wrapper frees the block — we can't
    directly observe the destructor, but a double-free or leak in the
    binding would surface under ASAN.
    """
    loc = mlir.Location.current
    assert loc
    int64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
    block = Block(arg_types=[int64, int64], arg_locs=[loc, loc])
    args = list(block.arguments)
    assert len(args) == 2
    assert all(arg.type == int64 for arg in args)
    del block  # Python owns; this should free without error.


def test_block_append_transfers_ownership(mlir_context: mlir.Context) -> None:
    """`Region.append` splices the block into the region's intrusive block
    list and marks the Python wrapper as non-owning. The wrapper remains a
    valid view: attribute access continues to work both before and after
    append, and dropping the wrapper does NOT double-free the block.
    """
    loc = mlir.Location.current
    assert loc
    int64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "host", [], [], is_subgraph=False)
    region = graph.regions[0]

    block = Block(arg_types=[int64], arg_locs=[loc])
    assert len(list(block.arguments)) == 1

    region.append(block)

    # Wrapper is still usable as a non-owning view.
    assert len(list(block.arguments)) == 1

    # nanobind's instance cache returns the same wrapper for the same C++
    # pointer, so re-fetching via the region yields the same object.
    assert region.back is block


def test_block_outlives_dropped_region_via_keep_alive(
    mlir_context: mlir.Context,
) -> None:
    """After `Region.append`, the block wrapper holds a `keep_alive` on the
    region, which transitively keeps the parent op chain alive. So even
    after callers drop every Python reference to the region, module, and
    builder, the original block reference remains valid — both the
    Python wrapper and the underlying MLIR block survive because the
    wrapper is rooted in the live op tree.
    """
    loc = mlir.Location.current
    assert loc
    int64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "host", [], [], is_subgraph=False)
    region = graph.regions[0]
    block = Block(arg_types=[int64], arg_locs=[loc])
    region.append(block)

    del region
    del graph
    del builder
    del module

    assert len(list(block.arguments)) == 1


def test_block_append_rejects_already_attached_block(
    mlir_context: mlir.Context,
) -> None:
    """`Region.append` rejects blocks already attached to another region.
    The underlying `mlir::Region::push_back` is backed by
    `llvm::iplist::push_back`, which asserts the node isn't already in
    another list — in debug builds that's a process abort, and in release
    builds it silently corrupts both intrusive lists. The binding catches
    the case early and raises ``ValueError`` (nanobind's mapping for
    ``std::invalid_argument``).
    """
    loc = mlir.Location.current
    assert loc
    int64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph1 = mo.GraphOp(builder, loc, "host1", [], [], is_subgraph=False)
    graph2 = mo.GraphOp(builder, loc, "host2", [], [], is_subgraph=False)

    block = Block(arg_types=[int64], arg_locs=[loc])
    graph1.regions[0].append(block)
    with pytest.raises(ValueError, match="already attached"):
        graph2.regions[0].append(block)


def test_block_drop_after_append_does_not_double_free(
    mlir_context: mlir.Context,
) -> None:
    """After `Region.append`, the region owns the block (and frees it via
    the region's destructor / parent op tear-down). Dropping the Python
    wrapper must NOT also delete the block — otherwise the region
    destructor would double-free.

    Observed indirectly: this test simply allocates, appends, drops the
    wrapper, and then lets the parent op chain go out of scope. A
    double-free would surface under ASAN.
    """
    loc = mlir.Location.current
    assert loc
    int64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "host", [], [], is_subgraph=False)
    region = graph.regions[0]
    block = Block(arg_types=[int64], arg_locs=[loc])
    region.append(block)
    del block  # Wrapper drop must not delete the block (region owns it).


def test_block_contents(mlir_context: mlir.Context) -> None:
    loc = mlir.Location.current
    assert loc

    module = builtin.ModuleOp(loc)
    block = module.body
    builder = OpBuilder(block.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [], is_subgraph=False)

    assert isinstance(block, Block)
    assert len(block) == 1
    [op] = block
    assert block[0] == block[-1] == op == graph


def test_op_operands() -> None:
    input_type = TensorType(DType.float32, [1], DeviceRef.GPU())
    with Graph("empty", input_types=[input_type, input_type]) as graph:
        with mlir.Location.unknown() as location:
            assert isinstance(location, mlir.Location)
            builder = OpBuilder(Block._from_cmlir(graph._current_block).end)
            x, y = graph.inputs
            params = kgen.ParamDeclArrayAttr([])
            op = rmo.AddOp(builder, location, x.to_mlir(), y.to_mlir(), params)
            op.verify()

        assert len(op.operands) == 2
        assert op.operands[0].value == x.to_mlir()
        assert op.operands[1].value == y.to_mlir()


def test_device_ref_attr(mlir_context) -> None:  # noqa: ANN001
    attr = m.DeviceRefAttr("cpu", 0)
    assert attr.label == "cpu"
    assert attr.id == 0


def test_dictattr_arrayview(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    assert list(attr.value) == [na]


def test_arrayview_dead_attr_reference(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    array_view = attr.value
    del attr
    assert array_view[0] == na


def test_arrayview_dead_array_reference(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    out = attr.value[0]
    del attr
    assert out == na


def test_discardable_attributes(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])

    attrs = graph.discardable_attributes

    # empty, even though graph has inherent attributes
    assert not attrs
    assert len(attrs) == 0
    assert dict(attrs.items()) == {}

    # __setitem__, __getitem__
    attrs["foo"] = builtin.StringAttr("foo")
    assert attrs
    assert len(attrs) == 1
    assert dict(attrs.items()) == {"foo": builtin.StringAttr("foo")}
    assert attrs["foo"] == builtin.StringAttr("foo")
    with pytest.raises(KeyError):
        attrs["bar"]

    signature = graph.signature
    # Set an inherent attribute
    # "signature" is inherent on `mo.GraphOp`
    attrs["signature"] = builtin.StringAttr("foo")
    assert "signature" in attrs
    assert signature == graph.signature  # inherent attribute is still fine
    assert attrs["signature"] == builtin.StringAttr("foo")
    del attrs["signature"]

    # __contains__
    assert "foo" in attrs
    assert "bar" not in attrs

    # __iter__
    assert list(attrs) == list(attrs.keys()) == ["foo"]

    # __del__
    del attrs["foo"]
    assert not attrs
    assert len(attrs) == 0
    assert dict(attrs.items()) == {}


def test_discardable_attrs__op_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    del graph
    del builder
    del module
    assert list(attrs) == ["foo"]


def test_discardable_attrs__dict_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs
    del graph
    del builder
    del module
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__attr_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs["foo"]
    assert "foo" not in attrs
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__concurrent_modification(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes

    attrs["foo"] = builtin.StringAttr("foo")
    attrs["bar"] = builtin.StringAttr("bar")

    keys = iter(attrs)
    items = iter(attrs.items())
    values = iter(attrs.values())

    keys2 = iter(attrs)
    next(keys2)
    items2 = iter(attrs.items())
    next(items2)
    values2 = iter(attrs.values())
    next(values2)

    del attrs["foo"]
    del attrs["bar"]
    assert not attrs

    # just make sure these don't error, there's not a defined behavior
    _ = list(keys)
    _ = list(items)
    _ = list(values)
    _ = list(keys2)
    _ = list(items2)
    _ = list(values2)


def test_lower_remove_dead_values(mlir_context) -> None:  # noqa: ANN001
    with Graph("empty", input_types=[]) as graph:
        graph.output()
    module = graph._module
    assert "mo.chain.create()" in module.asm()
    lower(module, [builtin.passes.RemoveDeadValuesPass()])
    assert "mo.chain.create()" not in module.asm()


def test_lowering_failure_diagnostic(mlir_context) -> None:  # noqa: ANN001
    # graph with no output!
    graph = Graph("empty", input_types=[])
    module = graph._module
    with pytest.raises(Exception):
        module.verify()
    with pytest.raises(Exception):
        lower(module, [builtin.passes.RemoveDeadValuesPass()])


def test_construct_pass_with_options(mlir_context) -> None:  # noqa: ANN001
    # Tablegen doesn't generate a public-visibility way to inspect
    # pass options, so don't try to test the actual pass option values.

    no_options = mo.passes.MOToMOGG()
    assert isinstance(no_options, Pass)
    assert no_options.name == "MOToMOGG"

    with_options = mo.passes.MOToMOGG(
        kernel_library_paths=["foo", "bar"],
        force_sync=True,
    )
    assert isinstance(with_options, Pass)
    assert with_options.name == "MOToMOGG"


def test_get_context_from_cpp(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    assert loc
    module = builtin.ModuleOp(loc)
    assert module.context is mlir_context


# ===----------------------------------------------------------------------=== #
# InferTypeOp-overload binding path (codegen'd state population +
# `finalize_checked`)
# ===----------------------------------------------------------------------=== #


def test_integer_attr_with_index_type(mlir_context) -> None:  # noqa: ANN001
    """`IntegerAttr` accepts `IndexType` (MLIR's `IndexAttr` pattern)."""
    index_type = builtin.IndexType()
    attr = builtin.IntegerAttr(index_type, 5)
    assert attr.type == index_type
    assert attr.value == 5
    # Default value is 0.
    assert builtin.IntegerAttr(index_type).value == 0


def test_infer_type_op_overload_infers_result_type() -> None:
    """The InferTypeOp overload of an `InferTypeOpAdaptor` op runs
    `Op::inferReturnTypes` and produces the right result type."""
    input_type = TensorType(DType.float32, [4], DeviceRef.GPU())
    with Graph("infer_type_op", input_types=[input_type, input_type]) as graph:
        with mlir.Location.unknown() as location:
            builder = OpBuilder(Block._from_cmlir(graph._current_block).end)
            x, y = graph.inputs
            params = kgen.ParamDeclArrayAttr([])
            # The 5-arg form (no `result=`) is the InferTypeOp overload.
            op = rmo.AddOp(builder, location, x.to_mlir(), y.to_mlir(), params)
            op.verify()
            [result] = op.results
            assert result.type == input_type.to_mlir()


def test_infer_type_op_overload_sets_properties() -> None:
    """The codegen'd state population assigns into `Op::Properties` so
    `inferReturnTypes` reads attributes that are stored as properties
    (not in the discardable attribute dict)."""
    input_type = TensorType(DType.float32, [2, 3], DeviceRef.CPU())
    output_shape = mosh.ShapeAttr([3, 2], mosh.ShapeType())
    with Graph("reshape_props", input_types=[input_type]) as graph:
        with mlir.Location.unknown() as location:
            builder = OpBuilder(Block._from_cmlir(graph._current_block).end)
            (x,) = graph.inputs
            # InferTypeOp overload (no `result=`); the helper must populate
            # `newShape` in `Properties` for inferReturnTypes to read it.
            op = rmo.ReshapeOp(builder, location, x.to_mlir(), output_shape)
            op.verify()
            assert op.new_shape == output_shape
            [result] = op.results
            expected = TensorType(DType.float32, [3, 2], DeviceRef.CPU())
            assert result.type == expected.to_mlir()


def test_infer_type_op_overload_failure_raises_value_error() -> None:
    """A failing `inferReturnTypes` surfaces as a `ValueError` instead of
    aborting the interpreter via `report_fatal_error`."""
    input_type = TensorType(DType.float32, [2, 3], DeviceRef.CPU())
    # 2x3 has 6 elements; reshape to 3x3 (9 elements) is invalid.
    bad_shape = mosh.ShapeAttr([3, 3], mosh.ShapeType())
    with Graph("reshape_bad", input_types=[input_type]) as graph:
        with mlir.Location.unknown() as location:
            builder = OpBuilder(Block._from_cmlir(graph._current_block).end)
            (x,) = graph.inputs
            with pytest.raises(
                ValueError, match=r"infer result type.*rmo\.reshape"
            ):
                rmo.ReshapeOp(builder, location, x.to_mlir(), bad_shape)
