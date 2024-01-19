# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable
struct __ParameterClosureCaptureList[fn_type: AnyRegType, fn_ref: fn_type]:
    var value: __mlir_type.`!kgen.pointer<none>`

    # Parameter closure invariant requires this function be marked 'capturing'.
    @closure
    @always_inline
    fn __init__() -> Self:
        return Self {
            value: __mlir_op.`kgen.capture_list.create`[callee=fn_ref]()
        }

    @always_inline
    fn __copyinit__(existing: Self) -> Self:
        return Self {
            value: __mlir_op.`kgen.capture_list.copy`[callee=fn_ref](
                existing.value
            )
        }

    @always_inline
    fn __del__(owned self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    fn expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)
