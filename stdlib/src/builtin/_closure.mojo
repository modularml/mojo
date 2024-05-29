# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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


@register_passable
struct __ParameterClosureCaptureList[
    fn_type: AnyTrivialRegType, fn_ref: fn_type
]:
    var value: __mlir_type.`!kgen.pointer<none>`

    # Parameter closure invariant requires this function be marked 'capturing'.
    @parameter
    @always_inline
    fn __init__(inout self):
        self.value = __mlir_op.`kgen.capture_list.create`[callee=fn_ref]()

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        self.value = __mlir_op.`kgen.capture_list.copy`[callee=fn_ref](
            existing.value
        )

    @always_inline
    fn __del__(owned self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    fn expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)


fn __closure_wrapper_noop_dtor(
    owned self: __mlir_type.`!kgen.pointer<none>`, /
):
    pass


fn __closure_wrapper_noop_copy(
    owned other: __mlir_type.`!kgen.pointer<none>`, /
) -> __mlir_type.`!kgen.pointer<none>`:
    return other
