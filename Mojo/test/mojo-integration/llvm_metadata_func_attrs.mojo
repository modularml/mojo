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
# RUN: kgen -emit=llvm %s -o - | FileCheck %s

# Strings need `__mlir_attr.`"..."``: bare `"..."` is a StringLiteral struct,
# not a StringAttr.


# ===-------------------------------------------------------------------=== #
# Unit-valued attributes (flags).
# ===-------------------------------------------------------------------=== #


# CHECK-LABEL: define{{.*}} @{{.*}}fn_always_inline
# CHECK-SAME: #[[AI:[0-9]+]]
@export
@__llvm_metadata(`llvm.always_inline`)
def fn_always_inline() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_no_inline
# CHECK-SAME: #[[NI:[0-9]+]]
@export
@__llvm_metadata(`llvm.no_inline`)
def fn_no_inline() abi("Mojo"):
    pass


# LLVMFuncOp's verifier requires `optimize_none` to come with `no_inline`.
# CHECK-LABEL: define{{.*}} @{{.*}}fn_optimize_none
# CHECK-SAME: #[[ON:[0-9]+]]
@export
@__llvm_metadata(`llvm.optimize_none`)
@__llvm_metadata(`llvm.no_inline`)
def fn_optimize_none() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_no_unwind
# CHECK-SAME: #[[NU:[0-9]+]]
@export
@__llvm_metadata(`llvm.no_unwind`)
def fn_no_unwind() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_will_return
# CHECK-SAME: #[[WR:[0-9]+]]
@export
@__llvm_metadata(`llvm.will_return`)
def fn_will_return() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_convergent
# CHECK-SAME: #[[CV:[0-9]+]]
@export
@__llvm_metadata(`llvm.convergent`)
def fn_convergent() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_cold
# CHECK-SAME: #[[CO:[0-9]+]]
@export
@__llvm_metadata(`llvm.cold`)
def fn_cold() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_hot
# CHECK-SAME: #[[HT:[0-9]+]]
@export
@__llvm_metadata(`llvm.hot`)
def fn_hot() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_inline_hint
# CHECK-SAME: #[[IH:[0-9]+]]
@export
@__llvm_metadata(`llvm.inline_hint`)
def fn_inline_hint() abi("Mojo"):
    pass


# OptionalAttr<UnitAttr> properties on LLVMFuncOp: optsize, minsize,
# save_reg_params.
# CHECK-LABEL: define{{.*}} @{{.*}}fn_optsize
# CHECK-SAME: #[[OS:[0-9]+]]
@export
@__llvm_metadata(`llvm.optsize`)
def fn_optsize() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_minsize
# CHECK-SAME: #[[MS:[0-9]+]]
@export
@__llvm_metadata(`llvm.minsize`)
def fn_minsize() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_save_reg_params
# CHECK-SAME: #[[SRP:[0-9]+]]
@export
@__llvm_metadata(`llvm.save_reg_params`)
def fn_save_reg_params() abi("Mojo"):
    pass


# ===-------------------------------------------------------------------=== #
# String-valued attributes.
# ===-------------------------------------------------------------------=== #
#
# `target_cpu`, `tune_cpu`, and `target_features` are deliberately rejected
# (see llvm_metadata_func_attrs_errors.mojo).


# CHECK-LABEL: define{{.*}} @{{.*}}fn_denorm_fp
# CHECK-SAME: #[[DFP:[0-9]+]]
@export
@__llvm_metadata(
    `llvm.denormal_fp_math`=__mlir_attr.`"preserve-sign,preserve-sign"`
)
def fn_denorm_fp():
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_denorm_fp32
# CHECK-SAME: #[[DFP32:[0-9]+]]
@export
@__llvm_metadata(`llvm.denormal_fp_math_f32`=__mlir_attr.`"ieee,ieee"`)
def fn_denorm_fp32() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_fp_contract
# CHECK-SAME: #[[FPC:[0-9]+]]
@export
@__llvm_metadata(`llvm.fp_contract`=__mlir_attr.`"fast"`)
def fn_fp_contract() abi("Mojo"):
    pass


# String/bool/integer-valued attrs whose LLVM IR names use dashes.
# CHECK-LABEL: define{{.*}} @{{.*}}fn_probe_stack
# CHECK-SAME: #[[PS:[0-9]+]]
@export
@__llvm_metadata(`llvm.probe_stack`=__mlir_attr.`"__chkstk"`)
def fn_probe_stack() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_no_inline_line_tables
# CHECK-SAME: #[[NILT:[0-9]+]]
@export
@__llvm_metadata(`llvm.no_inline_line_tables`=__mlir_attr.true)
def fn_no_inline_line_tables() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_stack_probe_size
# CHECK-SAME: #[[SPS:[0-9]+]]
@export
@__llvm_metadata(`llvm.stack_probe_size`=SIMDLength(4096))
def fn_stack_probe_size() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_warn_stack_size
# CHECK-SAME: #[[WSS:[0-9]+]]
@export
@__llvm_metadata(`llvm.warn_stack_size`=SIMDLength(1024))
def fn_warn_stack_size() abi("Mojo"):
    pass


# ===-------------------------------------------------------------------=== #
# Integer-valued attributes.
# ===-------------------------------------------------------------------=== #


# `align <N>` is printed after the attribute id, e.g. `... #6 align 32 {`.
# CHECK-LABEL: define{{.*}} @{{.*}}fn_alignment
# CHECK-SAME: align 32
@export
@__llvm_metadata(`llvm.alignment`=SIMDLength(32))
def fn_alignment() abi("Mojo"):
    pass


# function_entry_count lands as a `!prof` annotation, not in `attributes #N`.
# CHECK-LABEL: define{{.*}} @{{.*}}fn_entry_count
# CHECK-SAME: !prof
@export
@__llvm_metadata(`llvm.function_entry_count`=SIMDLength(1234))
def fn_entry_count() abi("Mojo"):
    pass


# ===-------------------------------------------------------------------=== #
# Boolean-valued attributes (fast-math flags).
# ===-------------------------------------------------------------------=== #


# CHECK-LABEL: define{{.*}} @{{.*}}fn_fast_math
# CHECK-SAME: #[[FM:[0-9]+]]
@export
@__llvm_metadata(`llvm.unsafe_fp_math`=__mlir_attr.true)
@__llvm_metadata(`llvm.no_infs_fp_math`=__mlir_attr.true)
@__llvm_metadata(`llvm.no_nans_fp_math`=__mlir_attr.true)
@__llvm_metadata(`llvm.approx_func_fp_math`=__mlir_attr.true)
@__llvm_metadata(`llvm.no_signed_zeros_fp_math`=__mlir_attr.true)
def fn_fast_math() abi("Mojo"):
    pass


# ===-------------------------------------------------------------------=== #
# Composite typed attributes.
# ===-------------------------------------------------------------------=== #


# CHECK-LABEL: define{{.*}} @{{.*}}fn_frame_pointer
# CHECK-SAME: #[[FP:[0-9]+]]
@export
@__llvm_metadata(`llvm.frame_pointer`=__mlir_attr.`"all"`)
def fn_frame_pointer() abi("Mojo"):
    pass


# CHECK-LABEL: define{{.*}} @{{.*}}fn_vscale_range
# CHECK-SAME: #[[VS:[0-9]+]]
@export
@__llvm_metadata(
    `llvm.vscale_range`=__mlir_attr.`#pop.array<1, 16> : !pop.array<2, i32>`
)
def fn_vscale_range():
    pass


# ===-------------------------------------------------------------------=== #
# Unknown llvm.* names fall back to passthrough (existing behavior).
# ===-------------------------------------------------------------------=== #


# CHECK-LABEL: define{{.*}} @{{.*}}fn_passthrough
# CHECK-SAME: #[[PT:[0-9]+]]
@export
@__llvm_metadata(`llvm.noredzone`)
def fn_passthrough() abi("Mojo"):
    pass


# Keep every function above live.
@export
def use_all() abi("Mojo"):
    fn_always_inline()
    fn_no_inline()
    fn_optimize_none()
    fn_no_unwind()
    fn_will_return()
    fn_convergent()
    fn_cold()
    fn_hot()
    fn_inline_hint()
    fn_optsize()
    fn_minsize()
    fn_save_reg_params()
    fn_denorm_fp()
    fn_denorm_fp32()
    fn_fp_contract()
    fn_probe_stack()
    fn_no_inline_line_tables()
    fn_stack_probe_size()
    fn_warn_stack_size()
    fn_alignment()
    fn_entry_count()
    fn_fast_math()
    fn_frame_pointer()
    fn_vscale_range()
    fn_passthrough()


# CHECK-DAG: attributes #[[AI]] = { {{.*}}alwaysinline{{.*}} }
# CHECK-DAG: attributes #[[NI]] = { {{.*}}noinline{{.*}} }
# CHECK-DAG: attributes #[[ON]] = { {{.*}}optnone{{.*}} }
# CHECK-DAG: attributes #[[NU]] = { {{.*}}nounwind{{.*}} }
# CHECK-DAG: attributes #[[WR]] = { {{.*}}willreturn{{.*}} }
# CHECK-DAG: attributes #[[CV]] = { {{.*}}convergent{{.*}} }
# CHECK-DAG: attributes #[[CO]] = { {{.*}}cold{{.*}} }
# CHECK-DAG: attributes #[[HT]] = { {{.*}}hot{{.*}} }
# CHECK-DAG: attributes #[[IH]] = { {{.*}}inlinehint{{.*}} }
# CHECK-DAG: attributes #[[OS]] = { {{.*}}optsize{{.*}} }
# CHECK-DAG: attributes #[[MS]] = { {{.*}}minsize{{.*}} }
# CHECK-DAG: attributes #[[SRP]] = { {{.*}}"save-reg-params"{{.*}} }
# CHECK-DAG: attributes #[[DFP]] = { {{.*}}"denormal-fp-math"="preserve-sign,preserve-sign"{{.*}} }
# CHECK-DAG: attributes #[[DFP32]] = { {{.*}}"denormal-fp-math-f32"="ieee,ieee"{{.*}} }
# CHECK-DAG: attributes #[[FPC]] = { {{.*}}"fp-contract"="fast"{{.*}} }
# CHECK-DAG: attributes #[[PS]] = { {{.*}}"probe-stack"="__chkstk"{{.*}} }
# CHECK-DAG: attributes #[[NILT]] = { {{.*}}"no-inline-line-tables"="true"{{.*}} }
# CHECK-DAG: attributes #[[SPS]] = { {{.*}}"stack-probe-size"="4096"{{.*}} }
# CHECK-DAG: attributes #[[WSS]] = { {{.*}}"warn-stack-size"="1024"{{.*}} }
# CHECK-DAG: attributes #[[FM]] = { {{.*}}"approx-func-fp-math"="true"{{.*}}"no-infs-fp-math"="true"{{.*}}"no-nans-fp-math"="true"{{.*}}"no-signed-zeros-fp-math"="true"{{.*}}"unsafe-fp-math"="true"{{.*}} }
# CHECK-DAG: attributes #[[FP]] = { {{.*}}"frame-pointer"="all"{{.*}} }
# CHECK-DAG: attributes #[[VS]] = { {{.*}}vscale_range(1,16){{.*}} }
# CHECK-DAG: attributes #[[PT]] = { {{.*}}noredzone{{.*}} }
