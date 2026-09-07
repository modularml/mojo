// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// Test trivial vector conversions to LLVM.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @trivial_conversions(%a: !kgen.simd<4, f32>, %b: !kgen.simd<4, f32>, %c: !kgen.simd<4, f32>, %d: !kgen.simd<4, bool>) {
  // CHECK: llvm.fneg
  %0 = pop.neg %a : !kgen.simd<4, f32>
  // CHECK: llvm.fadd
  %1 = pop.add %a, %b : !kgen.simd<4, f32>
  // CHECK: llvm.fsub
  %2 = pop.sub %a, %b : !kgen.simd<4, f32>
  // CHECK: llvm.fmul
  %3 = pop.mul %a, %b : !kgen.simd<4, f32>
  // CHECK: llvm.intr.fma
  %4 = pop.fma %a, %b, %c : !kgen.simd<4, f32>
  // CHECK: llvm.intr.fma{{.*}} {fastmathFlags = #llvm.fastmath<nsz>}
  %5 = pop.fma %a, %b, %c {fastmathFlags = #pop.fmf<nsz>} : !kgen.simd<4, f32>
  // CHECK: llvm.select
  %6 = pop.simd.select %d, %a, %b : !kgen.simd<4, f32>
  kgen.return
}

// CHECK-LABEL: @int_neg_simd
kgen.func @int_neg_simd(%arg0: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(dense<0> : vector<4xi32>)
  %0 = pop.neg %arg0 : !kgen.simd<4, si32>
  // CHECK: llvm.sub %[[ZERO]], %{{.*}}
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @neg_simd
kgen.func @neg_simd(%arg0: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  %0 = pop.neg %arg0 : !kgen.simd<4, f32>
  // CHECK: llvm.fneg %{{.*}}
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @floor_simd
kgen.func @floor_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, si32>, %arg2 : !kgen.scalar<index>, %arg3 : !kgen.scalar<uindex>
) -> (
  !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
) {
  // CHECK: llvm.floor
  // CHECK: kgen.return {{.*}}, %arg1, %arg2, %arg3
  %0 = pop.floor %arg0 : !kgen.simd<4, f32>
  %1 = pop.floor %arg1 : !kgen.simd<4, si32>
  %2 = pop.floor %arg2 : !kgen.scalar<index>
  %3 = pop.floor %arg3 : !kgen.scalar<uindex>
  kgen.return %0, %1, %2, %3 : !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @ceil_simd
kgen.func @ceil_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, si32>, %arg2 : !kgen.scalar<index>, %arg3 : !kgen.scalar<uindex>
) -> (
  !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
) {
  // CHECK: llvm.ceil
  // CHECK: kgen.return {{.*}}, %arg1, %arg2, %arg3
  %0 = pop.ceil %arg0 : !kgen.simd<4, f32>
  %1 = pop.ceil %arg1 : !kgen.simd<4, si32>
  %2 = pop.ceil %arg2 : !kgen.scalar<index>
  %3 = pop.ceil %arg3 : !kgen.scalar<uindex>
  kgen.return %0, %1, %2, %3 : !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @trunc_simd
kgen.func @trunc_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, si32>, %arg2 : !kgen.scalar<index>, %arg3 : !kgen.scalar<uindex>
) -> (
  !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
) {
  // CHECK: llvm.trunc
  // CHECK: kgen.return {{.*}}, %arg1, %arg2, %arg3
  %0 = pop.trunc %arg0 : !kgen.simd<4, f32>
  %1 = pop.trunc %arg1 : !kgen.simd<4, si32>
  %2 = pop.trunc %arg2 : !kgen.scalar<index>
  %3 = pop.trunc %arg3 : !kgen.scalar<uindex>
  kgen.return %0, %1, %2, %3 : !kgen.simd<4, f32>, !kgen.simd<4, si32>, !kgen.scalar<index>, !kgen.scalar<uindex>
}

// CHECK-LABEL: @add_simd_si32
kgen.func @add_simd_si32(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @add_simd_index
kgen.func @add_simd_index(%arg0: !kgen.simd<4, index>, %arg1: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg1: !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

// CHECK-LABEL: @fadd_simd
kgen.func @fadd_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @sub_simd
kgen.func @sub_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @fsub_simd
kgen.func @fsub_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @mul_simd
kgen.func @mul_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @fmul_simd
kgen.func @fmul_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @fmul_simd_default_contract
kgen.func @fmul_simd_default_contract(%a: !kgen.simd<4, f32>, %b: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // Default behavior should attach contract fastmath flags.
  // CHECK: llvm.fmul {{.*}}{fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>
  %0 = pop.mul %a, %b : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @fmul_simd_with_reassoc
kgen.func @fmul_simd_with_reassoc(%a: !kgen.simd<4, f32>, %b: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<reassoc>} : vector<4xf32>
  %0 = pop.mul %a, %b {fastmathFlags = #pop.fmf<reassoc>} : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @div_simd
kgen.func @div_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // The operand match keeps this from also accepting `llvm.sdiv exact`, so an
  // unconditionally-forwarded flag would fail here.
  // CHECK: llvm.sdiv {{%.*}}
  %0 = pop.div %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @div_simd_exact
kgen.func @div_simd_exact(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.sdiv exact
  %0 = pop.div %arg0, %arg1 {isExact} : !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @udiv_exact
kgen.func @udiv_exact(%arg0: !kgen.scalar<uindex>, %arg1: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  // CHECK: llvm.udiv exact
  %0 = pop.div %arg0, %arg1 {isExact} : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// CHECK-LABEL: @fdiv_simd
kgen.func @fdiv_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.fdiv
  %0 = pop.div %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @max_simd
kgen.func @max_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.intr.smax
  %0 = pop.max %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @fmax_simd
kgen.func @fmax_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.intr.maxnum
  %0 = pop.max %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @min_simd
kgen.func @min_simd(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.intr.smin
  %0 = pop.min %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @fmin_simd
kgen.func @fmin_simd(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.intr.minnum
  %0 = pop.min %arg0, %arg1: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @shl_simd_si32
kgen.func @shl_simd_si32(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @shl_simd_ui32
kgen.func @shl_simd_ui32(%arg0: !kgen.simd<4, ui32>, %arg1: !kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1: !kgen.simd<4, ui32>
  kgen.return %0 : !kgen.simd<4, ui32>
}

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd_si32(%arg0: !kgen.simd<4, si32>, %arg1: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.ashr
  %0 = pop.shr %arg0, %arg1: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}

// CHECK-LABEL: @shr_simd
kgen.func @shr_simd_ui32(%arg0: !kgen.simd<4, ui32>, %arg1: !kgen.simd<4, ui32>) -> !kgen.simd<4, ui32> {
  // CHECK: llvm.lshr
  %0 = pop.shr %arg0, %arg1: !kgen.simd<4, ui32>
  kgen.return %0 : !kgen.simd<4, ui32>
}

// CHECK-LABEL: @cmp_uint
kgen.func @cmp_uint(%lhs: !kgen.simd<4, ui32>, %rhs: !kgen.simd<4, ui32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !kgen.simd<4, ui32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !kgen.simd<4, ui32>
  // CHECK: llvm.icmp "ult"
  %2 = pop.cmp lt(%lhs, %rhs) : !kgen.simd<4, ui32>
  // CHECK: llvm.icmp "ugt"
  %3 = pop.cmp gt(%lhs, %rhs) : !kgen.simd<4, ui32>
  // CHECK: llvm.icmp "ule"
  %4 = pop.cmp le(%lhs, %rhs) : !kgen.simd<4, ui32>
  // CHECK: llvm.icmp "uge"
  %5 = pop.cmp ge(%lhs, %rhs) : !kgen.simd<4, ui32>
  kgen.return
}

// CHECK-LABEL: @cmp_sint
kgen.func @cmp_sint(%lhs: !kgen.simd<4, si32>, %rhs: !kgen.simd<4, si32>) {
  // CHECK: llvm.icmp "eq"
  %0 = pop.cmp eq(%lhs, %rhs) : !kgen.simd<4, si32>
  // CHECK: llvm.icmp "ne"
  %1 = pop.cmp ne(%lhs, %rhs) : !kgen.simd<4, si32>
  // CHECK: llvm.icmp "slt"
  %2 = pop.cmp lt(%lhs, %rhs) : !kgen.simd<4, si32>
  // CHECK: llvm.icmp "sgt"
  %3 = pop.cmp gt(%lhs, %rhs) : !kgen.simd<4, si32>
  // CHECK: llvm.icmp "sle"
  %4 = pop.cmp le(%lhs, %rhs) : !kgen.simd<4, si32>
  // CHECK: llvm.icmp "sge"
  %5 = pop.cmp ge(%lhs, %rhs) : !kgen.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @cmp_fp
kgen.func @cmp_fp(%lhs: !kgen.simd<4, f32>, %rhs: !kgen.simd<4, f32>) {
  // CHECK: llvm.fcmp "oeq"
  %0 = pop.cmp eq(%lhs, %rhs) : !kgen.simd<4, f32>
  // CHECK: llvm.fcmp "one"
  %1 = pop.cmp ne(%lhs, %rhs) : !kgen.simd<4, f32>
  // CHECK: llvm.fcmp "olt"
  %2 = pop.cmp lt(%lhs, %rhs) : !kgen.simd<4, f32>
  // CHECK: llvm.fcmp "ogt"
  %3 = pop.cmp gt(%lhs, %rhs) : !kgen.simd<4, f32>
  // CHECK: llvm.fcmp "ole"
  %4 = pop.cmp le(%lhs, %rhs) : !kgen.simd<4, f32>
  // CHECK: llvm.fcmp "oge"
  %5 = pop.cmp ge(%lhs, %rhs) : !kgen.simd<4, f32>
  kgen.return
}

// CHECK-LABEL: @fma_simd_si32
kgen.func @fma_simd_si32(%arg0: !kgen.simd<4, si32>,
                    %arg1: !kgen.simd<4, si32>,
                    %arg2: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.mul
  // CHECK: llvm.add
  %0 = pop.fma %arg0, %arg1, %arg2: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}


// CHECK-LABEL: @fma_simd_f32
kgen.func @fma_simd_f32(%arg0: !kgen.simd<4, f32>,
                    %arg1: !kgen.simd<4, f32>,
                    %arg2: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg1, %arg2: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @select_simd_si32
kgen.func @select_simd_si32(%arg0: !kgen.simd<4, bool>,
                    %arg1: !kgen.simd<4, si32>,
                    %arg2: !kgen.simd<4, si32>) -> !kgen.simd<4, si32> {
  // CHECK: llvm.select
  %0 = pop.simd.select %arg0, %arg1, %arg2: !kgen.simd<4, si32>
  kgen.return %0 : !kgen.simd<4, si32>
}


// CHECK-LABEL: @select_simd_f32
kgen.func @select_simd_f32(%arg0: !kgen.simd<4, bool>,
                    %arg1: !kgen.simd<4, f32>,
                    %arg2: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.select
  %0 = pop.simd.select %arg0, %arg1, %arg2: !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @bitcast
kgen.func @bitcast(%a: !kgen.scalar<si32>,
                   %b: !kgen.scalar<ui64>,
                   %c: !kgen.simd<4, f64>,
                   %d: !kgen.simd<2, f64>) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: [[ARG2:%.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK-DAG: [[ARG3:%.*]] = builtin.unrealized_conversion_cast %arg3

  // CHECK: llvm.bitcast [[ARG0]] : i32 to f32
  %0 = pop.bitcast %a: !kgen.scalar<si32> to !kgen.scalar<f32>

  // CHECK: llvm.bitcast [[ARG1]] : i64 to i64
  %1 = pop.bitcast %b: !kgen.scalar<ui64> to !kgen.scalar<si64>

  // CHECK: llvm.bitcast [[ARG2]] : vector<4xf64> to vector<4xi64>
  %2 = pop.bitcast %c: !kgen.simd<4, f64> to !kgen.simd<4, si64>

  // CHECK: llvm.bitcast [[ARG3]] : vector<2xf64> to vector<4xf32>
  %3 = pop.bitcast %d: !kgen.simd<2, f64> to !kgen.simd<4, f32>

  // CHECK: llvm.bitcast [[ARG1]] : i64 to vector<2xf32>
  %4 = pop.bitcast %b: !kgen.scalar<ui64> to !kgen.simd<2, f32>

  // CHECK: %[[B0:.*]] = llvm.bitcast [[ARG1]] : i64 to vector<64xi1>
  %5 = pop.bitcast %b: !kgen.scalar<ui64> to !kgen.simd<64, bool>

  // CHECK: llvm.bitcast %[[B0]] : vector<64xi1> to f64
  %6 = pop.bitcast %5: !kgen.simd<64, bool> to !kgen.simd<1, f64>

  kgen.return
}

// CHECK-LABEL: @simd_splat_scalar_to_2xf32
kgen.func @simd_splat_scalar_to_2xf32(%a: !kgen.scalar<f32>) -> !kgen.simd<2, f32> {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[VECTOR:.*]] = llvm.insertelement %[[E:.*]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xf32>
  // CHECK: %[[RESULT:.*]] = llvm.shufflevector %[[VECTOR]], %[[UNDEF]] [0, 0] : vector<2xf32>
  // CHECK: unrealized_conversion_cast %[[RESULT]]
  %0 = pop.simd.splat %a : !kgen.simd<2, f32>
  kgen.return %0 : !kgen.simd<2, f32>
}

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement(%vec: !kgen.simd<4, f32>, %idx: index) -> !kgen.scalar<f32> {
  // CHECK-DAG: %[[VEC:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: %[[IDX:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK: %[[SCALAR:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[SCALAR]]
  %0 = pop.simd.extractelement %vec[%idx] : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement(%val: !kgen.scalar<f32>, %vec: !kgen.simd<4, f32>, %idx: index) -> !kgen.simd<4, f32> {
  // CHECK-DAG: %[[VAL:.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: %[[VEC:.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: %[[IDX:.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK: %[[RES:.*]] = llvm.insertelement %[[VAL]], %[[VEC]][%[[IDX]] : {{.*}}] : vector<4xf32>
  // CHECK: unrealized_conversion_cast %[[RES]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !kgen.simd<4, f32>
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @simd_insertelement_1xf32
// CHECK-SAME: (%[[ARG0:[[:alnum:]]+]]:
kgen.func @simd_insertelement_1xf32(%val: !kgen.scalar<f32>, %vec: !kgen.scalar<f32>, %idx: index) -> !kgen.scalar<f32> {
  // CHECK-NEXT: kgen.return %[[ARG0]]
  %0 = pop.simd.insertelement %val, %vec[%idx] : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle(%a: !kgen.simd<2, f32>, %b: !kgen.simd<2, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.shufflevector %{{.*}}, %{{.*}} [2, 3, 1, 0]
  %0 = pop.simd.shuffle <2, f32> %a, %b -> <4, f32> :array<4,index> [2, 3, 1, 0]
  kgen.return %0 : !kgen.simd<4, f32>
}

// CHECK-LABEL: @simd_shuffle_1xf32
kgen.func @simd_shuffle_1xf32(%a: !kgen.scalar<f32>, %b: !kgen.scalar<f32>) -> (!kgen.simd<2, f32>, !kgen.scalar<f32>) {
  // CHECK-DAG: %[[F32VAL0:.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<f32> to f32
  // CHECK-DAG: %[[F32VAL1:.*]] = builtin.unrealized_conversion_cast %arg1 : !kgen.scalar<f32> to f32
  // CHECK: %[[VECVAL0_0:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK: %[[CONST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL0_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL0_0]][%[[CONST0_0]] : i32]
  // CHECK: %[[CONST0_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VECVAL0_2:.*]] = llvm.insertelement %[[F32VAL0]], %[[VECVAL0_1]][%[[CONST0_1]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL0_2]] : vector<2xf32> to !kgen.simd<2, f32>
  %0 = pop.simd.shuffle <1, f32> %a, %b -> <2, f32> :array<2,index> [1, 0]

  // CHECK: %[[VECVAL1_0:.*]] = llvm.mlir.undef : vector<1xf32>
  // CHECK: %[[CONST1_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VECVAL1_1:.*]] = llvm.insertelement %[[F32VAL1]], %[[VECVAL1_0]][%[[CONST1_0]] : i32]
  // CHECK: builtin.unrealized_conversion_cast %[[VECVAL1_1]] : vector<1xf32> to !kgen.scalar<f32>
  %1 = pop.simd.shuffle <1, f32> %a, %b -> <1, f32> :array<1,index> [1]

  kgen.return %0, %1 : !kgen.simd<2, f32>, !kgen.scalar<f32>
}

// CHECK-LABEL: @simd_load_store
kgen.func @simd_load_store(%i: index, %p0: !kgen.pointer<simd<4, f32>>) {
  // CHECK: llvm.getelementptr inbounds %{{.*}} : (!llvm.ptr, {{.*}}) -> !llvm.ptr
  %0 = pop.offset %p0[%i] : !kgen.pointer<simd<4, f32>>
  // CHECK: llvm.load
  %1 = pop.load %0 : !kgen.pointer<simd<4, f32>>
  // CHECK: llvm.store
  pop.store %1, %p0 : !kgen.pointer<simd<4, f32>>
  kgen.return
}

// CHECK-LABEL: @abs_simd
kgen.func @abs_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.scalar<uindex>,
  %arg2 : !kgen.scalar<index>, %arg3 : !kgen.scalar<bool>
) -> (
  !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.scalar<index>, !kgen.scalar<bool>
) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG2:%.*]] = builtin.unrealized_conversion_cast %arg2

  // CHECK: [[FABS:%.*]] = llvm.call_intrinsic "llvm.fabs"([[ARG0]])
  // CHECK: [[FABS_RET:%.*]] = builtin.unrealized_conversion_cast [[FABS]]
  %0 = pop.abs %arg0 : !kgen.simd<4, f32>

  // Deleted; replaced with %arg3 itself
  %1 = pop.abs %arg1 : !kgen.scalar<uindex>

  // CHECK: [[IS_INT_MIN_POISON:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK: [[IABS:%.*]] = llvm.call_intrinsic "llvm.abs"([[ARG2]], [[IS_INT_MIN_POISON]])
  // CHECK: [[IABS_RET:%.*]] = builtin.unrealized_conversion_cast [[IABS]]
  %2 = pop.abs %arg2 : !kgen.scalar<index>

  // Deleted; replaced with %arg3 itself
  %3 = pop.abs %arg3 : !kgen.scalar<bool>

  // CHECK: kgen.return [[FABS_RET]], %arg1, [[IABS_RET]], %arg3
  kgen.return %0, %1, %2, %3 : !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.scalar<index>, !kgen.scalar<bool>
}

// CHECK-LABEL: @round_simd
kgen.func @round_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.scalar<uindex>,
  %arg2 : !kgen.scalar<index>, %arg3 : !kgen.scalar<bool>
) -> (
  !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.scalar<index>, !kgen.scalar<bool>
) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0

  // CHECK: [[FROUND:%.*]] = llvm.call_intrinsic "llvm.roundeven"([[ARG0]])
  // CHECK: [[FROUND_RET:%.*]] = builtin.unrealized_conversion_cast [[FROUND]]
  %0 = pop.round %arg0 : !kgen.simd<4, f32>

  // Deleted; replaced with %arg1 itself
  %1 = pop.round %arg1 : !kgen.scalar<uindex>

  // Deleted; replaced with %arg2 itself
  %2 = pop.round %arg2 : !kgen.scalar<index>

  // Deleted; replaced with %arg3 itself
  %3 = pop.round %arg3 : !kgen.scalar<bool>

  // CHECK: kgen.return [[FROUND_RET]], %arg1, %arg2, %arg3
  kgen.return %0, %1, %2, %3 : !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.scalar<index>, !kgen.scalar<bool>
}

// CHECK-LABEL: @floordiv_simd
kgen.func @floordiv_simd(
  %arg0: !kgen.simd<4, f32>, %arg1 : !kgen.simd<4, f32>,
  %arg2 : !kgen.scalar<uindex>, %arg3 : !kgen.scalar<uindex>,
  %arg4 : !kgen.simd<2, index>, %arg5 : !kgen.simd<2, index>
) -> (
  !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.simd<2, index>
) {
  // CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0
  // CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %arg1
  // CHECK-DAG: [[ARG2:%.*]] = builtin.unrealized_conversion_cast %arg2
  // CHECK-DAG: [[ARG3:%.*]] = builtin.unrealized_conversion_cast %arg3
  // CHECK-DAG: [[ARG4:%.*]] = builtin.unrealized_conversion_cast %arg4
  // CHECK-DAG: [[ARG5:%.*]] = builtin.unrealized_conversion_cast %arg5

  // CHECK: [[FDIV:%.*]] = llvm.fdiv [[ARG0]], [[ARG1]] : vector<4xf32>
  // CHECK: llvm.call_intrinsic "llvm.floor"([[FDIV]])
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<4, f32>

  // CHECK: llvm.udiv [[ARG2]], [[ARG3]] : i64
  %1 = pop.floordiv %arg2, %arg3 : !kgen.scalar<uindex>

  // CHECK:      [[SDIV:%.*]] = llvm.sdiv [[ARG4]], [[ARG5]]
  // CHECK-NEXT: [[MUL:%.*]] = llvm.mul [[SDIV]], [[ARG5]] : vector<2xi64>
  // CHECK-NEXT: [[CMP:%.*]] = llvm.icmp "eq" [[MUL]], [[ARG4]] : vector<2xi64>
  // CHECK-NEXT: [[XOR:%.*]] = llvm.xor [[ARG4]], [[ARG5]] : vector<2xi64>
  // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(#M.dense_array<0, 0> : vector<2xi64>) : vector<2xi64>
  // CHECK-NEXT: [[BSHIFT:%.*]] = llvm.mlir.constant(#M.dense_array<63, 63> : vector<2xi64>) : vector<2xi64>
  // CHECK-NEXT: [[ASHR:%.*]] = llvm.ashr [[XOR]], [[BSHIFT]] : vector<2xi64>
  // CHECK-NEXT: [[SEL:%.*]] = llvm.select [[CMP]], [[ZERO]], [[ASHR]] : vector<2xi1>, vector<2xi64>
  // CHECK-NEXT: llvm.add [[SDIV]], [[SEL]] : vector<2xi64>
  %2 = pop.floordiv %arg4, %arg5 : !kgen.simd<2, index>

  kgen.return %0, %1, %2 : !kgen.simd<4, f32>, !kgen.scalar<uindex>, !kgen.simd<2, index>
}

}
