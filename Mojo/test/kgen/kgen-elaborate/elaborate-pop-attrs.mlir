// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// #kgen.param.expr<lt> with index: target-dependent folding on 32-bit.
// 3000000000 wraps to negative in 32-bit signed, so lt(3000000000, 0) is true.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_32
kgen.generator export @test_simd_cmp_index_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.expr<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// #kgen.param.expr<lt> with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so lt(3000000000, 0) is false.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_64
kgen.generator export @test_simd_cmp_index_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.expr<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// #pop.simd_shl with index: target-dependent folding on 64-bit.
// shl(1, 33) is poison on 32-bit (shift >= 32), so it can't fold without
// target. With 64-bit target, shl(1, 33) = 8589934592.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shl_index_64
kgen.generator export @test_simd_shl_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shl<#kgen<simd 1> : !kgen.scalar<index>, #kgen<simd 33> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <8589934592>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so ashr(-1294967296, 1) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_32
kgen.generator export @test_simd_shr_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shr<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 1> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so shr(3000000000, 1) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_64
kgen.generator export @test_simd_shr_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shr<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 1> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so abs(-1294967296) = 1294967296.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_32
kgen.generator export @test_simd_abs_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_abs<#kgen<simd 3000000000> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1294967296>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so abs(3000000000) = 3000000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_64
kgen.generator export @test_simd_abs_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_abs<#kgen<simd 3000000000> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <3000000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #kgen.param.expr<div> with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed,
// so div(-1294967296, 2) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_div_index_32
kgen.generator export @test_simd_div_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#kgen.param.expr<div,#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #kgen.param.expr<div> with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so div(3000000000, 2) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_div_index_64
kgen.generator export @test_simd_div_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#kgen.param.expr<div,#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #kgen.param.expr<floor_div_s> with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed,
// so floordiv(-1294967296, 2) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_floordiv_index_32
kgen.generator export @test_simd_floordiv_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#kgen.param.expr<floor_div_s,#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #kgen.param.expr<floor_div_s> with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so floordiv(3000000000, 2) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_floordiv_index_64
kgen.generator export @test_simd_floordiv_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#kgen.param.expr<floor_div_s,#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #kgen.param.identical with index: identity asks whether two values are the
// *same value*, which for `index` depends on the target just as `eq` does --
// 2^32 truncates to 0 at 32-bit width. So it defers at construction and settles
// here, and must agree with `eq`.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_index_32
kgen.generator export @test_identical_index_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// The same operands at 64-bit width are different values.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_index_64
kgen.generator export @test_identical_index_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// Index data nested inside an aggregate defers at construction (see
// @param_identical_nested_index in kgen-dialect/kgen-param-exprs.mlir) and
// settles here, where the index width is known.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_index_agg_32
kgen.generator export @test_identical_index_agg_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.param_list<#kgen<simd 4294967296> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>, #kgen.param_list<#kgen<simd 0> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// The same aggregates at 64-bit width hold different values.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_index_agg_64
kgen.generator export @test_identical_index_agg_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.param_list<#kgen<simd 4294967296> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>, #kgen.param_list<#kgen<simd 0> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// `index` is signed: -1 and 4294967295 are the same value at 32-bit width and
// different at 64-bit, which is the only case that exercises sign-extension in
// the nested comparison.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_negative_index_32
kgen.generator export @test_identical_negative_index_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.param_list<#kgen<simd -1> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>, #kgen.param_list<#kgen<simd 4294967295> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_negative_index_64
kgen.generator export @test_identical_negative_index_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.param_list<#kgen<simd -1> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>, #kgen.param_list<#kgen<simd 4294967295> : !kgen.scalar<index>> : !kgen.param_list<!kgen.scalar<index>>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// An n-ary identity settles here too. 2^32, 0 and 2^33 all truncate to 0, so
// every pair is ambiguous without a target and all three operands survive
// construction -- which is what makes this n-ary rather than a leftover pair.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_nary_index_32
kgen.generator export @test_identical_nary_index_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>, #kgen<simd 8589934592> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// At 64-bit width one pair is enough to settle the class as different values.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_nary_index_64
kgen.generator export @test_identical_nary_index_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>, #kgen<simd 8589934592> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// An unknown carries no value, so it never settles a pair. Two other operands
// that differ at the target's width still settle the class around it: 2^32 and
// 2^33 both truncate to 0 at 32 bits, so this is a `false` only the target
// reaches, and only from a pair neither of whose members is the unknown.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_nary_unknown_settled
kgen.generator export @test_identical_nary_unknown_settled() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.unknown : !kgen.scalar<index>, #kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 8589934592> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// The same around an operand that is not fully evaluated rather than unknown --
// a division by zero, which never folds. Neither has a value for a key to stand
// for, so both sit out the partition. Without that the class would stay
// residual and the unfoldable operand would reach the elaborator.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_identical_nary_residual_settled
kgen.generator export @test_identical_nary_residual_settled() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#kgen.param.identical<#kgen.param.expr<div, #kgen<simd 1> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<index>, #kgen<simd 4294967296> : !kgen.scalar<index>, #kgen<simd 8589934592> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}
