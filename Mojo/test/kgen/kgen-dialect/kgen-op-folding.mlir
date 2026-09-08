// RUN: kgen-opt -canonicalize -mlir-print-debuginfo %s | FileCheck %s

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() ->
  (!pop.float_literal, !pop.float_literal) {
  %il1 = kgen.param.constant: !pop.float_literal = <#pop<int_to_float_literal<5>>>
  // sugar_alias wraps IntLiteralAttr{42} (sugared) over IntLiteralAttr{5} (expanded).
  %il2 = kgen.param.constant: !pop.float_literal = <#pop<int_to_float_literal<sugar_alias(42, 5)>>>
  // CHECK: #pop.float_literal<5|1>
  kgen.return %il1, %il2 : !pop.float_literal, !pop.float_literal
}


// CHECK-LABEL: @float_literal_isa
kgen.func @float_literal_isa() -> (
  i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_zero #pop.float_literal<0|1>>>>

  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_zero #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<neg_zero>>>>

  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_isa<inf #pop.float_literal<inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_inf #pop.float_literal<neg_inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b8 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<neg_inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#pop<float_literal_isa<nan #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b10 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<nan>>>>
  // sugar_alias wraps FloatLiteralAttr{42} (sugared) over FloatLiteralAttr{0/1} (expanded).
  // float_literal_isa<normal> of a normal zero value returns true.
  // CHECK-SAME: [[TRUE]]
  %b11 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<0|1>)>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9, %b10, %b11
    : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_normal_diff
kgen.func @float_literal_cmp_normal_diff() -> (i1, i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Test normal cases with different normal numbers
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // sugar_alias wraps FloatLiteralAttr{42} (sugared) over FloatLiteralAttr{5/3} and {8/3} (expanded).
  // 5/3 < 8/3 is true.
  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<5|3>), sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<8|3>)>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7 : i1, i1, i1, i1, i1, i1,
    i1
}

// CHECK-LABEL: @float_literal_cmp_normal_same
kgen.func @float_literal_cmp_normal_same() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test normal cases with the same normal number
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_neg_zero
kgen.func @float_literal_cmp_neg_zero() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test negative zero cases
  // Note that in Python, -0 = 0
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_zero>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<0|1>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<0|1>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b8 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<-5|3>, #pop.float_literal<neg_zero>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9:
    i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_inf
kgen.func @float_literal_cmp_inf() -> (i1, i1, i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>

  // CHECK: kgen.return
  // Some infinity cases
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<inf>, #pop.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_inf>, #pop.float_literal<neg_inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<neg_inf>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b6= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b7= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b8= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8:
    i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_nan
kgen.func @float_literal_cmp_nan() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Some NAN cases
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<nan>, #pop.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_binop_nan
kgen.func @float_literal_binop_nan() -> (
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal
  ) {
  // CHECK: [[NAN:%.*]] = kgen.param.constant: !pop.float_literal = <#pop.float_literal<nan>>
  // CHECK: kgen.return

  // Nan always results in Nan
  // CHECK-SAME: [[NAN]]
  %r1 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r2 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r3 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r4 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r5 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r6 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r7 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r8 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r9 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r10 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<nan>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10
  :
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal
}

// CHECK-LABEL: @float_literal_binop_uniques
kgen.func @float_literal_binop_uniques() ->
  (!pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal) {

  // CHECK: <0|1>
  %r1 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <-25|9>
  %r2 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <-1|1>
  %r3 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <10|3>
  %r4 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // sugar_alias wraps FloatLiteralAttr{42} (sugared) over FloatLiteralAttr{5/3} and {8/3} (expanded).
  // 5/3 + 8/3 = 13/3
  // CHECK: <13|3>
  %r5 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<5|3>), sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<8|3>)>>>

  kgen.return %r1, %r2, %r3, %r4, %r5 :
    !pop.float_literal, !pop.float_literal, !pop.float_literal,
    !pop.float_literal, !pop.float_literal
}

// CHECK-LABEL: @float_literal_convert
kgen.func @float_literal_convert()
  -> (!kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>) {
  // CHECK: kgen.param.constant: scalar<f64> = <"1.666666{{.*}}">
  %r1 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-1.666666{{.*}}">
  %r2 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"0">
  %r3 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<0|1>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-0">
  %r4 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<neg_zero>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"+Inf">
  %r5 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<inf>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-Inf">
  %r6 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<neg_inf>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"NaN">
  %r7 = kgen.param.constant: !kgen.scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<nan>>>>
  // sugar_alias wraps a dummy float literal over 5/3; convert should use the expanded literal.
  %r8 = kgen.param.constant: !kgen.scalar<f64> = <#pop<float_literal_convert<sugar_alias(#pop.float_literal<99|1>, #pop.float_literal<5|3>)>>>
  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8
    : !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>
}

// CHECK-LABEL: @float_e4m3_literal_convert
kgen.func @float_e4m3_literal_convert()
  -> (!kgen.scalar<f8e4m3fn>, !kgen.scalar<f8e4m3fn>, !kgen.scalar<f8e4m3fn>) {
  // CHECK: kgen.param.constant: scalar<f8e4m3fn> = <"0.75">
  %r1 = kgen.param.constant: scalar<f8e4m3fn> = <#pop<float_literal_convert<#pop.float_literal<3|4>>>>
  // CHECK: kgen.param.constant: scalar<f8e4m3fn> = <"1">
  %r2 = kgen.param.constant: scalar<f8e4m3fn> = <#pop<float_literal_convert<#pop.float_literal<1|1>>>>
  // CHECK: kgen.param.constant: scalar<f8e4m3fn> = <"0.00195">
  %r3 = kgen.param.constant: scalar<f8e4m3fn> = <#pop<float_literal_convert<#pop.float_literal<1|512>>>>
  kgen.return %r1, %r2, %r3
    : !kgen.scalar<f8e4m3fn>, !kgen.scalar<f8e4m3fn>, !kgen.scalar<f8e4m3fn>
}

// CHECK-LABEL: @float_literal_to_int_literal
kgen.func @float_literal_to_int_literal() ->
  (!pop.int_literal, !pop.int_literal, !pop.int_literal, !pop.int_literal,
   !pop.int_literal, !pop.int_literal) {
  // CHECK: kgen.param.constant: !pop.int_literal = <1>
  %r1 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <2>
  %r2 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<8|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <-1>
  %r3 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <-2>
  %r4 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<-8|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <0>
  %r5 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<neg_zero>>>>
  // sugar_alias wraps FloatLiteralAttr{42} (sugared) over FloatLiteralAttr{5/3} (expanded).
  %r6 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<sugar_alias(#pop.float_literal<42|1>, #pop.float_literal<5|3>)>>>
  kgen.return %r1, %r2, %r3, %r4, %r5, %r6 : !pop.int_literal,
    !pop.int_literal, !pop.int_literal, !pop.int_literal, !pop.int_literal,
    !pop.int_literal
}

// CHECK-LABEL: @int_literal_bin_sugar
kgen.func @int_literal_bin_sugar() -> !pop.int_literal {
  // SugarAttr wrapping IntLiteralAttr{42} (sugared) over IntLiteralAttr{3} and {5} (expanded).
  // 3 + 5 = 8
  %r = kgen.param.constant: !pop.int_literal = <#pop<int_literal_bin<add #kgen<sugar alias, !pop.int_literal, 42, 3>, #kgen<sugar alias, !pop.int_literal, 42, 5>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <8>
  kgen.return %r : !pop.int_literal
}

// CHECK-LABEL: @int_literal_cmp_sugar
kgen.func @int_literal_cmp_sugar() -> i1 {
  // sugar_alias wraps IntLiteralAttr{42} (sugared) over IntLiteralAttr{3} and {5} (expanded).
  // 3 < 5 is true.
  %r = kgen.param.constant: i1 = <#pop<int_literal_cmp<lt sugar_alias(42, 3), sugar_alias(42, 5)>>>
  // CHECK: kgen.param.constant: i1 = <1>
  kgen.return %r : i1
}

// CHECK-LABEL: @cast_to_builtin_sugar_identity
kgen.func @cast_to_builtin_sugar_identity() -> f32 {
  // sugarDynCastIfPresent peels SugarAttr so cast_to_builtin folds through cast_from_builtin.
  // Use #kgen<sugar ...> (not sugar_alias(...)) so parsing stays valid inside cast_to_builtin<...>.
  %r = kgen.param.constant: f32 = <#kgen.cast_to_builtin<#kgen<sugar alias, !kgen.scalar<f32>, *?, #kgen.cast_from_builtin<2.5 : f32>>>>
  // CHECK: kgen.param.constant: f32 = <2.500000e+00>
  kgen.return %r : f32
}

// CHECK-LABEL: @cast_from_builtin_sugar_identity
kgen.generator @cast_from_builtin_sugar_identity<simd_param: !kgen.scalar<si32>>() -> !kgen.scalar<si32> {
  %r = kgen.param.constant: !kgen.scalar<si32> = <#kgen.cast_from_builtin<#kgen<sugar alias, si32, *?, #kgen.cast_to_builtin< #kgen.param.decl.ref<"simd_param"> : !kgen.scalar<si32> >>>>
  // CHECK: kgen.param.constant: scalar<si32> = <sugar_preserved(from_builtin(:si32 *?), simd_param)> loc(#loc55)
  kgen.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: @simd_cast_sugar_fold
kgen.func @simd_cast_sugar_fold() -> !kgen.scalar<ui32> {
  // sugarDynCastIfPresent sees the inner scalar SIMD for #pop.cast folding.
  %r = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<sugar alias, !kgen.scalar<si32>, *?, #kgen<simd 1>>>>
  // CHECK: kgen.param.constant: scalar<ui32> = <1>
  kgen.return %r : !kgen.scalar<ui32>
}

// CHECK-LABEL: @simd_splat_sugar_fold
kgen.func @simd_splat_sugar_fold() -> !kgen.simd<3, f16> {
  %r = kgen.param.constant: !kgen.simd<3, f16> = <#kgen.simd_splat<#kgen<sugar alias, !kgen.scalar<f16>, *?, #kgen<simd "1.0">>>>
  // CHECK: kgen.param.constant: simd<3, f16> = <"1">
  kgen.return %r : !kgen.simd<3, f16>
}
