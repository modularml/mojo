// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// Test that IREvaluatorContext::printParamValue (via get_type_name) correctly
// formats SIMD-typed struct parameters for all supported dtypes.  This
// exercises all branches of POP::printDTypeValue in a single elaboration pass.
//
// Related: MOCO-3651.

kgen.struct.generator @SIMDParamStruct<
    i: !kgen.scalar<si32>,
    f: !kgen.scalar<f32>,
    b: !kgen.scalar<bool>,
    x: !kgen.scalar<index>,
    u: !kgen.scalar<uindex>,
    v: !kgen.simd<4, si32>
> = struct_inst<"SIMDParamStruct"> {}

// CHECK-LABEL: kgen.func export @test_simd_param_printing
kgen.generator export @test_simd_param_printing() {
  // CHECK-NEXT: constant: string = <"SIMDParamStruct[42 : SIMD[DType.int32, 1], 1.5 : SIMD[DType.float32, 1], True, 7 : SIMD[DType.int, 1], 8 : SIMD[DType.uint, 1], [1, 2, 3, 4] : SIMD[DType.int32, 4]]">
  kgen.param.constant: string = <#kgen.get_type_name<
    #kgen.genref<@SIMDParamStruct<
      :!kgen.scalar<si32> #kgen<simd 42>,
      :!kgen.scalar<f32> #kgen<simd "1.5">,
      :!kgen.scalar<bool> #kgen<simd true>,
      :!kgen.scalar<index> #kgen<simd 7>,
      :!kgen.scalar<uindex> #kgen<simd 8>,
      :!kgen.simd<4, si32> #kgen<simd<1, 2, 3, 4>>>>,
    #kgen.simd<false>:!kgen.scalar<bool>>>
  kgen.return
}
