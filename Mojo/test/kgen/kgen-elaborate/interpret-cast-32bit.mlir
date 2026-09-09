// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -o - | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -o - | FileCheck %s

// COM: Compilation should succeed.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:32:32">} {
  kgen.generator @cast_index_to_si64_large() -> !kgen.scalar<si64> {
    %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
    %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si64>
    kgen.return %1 : !kgen.scalar<si64>
  }

  kgen.generator @cast_index_to_si32_large() -> !kgen.scalar<si32> {
    %0 = kgen.param.constant: simd<1, index> = <-8664705627211539068>
    %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si32>
    kgen.return %1 : !kgen.scalar<si32>
  }

  // CHECK-LABEL: @main
  kgen.generator export @main() {
    kgen.param.apply x = [() -> !kgen.scalar<si64> : @cast_index_to_si64_large]()
    kgen.param.apply y = [() -> !kgen.scalar<si32> : @cast_index_to_si32_large]()
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <-1095082620>
    %0 = kgen.param.constant: !kgen.scalar<si64> = <x>
    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1095082620>
    %1 = kgen.param.constant: !kgen.scalar<si32> = <y>

    // CHECK-NEXT: kgen.param.constant: simd<4, si64> = <-1>
    %2 = kgen.param.constant: !kgen.simd<4, si64> = <#kgen.simd_splat<#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>: !kgen.simd<4, si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <18446744073709551615>
    %3 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <2147483647>
    %4 = kgen.param.constant: !kgen.scalar<si64> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <2147483647>
    %5 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <-1>
    %6 = kgen.param.constant: !kgen.scalar<si64> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <18446744073709551615>
    %7 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>

    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1>
    %8 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <4294967295>
    %9 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>
    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <2147483647>
    %10 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <2147483647>
    %11 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>
    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1>
    %12 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <4294967295>
    %13 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>

    // CHECK-NEXT: kgen.param.constant: scalar<f64> = <"0">
    %14 = kgen.param.constant: !kgen.scalar<f64> = <#pop.cast<#kgen<simd 9007199254740992> : !kgen.scalar<index>> : !kgen.scalar<f64>>
    // CHECK-NEXT: kgen.param.constant: scalar<f32> = <"0">
    %15 = kgen.param.constant: !kgen.scalar<f32> = <#pop.cast<#kgen<simd 9007199254740992> : !kgen.scalar<index>> : !kgen.scalar<f32>>

    // CHECK-NEXT: kgen.param.constant: scalar<si128> = <0>
    %16 = kgen.param.constant : !kgen.scalar<si128> = <#pop.cast<#kgen<simd -9223372036854775808> : !kgen.scalar<index>> : !kgen.scalar<si128>>
    // CHECK-NEXT: kgen.param.constant: scalar<si128> = <-2147483648>
    %17 = kgen.param.constant : !kgen.scalar<si128> = <#pop.cast<#kgen<simd -2147483648> : !kgen.scalar<index>> : !kgen.scalar<si128>>

    kgen.return
  }
}
