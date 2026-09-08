// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=false" %s | FileCheck %s
// RUN: kgen-opt -elaborate-generators="use-parametric-interpret=true" %s | FileCheck %s

kgen.generator @generic_offset_load_store<ty: type>(%i: index, %p: !kgen.pointer<ty>) {
  %0 = pop.offset %p[%i] : !kgen.pointer<ty>
  %1 = pop.load %0 : !kgen.pointer<ty>
  pop.store %1, %p : !kgen.pointer<ty>
  kgen.return
}

// CHECK-LABEL: @"generic_offset_load_store,ty=scalar<si32>"
// CHECK: pop.offset %{{.*}} : !kgen.pointer<scalar<si32>>

// CHECK-LABEL: @"generic_offset_load_store,ty=simd<4, f32>"
// CHECK: pop.offset %{{.*}} : !kgen.pointer<simd<4, f32>>

kgen.generator @impl(
    %i: index,
    %p0: !kgen.pointer<simd<4, f32>>,
    %p1: !kgen.pointer<scalar<si32>>) {
  kgen.call @generic_offset_load_store<:type !kgen.simd<4, f32>>(%i, %p0) : (index, !kgen.pointer<simd<4, f32>>) -> ()
  kgen.call @generic_offset_load_store<:type !kgen.scalar<si32>>(%i, %p1) : (index, !kgen.pointer<scalar<si32>>) -> ()
  kgen.return
}
