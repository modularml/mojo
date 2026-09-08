// RUN: kgen-opt %s -allow-unregistered-dialect -split-input-file -verify-parameters | FileCheck %s

// CHECK-LABEL: kgen.generator @parameterIsolatedRegions
kgen.generator @parameterIsolatedRegions<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region Fn = <B>() {
    kgen.param.constant = <B>
    kgen.return
  }
  // CHECK: {isolated}

  // CHECK: kgen.param.if
  kgen.param.if <lt(A, 1)> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK: {elseIsolated, thenIsolated}
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @struct_of_simd
// CHECK-SAME: -> !kgen.struct<(simd<size, type>)>
kgen.generator @struct_of_simd<size, type: dtype>(%arg0: !kgen.simd<size, type>) -> !kgen.struct<(simd<size, type>)> {
  %1 = kgen.struct.create(%arg0) : !kgen.struct<(simd<size, type>)>
  kgen.return %1 : !kgen.struct<(simd<size, type>)>
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it<size, type: dtype, target: dtype>(%arg0: !kgen.struct<(simd<size, type>)>) -> !kgen.struct<(simd<size, target>)> {
  %1 = kgen.struct.extract %arg0[0] : !kgen.struct<(simd<size, type>)>
  %3 = pop.cast %1 : !kgen.simd<size, type> to !kgen.simd<size, target>
  // CHECK: kgen.call @struct_of_simd<size, :dtype target>
  // CHECK-SAME: (!kgen.simd<size, target>) -> !kgen.struct<(simd<size, target>)>
  %4 = kgen.call @struct_of_simd<size, :dtype target>(%3) : (!kgen.simd<size, target>) -> !kgen.struct<(simd<size, target>)>
  kgen.return %4 : !kgen.struct<(simd<size, target>)>
}

// -----

lit.struct.decl @TakeArrayStruct<t, a: !pop.array<t, i1>> {}

kgen.generator @pass_index<t>(%arg0: index) -> !pop.array<t, i1> {
  %0 = "foo.op"() : () -> !pop.array<t, i1>
  kgen.return %0 : !pop.array<t, i1>
}

// CHECK-LABEL: kgen.generator @apply_result
// CHECK-SAME: @TakeArrayStruct<t, :array<t, i1> apply(:(index) -> !pop.array<t, i1> @pass_index<t>, t)>
kgen.generator @apply_result<t>(
  %arg0: !lit.struct<@TakeArrayStruct<
    t,
    :array<t, i1> apply(:(index) -> !pop.array<t, i1> @pass_index<t>, t)
  >>
) {
  kgen.return
}

// -----

lit.struct.decl @Int {}

kgen.generator @make(%arg0: index) -> !lit.struct<@Int> {
  kgen.unreachable
}

lit.struct.decl @List<l: @Int> {}

kgen.generator @create<l: @Int>() -> !lit.struct<@List<:@Int l>> {
  kgen.unreachable
}

lit.struct.decl @Buf<r: @Int, s: @List<:@Int r>> {}

// CHECK-LABEL: kgen.generator @buffer
kgen.generator @buffer<rank>() ->
  !lit.struct<@Buf<
    :@Int apply(:(index) -> !lit.struct<@Int> @make, rank),
    :@List<:@Int apply(:(index) -> !lit.struct<@Int> @make, rank)>
      apply(:() -> !lit.struct<@List<:@Int apply(:(index) -> !lit.struct<@Int> @make, rank)>>
              @create<:@Int apply(:(index) -> !lit.struct<@Int> @make, rank)>)>> {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @ref_it
kgen.generator @ref_it() {
  // CHECK-NEXT: apply(:() -> !lit.struct<@List<:!lit.struct<@Int> apply(:(index) -> !lit.struct<@Int> @make, *(1,0))
  kgen.param.declare fn: <index>() ->
     !lit.struct<@Buf<
       :@Int apply(:(index) -> !lit.struct<@Int> @make, *(0,0)),
       :@List<:@Int apply(:(index) -> !lit.struct<@Int> @make, *(0,0))>
         apply(:() -> !lit.struct<@List<:@Int apply(:(index) -> !lit.struct<@Int> @make, *(1,0))>>
                 @create<:@Int apply(:(index) -> !lit.struct<@Int> @make, *(0,0))>)>>
    = <@buffer>
  kgen.return
}

// -----

kgen.generator @pass_type<T: type> () -> !kgen.param<T> {
  kgen.unreachable
}

kgen.generator @use() {
  // COM: Construct a scenario where a signature with an escaped index reference
  // COM: is being passed as a type parameter to a function that references it
  // COM: in its result.
  // CHECK: rebind(:() -> !kgen.simd<*(1,0), si8> apply(:() -> !kgen.generator<() -> !kgen.simd<*(2,0), si8>> @pass_type<:type () -> !kgen.simd<*(1,0), si8>>)
  kgen.param.declare use: <
    index,
    !kgen.param<rebind(:() -> !kgen.simd<*(1,0), si8>
      apply(
        :() -> !kgen.generator<() -> !kgen.simd<*(2,0), si8>>
          @pass_type<:type () -> !kgen.simd<*(1,0), si8>>))>
  >() -> () = <?>
  kgen.return
}

// -----

lit.struct.decl @T<a> {
}

// CHECK-LABEL: kgen.generator @f
kgen.generator @f<a, b: @T<a>>() -> !kgen.type {
  // CHECK-NEXT: ref: <index, !lit.struct<@T<*(0,0)>>, <apply(:() -> !kgen.type @f<*(1,0), :!lit.struct<@T<*(1,0)>> *(1,1)>)>() -> ()>() -> () = <?>
  kgen.param.declare ref: <index, @T<*(0,0)>, <apply(:() -> !kgen.type @f<*(1,0), :@T<*(1,0)> *(1,1)>)>() -> ()>() -> () = <?>

  // CHECK-NEXT: relative: <type, <*(1,0)>(!kgen.pointer<:!kgen.param<*(1,0)> *(0,0)>) -> ()>
  kgen.param.declare relative: <type, <*(1,0)>(!kgen.pointer<:!kgen.param<*(1,0)> *(0,0)>) -> ()>() -> () = <?>
  kgen.unreachable
}

// -----

kgen.generator @f<a, b, c: array<b, index>>() {
  kgen.return
}


// COM: Partially bind a function with dependent parameters.

// CHECK-LABEL: @partially_bind_dependent
kgen.generator @partially_bind_dependent() {
  // CHECK-NEXT: partial: <index, array<*(0,0), index>>() -> () = <@f<1, ?, :array<?, index> ?>>
  kgen.param.declare partial: <index, array<*(0,0), index>>() -> () = <@f<1, ?, :array<?, index> ?>>
  kgen.return
}
