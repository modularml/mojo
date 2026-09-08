// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// Test that `#kgen.get_witness` resolves a trait conformance supplied by a
// `#kgen.extension` when the anchor type does not itself carry that trait.

kgen.generator @call(%arg0: !kgen.struct<(index)>) -> index {
  %c = kgen.param.constant : index = <1>
  kgen.return %c : index
}

// The source closure struct: conforms to `def() -> T` with `T = index`.
kgen.struct.generator @make = struct_inst<"make"(data: index)> {
  kgen.conformance @"def() -> T" {
    kgen.witness "T" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call
  }
}

// The bridging extension struct: stateless, single type parameter `A`, conforms
// to `def() -> V` by forwarding each witness to `A`'s own `def() -> T` witnesses.
kgen.struct.generator @extension1<A: type> = struct_inst<"extension1"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.param<A>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

// `sink` looks up the `def() -> V` __call__ witness on `F` and invokes it.
// CHECK-LABEL: kgen.func @"sink
// CHECK: kgen.call @call
kgen.generator @sink<V: type, F: type>(%arg: !kgen.param<F>) -> !kgen.param<V> {
  kgen.param.declare callFn : (!kgen.param<F>) -> !kgen.param<V> = <#kgen.get_witness<F, @"def() -> V", "__call__">>
  %result = kgen.call_param[(!kgen.param<F>) -> !kgen.param<V> : callFn](%arg)
  kgen.return %result : !kgen.param<V>
}

// `forward` augments `G`'s trait view with an extension supplying `def() -> V`
// CHECK-LABEL: kgen.func @"forward
// CHECK-NEXT: kgen.call @"sink
// CHECK-NEXT: kgen.return
kgen.generator @forward<T: type, G: type>(%arg: !kgen.param<G>) -> !kgen.param<T> {
  kgen.param.declare P1 : type = <#kgen.extension<G, [#kgen.type<typevalue<:!kgen.type #kgen.genref<@extension1<:type G>>>, struct<()>> : !kgen.type]> : !kgen.type>
  %arg1 = kgen.rebind %arg : !kgen.param<G> to !kgen.param<P1>
  %result = kgen.call @sink<:type T, :type P1>(%arg1) : (!kgen.param<P1>) -> !kgen.param<T>
  kgen.return %result : !kgen.param<T>
}

#make = #kgen.type<typevalue<:!kgen.type #kgen.genref<@make>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @foo
kgen.generator @foo(%arg0: !kgen.struct<(index)>) -> index {
  %0 = kgen.call @forward<:type index, :type #make>(%arg0) : (!kgen.struct<(index)>) -> index
  kgen.return %0 : index
}
