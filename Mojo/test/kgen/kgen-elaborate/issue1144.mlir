// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -verify-diagnostics -allow-unregistered-dialect
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -verify-diagnostics -allow-unregistered-dialect

kgen.generator @"__mlir_i1__"(%arg0: !kgen.scalar<bool> imm) -> i1 always_inline_no_debug {
  %0 = pop.cast_to_builtin %arg0 : !kgen.scalar<bool> to i1
  kgen.return %0 : i1
}

kgen.generator @err() {
  kgen.call @call() : () -> ()
  kgen.return
}

kgen.generator @"_mlirtype_is_eq"<t1: type, t2: type>() -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> identical(:type t1, t2))>
  %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  kgen.return %1 : !kgen.scalar<bool>
}

kgen.generator @getattr() {
  kgen.call @err() : () -> ()
  kgen.return
}

kgen.generator @call() {
  kgen.call @err() : () -> ()
  kgen.return
}

// expected-error @+1 {{function instantiation failed}}
kgen.generator export @main() {
  kgen.call @getattr() : () -> ()
  kgen.call @call() : () -> ()
  %variant = kgen.param.constant: variant<struct<(pointer<scalar<si8>>, index)>, none> = <#kgen.variant<:none #kgen.none, 1>>
  %variant_0 = kgen.param.constant: variant<struct<()>, scalar<bool>, scalar<si64>, scalar<f64>, struct<(pointer<scalar<si8>>, index)>, pointer<none>, pointer<scalar<si16>>, pointer<scalar<si8>>> = <#kgen.variant<:struct<()> {  }, 0>>
  hlcf.loop "inlined_cf_scope" {
      kgen.param.declare _19x17_T: type = <string>
      kgen.param.apply *"(lifted)apply_0" = [() -> !kgen.scalar<bool>: @"_mlirtype_is_eq"<:type _19x17_T, :type index>]()
      kgen.param.if <*"(lifted)apply_0"> {
        hlcf.break "inlined_cf_scope"
      } else {
        kgen.param.apply *"(lifted)apply_2" = [() -> !kgen.scalar<bool>: @"_mlirtype_is_eq"<:type _19x17_T, :type none>]()
        // expected-note @below {{constraint failed: expected Int or NoneType}}
        kgen.param.assert <*"(lifted)apply_2">, "expected Int or NoneType"
        hlcf.break "inlined_cf_scope"
      } {thenIsolated}
      kgen.unreachable
  }
  kgen.return
}
