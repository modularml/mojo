// RUN: kgen-opt %s -verify-parameters | kgen-opt -verify-parameters | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode | kgen-opt -verify-parameters | FileCheck %s

// CHECK-LABEL: lit.fn @argNameParsing(
// CHECK-SAME: %a: index, %woof: index, %_21451[*"!451"]: index
lit.fn @argNameParsing(%a: index, %b[woof]: index, %c[*"!451"]: index) {
  kgen.return
}

// CHECK-LABEL: lit.fn @outer(%foo: index) {
lit.fn @outer(%a[foo]: index) {
  // CHECK-NEXT: lit.fn @inner(%foo_0[foo]: index, %foo_0_1[foo_0]: index) {
  lit.fn @inner(%b[foo]: index, %c[foo_0]: index) {
    // CHECK-NEXT: lit.fn @more_inner(%foo_2[foo]: index) {
    lit.fn @more_inner(%d[foo]: index) {
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: lit.fn @slash(%a: index, |, %b: index, %c: index)
lit.fn @slash(%a: index, |, %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, |, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, "b": index, "c": index) -> ()> = <@slash>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashOnly()
lit.fn @slashOnly(|) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(|) -> ()> = <@slashOnly>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashFirst(%a: index, %b: index, %c: index)
lit.fn @slashFirst(|, %a: index, %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<(|, "a": index, "b": index, "c": index) -> ()> = <@slashFirst>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashLast(%a: index, %b: index, %c: index, |)
lit.fn @slashLast(%a: index, %b: index, %c: index, |) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index, |) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, "b": index, "c": index, |) -> ()> = <@slashLast>
  kgen.return
}

// CHECK-LABEL: lit.fn @star(%a: index, *, %b: index, %c: index)
lit.fn @star(%a: index, *, %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, *, "b": index, "c": index) -> ()> = <@star>
  kgen.return
}

// CHECK-LABEL: lit.fn @starOnly()
lit.fn @starOnly(*) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(*) -> ()> = <@starOnly>
  kgen.return
}

// CHECK-LABEL: lit.fn @starFirst(*, %a: index, %b: index, %c: index)
lit.fn @starFirst(*, %a: index, %b: index, %c: index) {
  // CHECK: !lit.generator<(*, "a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<(*, "a": index, "b": index, "c": index) -> ()> = <@starFirst>
  kgen.return
}

// CHECK-LABEL: lit.fn @starLast(%a: index, %b: index, %c: index)
lit.fn @starLast(%a: index, %b: index, %c: index, *) {
  // CHECK: !lit.generator<("a": index, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, "b": index, "c": index, *) -> ()> = <@starLast>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashAndStar(%a: index, |, %b: index, *, %c: index)
lit.fn @slashAndStar(%a: index, |,  %b: index, *, %c: index) {
  // CHECK: !lit.generator<("a": index, |, "b": index, *, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, "b": index, *, "c": index) -> ()> = <@slashAndStar>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashAndStarTogether(%a: index, |, *, %b: index, %c: index)
lit.fn @slashAndStarTogether(%a: index, |, *,  %b: index, %c: index) {
  // CHECK: !lit.generator<("a": index, |, *, "b": index, "c": index) -> ()>
  kgen.param.declare self: !lit.generator<("a": index, |, *, "b": index, "c": index) -> ()> = <@slashAndStarTogether>
  kgen.return
}

// CHECK-LABEL: lit.fn @slashAndStarOnly()
lit.fn @slashAndStarOnly(|, *) {
  // CHECK: !lit.generator<() -> ()>
  kgen.param.declare self: !lit.generator<(|, *) -> ()> = <@slashAndStarOnly>
  kgen.return
}

// CHECK-LABEL: lit.fn @signature_type<dt: dtype, w: scalar<dt>>(%a: index owned = 1)
lit.fn @signature_type<dt: dtype, w: scalar<dt>>(%a: index owned = 1) {
  // CHECK: self: !lit.generator<<"dt": dtype, "w": scalar<*(0,0)>>("a": index owned = 1) -> ()> = <@signature_type>
  kgen.param.declare self: !lit.generator<<"dt": dtype, "w": scalar<*(0,0)>>("a": index owned = 1) -> ()> = <@signature_type>
  // CHECK: call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.generator<("a": index owned = 1) -> ()>
  kgen.call @signature_type<:dtype si32, :scalar<si32> 1>(%a) : !lit.generator<("a": index owned = 1) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.fn @default_params<
// CHECK-SAME: a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
// CHECK-SAME: d: dtype, e: dtype = si8, f: scalar<si16> = 2
// CHECK-SAME: >(%z: index owned = 42)
lit.fn @default_params<
  a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
  d: dtype, e: dtype = si8, f: scalar<si16> = 2
>(%z: index owned = 42) {
  // CHECK: self: !lit.generator<
  // CHECK-SAME: <"a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  // CHECK-SAME: >("z": index owned = 42) -> ()> = <@default_params>
  kgen.param.declare self: !lit.generator<<
    "a": dtype, "b": dtype = f32, "c": scalar<si32> = 1, *, "d": dtype, "e": dtype = si8, "f": scalar<si16> = 2
  >("z": index owned = 42) -> ()> = <@default_params>

  // CHECK: call @default_params<
  // CHECK-SAME: :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  // CHECK-SAME: >(%z) : !lit.generator<("z": index owned = 42) -> ()>
  kgen.call @default_params<
    :dtype si16, :dtype f16, :scalar<si32> 5, :dtype si16, :dtype f16, :scalar<si16> 5
  >(%z) : !lit.generator<("z": index owned = 42) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.fn @default_args(
// CHECK-SAME: %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3)
lit.fn @default_args(
  %a: index, %b: index = 0, %c: index = 1, *, %d: index, %e: index = 2, %f: index = 3
) {
  // CHECK: call @default_args(%a, %b, %c, %d, %e, %f) : !lit.generator<
  // CHECK-SAME: ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>
  kgen.call @default_args(%a, %b, %c, %d, %e, %f) : !lit.generator<
    ("a": index, "b": index = 0, "c": index = 1, *, "d": index, "e": index = 2, "f": index = 3) -> ()>

  kgen.return
}

// CHECK-LABEL: lit.fn @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index owned = 42)
lit.fn @star_slash_params<a: dtype, |, b: dtype = f32, *, w: scalar<si32> = 1>(%z: index owned = 42) {
  // CHECK: self: !lit.generator<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index owned = 42) -> ()> = <@star_slash_params>
  kgen.param.declare self: !lit.generator<<"a": dtype, |, "b": dtype = f32, *, "w": scalar<si32> = 1>("z": index owned = 42) -> ()> = <@star_slash_params>
  kgen.return
}

lit.fn @create_simd<x>() -> !kgen.simd<x, si8> {
  kgen.unreachable
}

// CHECK-LABEL: lit.fn @parametric_default_arg
// CHECK-SAME: <x>(%y: !kgen.simd<x, si8> =
// CHECK-SAME: apply(:!lit.generator<() -> !kgen.simd<x, si8>> @create_simd<x>))
lit.fn @parametric_default_arg<x>(%y: !kgen.simd<x, si8> =
    apply(:!lit.generator<() -> !kgen.simd<x, si8>> @create_simd<x>)) {
  kgen.return
}

// CHECK-LABEL: lit.fn @call_parametric_default_arg
lit.fn @call_parametric_default_arg(%x: !kgen.simd<4, si8>) {
  // CHECK: call @parametric_default_arg<4>(%x) : !lit.generator<("y": !kgen.simd<4, si8> =
  // CHECK-SAME: apply(:!lit.generator<() -> !kgen.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.call @parametric_default_arg<4>(%x) : !lit.generator<("y": !kgen.simd<4, si8> =
    apply(:!lit.generator<() -> !kgen.simd<4, si8>> @create_simd<4>)) -> ()>
  kgen.return
}

// CHECK-LABEL: lit.fn @parametric_default_param
// CHECK-SAME: <x, y = x>()
lit.fn @parametric_default_param<x, y = x>() {
  kgen.return
}

// CHECK-LABEL: @call_default_param
lit.fn @call_default_param() {
  // CHECK: ref: !lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  kgen.param.declare ref: !lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> = <@parametric_default_param>
  // CHECK: bound: !lit.generator<<"y": index = 1>() -> ()> = <bind_params(
  // CHECK-SAME: :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.param.declare bound: !lit.generator<<"y": index = 1>() -> ()> = <bind_params(
    :!lit.generator<<"x": index, "y": index = *(0,0)>() -> ()> ref, 1, ?)>
  kgen.return
}

// CHECK-LABEL: @address_default
// CHECK-SAME: %p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>
lit.fn @address_default[mut lt](%p: !lit.ref<simd<2, si8>, mut lt> owned_in_mem = <1, 2>) {
  // CHECK: ref: !lit.generator<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.param.declare ref: !lit.generator<[1]("p": !lit.ref<simd<2, si8>, mut *[0,0]> owned_in_mem = <1, 2>) -> ()> = <@address_default>
  kgen.return
}

// CHECK-LABEL: lit.fn @inferred
// CHECK-SAME: <a: i1, b, +, c = 1, |>
lit.fn @inferred<a: i1, b, +, c = 1, |>() {
  // CHECK-NEXT: !lit.generator<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()>
  kgen.param.constant: !lit.generator<<"a": i1, "b": index, +, "c": index = 1, |>() -> ()> = <@inferred>

  // CHECK-NEXT: !lit.generator<<index, +, *, index>() -> ()> = <?>
  kgen.param.constant: !lit.generator<<index, +, *, index>() -> ()> = <?>

  // CHECK-NEXT: !lit.generator<<index, +>() -> ()> = <?>
  kgen.param.constant: !lit.generator<<index, +>() -> ()> = <?>
  kgen.return
}

// CHECK-LABEL: lit.fn @different_param_name
lit.fn @different_param_name() {
  // CHECK: lit.fn nested_fn<["a"]param, |>()
  lit.fn nested_fn<["a"]param, |>() {
    kgen.return
  }
  // CHECK: ref: !lit.generator<<"a": index, |>() -> ()> = <nested_fn>
  kgen.param.declare ref: !lit.generator<<"a": index, |>() -> ()> = <nested_fn>
  kgen.return
}

// CHECK-LABEL: lit.fn @lifetime_set
lit.fn @lifetime_set<set: origin.set>[mut lt]() {
  // CHECK-NEXT: f0: !lit.generator<:set:() capturing -> ()> = <?>
  kgen.param.declare f0: !lit.generator<:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f1: !lit.generator<:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f1: !lit.generator<:{mut lt}:() capturing -> ()> = <?>
  // CHECK-NEXT: f2: !lit.generator<[1]:set:() capturing -> ()> = <?>
  kgen.param.declare f2: !lit.generator<[1]:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f3: !lit.generator<[1]:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f3: !lit.generator<[1]:{mut lt}:() capturing -> ()> = <?>
  // CHECK-NEXT: f4: !lit.generator<<index>:set:() capturing -> ()> = <?>
  kgen.param.declare f4: !lit.generator<<index>:set:() capturing -> ()> = <?>
  // CHECK-NEXT: f5: !lit.generator<<index>:{mut lt}:() capturing -> ()> = <?>
  kgen.param.declare f5: !lit.generator<<index>:{mut lt}:() capturing -> ()> = <?>
  kgen.return
}

// CHECK-LABEL: lit.fn @lambda_capture_lifetimes
lit.fn @lambda_capture_lifetimes<set: origin.set>[mut lt]() {
  // CHECK: lit.fn set_capture:set:<param>() capturing
  lit.fn set_capture:set:<param>() capturing {
    kgen.return
  }
  // CHECK: lit.fn lt_capture:{mut lt}:() capturing
  lit.fn lt_capture:{mut lt}:() capturing {
    kgen.return
  }

  // CHECK: ref0: !lit.generator<<"param": index>:set:() capturing -> ()> = <set_capture>
  kgen.param.declare ref0: !lit.generator<<"param": index>:set:() capturing -> ()> = <set_capture>
  // CHECK: ref1: !lit.generator<:{mut lt}:() capturing -> ()> = <lt_capture>
  kgen.param.declare ref1: !lit.generator<:{mut lt}:() capturing -> ()> = <lt_capture>
  kgen.return
}
