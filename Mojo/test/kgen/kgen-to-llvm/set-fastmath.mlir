//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Test the `-set-fastmath` pass that implements `-fp-mode` by controlling the
// `contract` fast-math flag on every op that takes fastmath flags.

// RUN: kgen-opt %s -set-fastmath=contract=true | FileCheck %s --check-prefix=CONTRACT
// RUN: kgen-opt %s -set-fastmath=contract=false | FileCheck %s --check-prefix=PRECISE

// `contract=true` (default) is a no-op: mul/add/sub keep their built-in
// `contract` flag, which is elided on print because it is the attribute's
// default. An explicitly-requested `fast` is likewise left untouched.
// CONTRACT-LABEL: kgen.func @arith
// CONTRACT: pop.mul %arg0, %arg1 : !kgen.scalar<f32>
// CONTRACT: pop.add %0, %arg2 : !kgen.scalar<f32>
// CONTRACT: pop.sub %1, %arg2 : !kgen.scalar<f32>
// CONTRACT-LABEL: kgen.func @explicit
// CONTRACT: pop.mul %arg0, %arg1 {fastmathFlags = #pop.fmf<fast>}
// The pass reaches every op taking the flags, not just mul/add/sub.
// CONTRACT-LABEL: kgen.func @other_ops
// CONTRACT: pop.fma {{.*}} {fastmathFlags = #pop.fmf<contract>}
// CONTRACT: pop.cast {{.*}} {fastmathFlags = #pop.fmf<contract>}

// `contract=false` clears only the `contract` bit: mul/add/sub become `none`;
// the explicit `fast` keeps every bit except `contract`.
// PRECISE-LABEL: kgen.func @arith
// PRECISE: pop.mul %arg0, %arg1 {fastmathFlags = #pop.fmf<none>} : !kgen.scalar<f32>
// PRECISE: pop.add %0, %arg2 {fastmathFlags = #pop.fmf<none>} : !kgen.scalar<f32>
// PRECISE: pop.sub %1, %arg2 {fastmathFlags = #pop.fmf<none>} : !kgen.scalar<f32>
// PRECISE-LABEL: kgen.func @explicit
// PRECISE: pop.mul %arg0, %arg1 {fastmathFlags = #pop.fmf<nnan|ninf|nsz|arcp|afn|reassoc>}
// PRECISE-LABEL: kgen.func @other_ops
// PRECISE: pop.fma {{.*}} : !kgen.scalar<f32>
// PRECISE: pop.cast %0 : !kgen.scalar<f32> to !kgen.scalar<f16>

kgen.func @arith(%a: !kgen.scalar<f32>, %b: !kgen.scalar<f32>,
                 %c: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  %0 = pop.mul %a, %b : !kgen.scalar<f32>
  %1 = pop.add %0, %c : !kgen.scalar<f32>
  %2 = pop.sub %1, %c : !kgen.scalar<f32>
  kgen.return %2 : !kgen.scalar<f32>
}

kgen.func @explicit(%a: !kgen.scalar<f32>, %b: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  %0 = pop.mul %a, %b {fastmathFlags = #pop.fmf<fast>} : !kgen.scalar<f32>
  kgen.return %0 : !kgen.scalar<f32>
}

kgen.func @other_ops(%a: !kgen.scalar<f32>, %b: !kgen.scalar<f32>,
                     %c: !kgen.scalar<f32>) -> !kgen.scalar<f16> {
  %0 = pop.fma %a, %b, %c {fastmathFlags = #pop.fmf<contract>} : !kgen.scalar<f32>
  %1 = pop.cast %0 {fastmathFlags = #pop.fmf<contract>} : !kgen.scalar<f32> to !kgen.scalar<f16>
  kgen.return %1 : !kgen.scalar<f16>
}
