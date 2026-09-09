// RUN: kgen-opt -canonicalize %s | FileCheck %s

kgen.func @bitcast() -> !kgen.scalar<si64> {
  %0 = pop.global_constant: array<1, scalar<si64>> = <[2]>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<array<1, scalar<si64>>> to !kgen.pointer<scalar<si64>>
  %2 = pop.load %1 align<1> : !kgen.pointer<scalar<si64>>
  kgen.return %2 : !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @bitcast()
// CHECK-NEXT: %[[OUT:.*]] = kgen.param.constant: scalar<si64> = <2>
// CHECK-NEXT: kgen.return %[[OUT]]

kgen.func @bitcast_with_offset() -> !kgen.scalar<si64> {
  %one = kgen.param.constant = <1>
  %two = kgen.param.constant = <2>

  %0 = pop.global_constant: array<3, scalar<si64>> = <[2, 3, 4]>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<array<3, scalar<si64>>> to !kgen.pointer<scalar<si64>>

  %2 = pop.offset %1[%one] : !kgen.pointer<scalar<si64>>
  %3 = pop.offset %1[%two] : !kgen.pointer<scalar<si64>>

  %load1 = pop.load %2 : !kgen.pointer<scalar<si64>>
  %load2 = pop.load %3 : !kgen.pointer<scalar<si64>>

  %add = pop.add %load1, %load2 : !kgen.scalar<si64>
  kgen.return %add : !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @bitcast_with_offset()
// CHECK-NEXT: %[[OUT:.*]] = kgen.param.constant: scalar<si64> = <7>
// CHECK-NEXT: kgen.return %[[OUT]]

kgen.func @bitcast_muli_use_offset() -> (!kgen.scalar<si64>, !kgen.pointer<scalar<si64>>) {
  %one = kgen.param.constant = <1>
  %two = kgen.param.constant = <2>
  %0 = pop.global_constant: array<3, scalar<si64>> = <[2, 3, 4]>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<array<3, scalar<si64>>> to !kgen.pointer<scalar<si64>>
  %2 = pop.offset %1[%one] : !kgen.pointer<scalar<si64>>
  %3 = pop.offset %1[%two] : !kgen.pointer<scalar<si64>>
  %load = pop.load %3 : !kgen.pointer<scalar<si64>>
  kgen.return %load, %2 : !kgen.scalar<si64>, !kgen.pointer<scalar<si64>>
}

// We should have replaced the load and preserved the other access; the
// `pop.pointer.bitcast` + `pop.offset` chain canonicalizes to a single
// `pop.array.gep`.
// CHECK-LABEL: kgen.func @bitcast_muli_use_offset
// CHECK: %[[OUT:.*]] = kgen.param.constant: scalar<si64> = <4>
// CHECK: %[[IDX:.*]] = kgen.param.constant = <1>
// CHECK: %[[GLOBAL:.*]] = pop.global_constant: array<3, scalar<si64>> = <[2, 3, 4]>
// CHECK: %[[GEP:.*]] = pop.array.gep %[[GLOBAL]][%[[IDX]]]
// CHECK: kgen.return %[[OUT]], %[[GEP]]


kgen.func @array_gep() -> !kgen.scalar<si64> {
  %two = kgen.param.constant = <2>
  %0 = pop.global_constant: array<3, scalar<si64>> = <[2, 3, -4]>
  %1 = pop.array.gep %0[%two] : <array<3, scalar<si64>>>
  %2 = pop.load %1 align<1> : !kgen.pointer<scalar<si64>>
  kgen.return %2 : !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @array_gep()
// CHECK-NEXT: %[[OUT:.*]] = kgen.param.constant: scalar<si64> = <-4>
// CHECK-NEXT: kgen.return %[[OUT]]


kgen.func @array_neg_index() -> !kgen.scalar<si64> {
  %two = kgen.param.constant = <-2>
  %0 = pop.global_constant: array<3, scalar<si64>> = <[2, 3, -4]>
  %1 = pop.array.gep %0[%two] : <array<3, scalar<si64>>>
  %2 = pop.load %1 align<1> : !kgen.pointer<scalar<si64>>
  kgen.return %2 : !kgen.scalar<si64>
}

// Shouldn't touch negative indices.
// CHECK-LABEL: kgen.func @array_neg_index()
// CHECK: pop.global_constant

kgen.func @bitcast_neg_index() -> !kgen.scalar<si64> {
  %ntwo = kgen.param.constant = <-2>
  %0 = pop.global_constant: array<3, scalar<si64>> = <[2, 3, 4]>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<array<3, scalar<si64>>> to !kgen.pointer<scalar<si64>>
  %2 = pop.offset %1[%ntwo] : !kgen.pointer<scalar<si64>>
  %load = pop.load %2 : !kgen.pointer<scalar<si64>>
  kgen.return %load : !kgen.scalar<si64>
}

// Shouldn't touch negative indices.
// CHECK-LABEL: kgen.func @bitcast_neg_index()
// CHECK: pop.global_constant
