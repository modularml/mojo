// RUN: kgen-opt -sroa -allow-unregistered-dialect %s | FileCheck %s

// Check sroa runs as expected along side mem-2-reg
// RUN: kgen-opt -sroa -mem-2-reg -allow-unregistered-dialect %s | FileCheck -check-prefix="MEM2REG" %s

// CHECK-LABEL: @simple_struct
// MEM2REG-LABEL: @simple_struct
kgen.func @simple_struct(%arg1: !kgen.struct<(index, index)>) -> !kgen.scalar<index> {
  %array = pop.stack_allocation 1 x !kgen.struct<(index, index)>
  pop.store %arg1, %array : !kgen.pointer<struct<(index, index)>>

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index

  // Extract from the input and store into stack.
  // CHECK-NEXT: %[[EXTRACT:.*]] = kgen.struct.extract %[[ARG0:.*]][0] : <(index, index)>
  // CHECK-NEXT: pop.store %[[EXTRACT]], %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = kgen.struct.extract %[[ARG0]][1] : <(index, index)>
  // CHECK-NEXT: pop.store %[[EXTRACT2]], %[[MEM2]] : !kgen.pointer<index>


  // Load from stack.
  // CHECK-NEXT: pop.load %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.load %[[MEM2]] : !kgen.pointer<index>


  // When running with mem2reg check we get rid of the allocs.
  // MEM2REG-NEXT: %[[SCALAR1:.*]] = kgen.struct.extract %[[ARG0:.*]][0] : <(index, index)>
  // MEM2REG-NEXT: %[[SCALAR2:.*]] =  kgen.struct.extract %[[ARG0]][1] : <(index, index)>
  // MEM2REG-NEXT: pop.cast_from_builtin %[[SCALAR1]] : index to !kgen.scalar<index>
  // MEM2REG-NEXT: pop.cast_from_builtin %[[SCALAR2]] : index to !kgen.scalar<index>

  %gep1 = kgen.struct.gep %array[0] : <struct<(index, index)>>
  %gep2 = kgen.struct.gep %array[1] : <struct<(index, index)>>

  %load1 = pop.load %gep1 : !kgen.pointer<index>
  %load2 = pop.load %gep2 : !kgen.pointer<index>
  %scalar1 = pop.cast_from_builtin %load1 : index to !kgen.scalar<index>
  %scalar2 = pop.cast_from_builtin %load2 : index to !kgen.scalar<index>
  %out = pop.add %scalar1, %scalar2 : !kgen.scalar<index>
  kgen.return %out : !kgen.scalar<index>
}

// CHECK-LABEL: @simple_array
// MEM2REG-LABEL: @simple_array
kgen.func @simple_array(%arg1: !pop.array<2, index>) -> !kgen.scalar<index> {
   %0 = kgen.param.constant = <0>
   %1 = kgen.param.constant = <1>

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index

  // Extract from the input and store into stack.
  // CHECK-NEXT: %[[GET:.*]] = pop.array.get %[[ARG0:.*]][0] : !pop.array<2, index>
  // CHECK-NEXT: pop.store %[[GET]], %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: %[[GET2:.*]] = pop.array.get %[[ARG0]][1] : !pop.array<2, index>
  // CHECK-NEXT: pop.store %[[GET2]], %[[MEM2]] : !kgen.pointer<index>


  // Load from stack.
  // CHECK-NEXT: pop.load %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.load %[[MEM2]] : !kgen.pointer<index>

  // MEM2REG: %[[OP1:.*]] = pop.array.get %[[ARG0:.*]][0] : !pop.array<2, index>
  // MEM2REG-NEXT: %[[OP2:.*]] = pop.array.get %[[ARG0]][1] : !pop.array<2, index>
  // MEM2REG-NEXT: %[[CONVERT1:.*]] = pop.cast_from_builtin %[[OP1]] : index to !kgen.scalar<index>
  // MEM2REG-NEXT: %[[CONVERT2:.*]] = pop.cast_from_builtin %[[OP2]] : index to !kgen.scalar<index>
  // MEM2REG-NEXT: %[[ADD:.*]] = pop.add %[[CONVERT1]], %[[CONVERT2]] : !kgen.scalar<index>
  // MEM2REG-NEXT: kgen.return %[[ADD]] : !kgen.scalar<index>

   %array = pop.stack_allocation 1 x !pop.array<2, index>
   pop.store %arg1, %array : !kgen.pointer<array<2, index>>

   %gep1 = pop.array.gep %array[%0] : <array<2, index>>
   %gep2 = pop.array.gep %array[%1] : <array<2, index>>

   %load1 = pop.load %gep1 : !kgen.pointer<index>
   %load2 = pop.load %gep2 : !kgen.pointer<index>
   %scalar1 = pop.cast_from_builtin %load1 : index to !kgen.scalar<index>
   %scalar2 = pop.cast_from_builtin %load2 : index to !kgen.scalar<index>
   %out = pop.add %scalar1, %scalar2 : !kgen.scalar<index>
   kgen.return %out : !kgen.scalar<index>
 }

// CHECK-LABEL: @struct_of_structs
// MEM2REG-LABEL: @struct_of_structs
kgen.func @struct_of_structs(%arg1: !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>) {
  %memory = pop.stack_allocation 1 x !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
  pop.store %arg1, %memory : !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>
  hlcf.loop {
    %load = pop.load %memory : !kgen.pointer<struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>
    hlcf.loop "inlined_cf_scope" {
      %getElem1 = kgen.struct.extract %load[2] : !kgen.struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
      %getElem2 = kgen.struct.extract %getElem1[0] : !kgen.struct<(scalar<index>)>

      %gep = kgen.struct.gep %memory[0] : <struct<(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>>
      %newLoad = pop.load %gep : !kgen.pointer<struct<(scalar<index>)>>
      %getElem3 = kgen.struct.extract %newLoad[0] : !kgen.struct<(scalar<index>)>

      %out = pop.div %getElem3, %getElem2 : !kgen.scalar<index>
      hlcf.break
    }
    hlcf.break
  }

  // Just check this has been broken into several allocations.
  // CHECK-NEXT: %[[MEM1:.*]] = pop.stack_allocation 1 x scalar<index>
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x scalar<index>
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x scalar<index>

  // In this test the incoming argument is a struct of structs so we can't
  // sroa the argument. Still check that we are still left with no stack alloc
  // and that all the innerloop uses are of the fully extracted base type.

  // MEM2REG: %[[OP0:.*]] = kgen.struct.extract %[[ARG0:.*]][0] : <(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
  // MEM2REG-DAG: %[[OP1:.*]] = kgen.struct.extract %[[OP0]][0] : <(scalar<index>)>
  // MEM2REG-DAG: %[[OP2:.*]] = kgen.struct.extract %[[ARG0]][1] : <(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
  // MEM2REG-DAG: %[[OP3:.*]] = kgen.struct.extract %[[OP2]][0] : <(scalar<index>)>
  // MEM2REG-DAG: %[[OP4:.*]] = kgen.struct.extract %[[ARG0]][2] : <(struct<(scalar<index>)>, struct<(scalar<index>)>, struct<(scalar<index>)>)>
  // MEM2REG-DAG: %[[OP5:.*]] = kgen.struct.extract %[[OP4]][0] : <(scalar<index>)>
  // MEM2REG-DAG:  hlcf.loop {
  // MEM2REG-DAG:    hlcf.loop "inlined_cf_scope" {
  // MEM2REG-DAG:       pop.div %[[OP1]], %[[OP5]] : !kgen.scalar<index>

  kgen.return
}

// CHECK-LABEL: @stack_of_N
// MEM2REG-LABEL: @stack_of_N
// CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
kgen.func @stack_of_N(%val1: index, %val2: index, %val3: index, %output : !kgen.pointer<index>) {
  %0 = kgen.param.constant = <0>
  %1 = kgen.param.constant = <1>
  %2 = kgen.param.constant = <2>

  %alloc = pop.stack_allocation 3 x index

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %[[ARG0]], %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store %[[ARG1]], %[[MEM2]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store %[[ARG2]], %[[MEM3]] : !kgen.pointer<index>

  // Mem2Reg should eliminate everything
  // MEM2REG-NEXT: kgen.param.constant = <0>
  // MEM2REG-NEXT: kgen.param.constant = <1>
  // MEM2REG-NEXT: kgen.param.constant = <2>
  // MEM2REG-NEXT: pop.store %[[ARG2]], %[[OUT_PTR]] : !kgen.pointer<index>
  // MEM2REG-NEXT: kgen.return

  %offset1 = pop.offset %alloc[%0] : !kgen.pointer<index>
  pop.store %val1, %offset1 : !kgen.pointer<index>

  %offset2 = pop.offset %alloc[%1] : !kgen.pointer<index>
  pop.store %val2, %offset2 : !kgen.pointer<index>

  %offset3 = pop.offset %alloc[%2] : !kgen.pointer<index>
  pop.store %val3, %offset3 : !kgen.pointer<index>

  %annoying_offset = pop.offset %alloc[%2] : !kgen.pointer<index>
  %load = pop.load %annoying_offset align<8> : !kgen.pointer<index>
  pop.store %load, %output : !kgen.pointer<index>
  kgen.return
}


// CHECK-LABEL: @bigger_stack
// CHECK: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
kgen.func @bigger_stack(%val1: index, %output : !kgen.pointer<index>) {
  %0 = kgen.param.constant = <0>

  // Larger stacks should not be touched.
  // CHECK: pop.stack_allocation 512 x index

  %alloc = pop.stack_allocation 512 x index
  %offset = pop.offset %alloc[%0] : !kgen.pointer<index>
  pop.store %val1, %offset : !kgen.pointer<index>
  %load = pop.load %offset align<8> : !kgen.pointer<index>
  pop.store %load, %output : !kgen.pointer<index>
  kgen.return
}

// Handle storing directly to the stack as an implicit offset of 0.
// CHECK-LABEL: @n_stack_store
// MEM2REG-LABEL: @n_stack_store
// CHECK: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
kgen.func @n_stack_store(%val1: index, %output : !kgen.pointer<index>) {
  %alloc = pop.stack_allocation 3 x index
  pop.store %val1, %alloc : !kgen.pointer<index>
  %load = pop.load %alloc align<8> : !kgen.pointer<index>
  pop.store %load, %output : !kgen.pointer<index>

  // CHECK-NEXT: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %[[ARG0]], %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: %[[LOAD:.*]] = pop.load %[[MEM1]] : !kgen.pointer<index>
  // CHECK-NEXT: pop.store %[[LOAD]], %[[OUT_PTR]] : !kgen.pointer<index>


  // Mem2Reg should eliminate everything
  // MEM2REG-NEXT: pop.store %[[ARG0]], %[[OUT_PTR]] : !kgen.pointer<index>
  // MEM2REG-NEXT: kgen.return

  kgen.return
}

// CHECK-LABEL: @n_stack_arrays
// MEM2REG-LABEL: @n_stack_arrays
// CHECK: (%[[ARG0:.*]]: !pop.array<3, index>, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: !pop.array<3, index>, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
kgen.func @n_stack_arrays(%val: !pop.array<3, index>, %output : !kgen.pointer<index>) {
  %0 = kgen.param.constant = <0>
  %1 = kgen.param.constant = <1>

  // CHECK: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index

  // Mem2reg should have enough information to realize they are aliases of the
  // same value.
  // MEM2REG: pop.array.get %[[ARG0]][0] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][1] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][2] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][0] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][1] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][2] : !pop.array<3, index>

  %alloc = pop.stack_allocation 5 x !pop.array<3, index>
  pop.store %val, %alloc : !kgen.pointer<array<3, index>>
  %offset = pop.offset %alloc[%1] : !kgen.pointer<array<3, index>>
  pop.store %val, %offset : !kgen.pointer<array<3, index>>

  %gep1 = pop.array.gep %alloc[%0] : <array<3, index>>
  %gep2 = pop.array.gep %offset[%1] : <array<3, index>>

  %load = pop.load %gep1 align<8> : !kgen.pointer<index>
  pop.store %load, %output : !kgen.pointer<index>

  %load2 = pop.load %gep2 align<8> : !kgen.pointer<index>
  pop.store %load2, %output : !kgen.pointer<index>

  kgen.return
}

// CHECK-LABEL: @n_stack_structs
// MEM2REG-LABEL: @n_stack_structs
// CHECK: (%[[ARG0:.*]]: !kgen.struct<(index, index)>, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: !kgen.struct<(index, index)>, %[[OUT_PTR:.*]]: !kgen.pointer<index>)
kgen.func @n_stack_structs(%val: !kgen.struct<(index, index)>, %output : !kgen.pointer<index>) {
  %1 = kgen.param.constant = <1>

  %alloc = pop.stack_allocation 5 x !kgen.struct<(index, index)>
  pop.store %val, %alloc : !kgen.pointer<struct<(index, index)>>
  %offset = pop.offset %alloc[%1] : !kgen.pointer<struct<(index, index)>>
  pop.store %val, %offset : !kgen.pointer<struct<(index, index)>>

  // CHECK: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index


  // MEM2REG: kgen.struct.extract %[[ARG0]][0] : <(index, index)>
  // MEM2REG-NEXT: kgen.struct.extract %[[ARG0]][1] : <(index, index)>
  // MEM2REG-NEXT: kgen.struct.extract %[[ARG0]][0] : <(index, index)>
  // MEM2REG-NEXT: kgen.struct.extract %[[ARG0]][1] : <(index, index)>

  %gep1 = kgen.struct.gep %alloc[0] : <struct<(index, index)>>
  %gep2 = kgen.struct.gep %offset[1] : <struct<(index, index)>>

  %load = pop.load %gep1 align<8> : !kgen.pointer<index>
  pop.store %load, %output : !kgen.pointer<index>

  %load2 = pop.load %gep2 align<8> : !kgen.pointer<index>
  pop.store %load2, %output : !kgen.pointer<index>

  kgen.return
}

// CHECK-LABEL: @store_arg
kgen.func @store_arg(
    %arg0: !kgen.pointer<pointer<index>>,
    %arg1: !kgen.pointer<pointer<struct<(index)>>>,
    %arg2: !kgen.pointer<pointer<array<2, index>>>) {
  // CHECK: stack_allocation 2 x index
  %0 = pop.stack_allocation 2 x index
  pop.store %0, %arg0 : !kgen.pointer<pointer<index>>
  // CHECK: stack_allocation 1 x struct<(index)>
  %1 = pop.stack_allocation 1 x !kgen.struct<(index)>
  pop.store %1, %arg1 : !kgen.pointer<pointer<struct<(index)>>>
  // CHECK: stack_allocation 1 x array<2, index>
  %2 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %2, %arg2 : !kgen.pointer<pointer<array<2, index>>>
  kgen.return
}

// CHECK-LABEL: kgen.func @negArrayGep
kgen.func @negArrayGep() {
  // CHECK: kgen.param.constant = <-1>
  // CHECK-NEXT: pop.stack_allocation 1 x array<2, index>
  // CHECK-NEXT: pop.array.gep
  %0 = kgen.param.constant = <-1>
  %array = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %array[%0] : <array<2, index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @negOffsetGep
kgen.func @negOffsetGep() {
  // CHECK: kgen.param.constant = <-1>
  // CHECK-NEXT: pop.stack_allocation 2 x index
  // CHECK-NEXT: pop.offset
  %0 = kgen.param.constant = <-1>
  %alloc = pop.stack_allocation 2 x index
  %offset = pop.offset %alloc[%0] : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @oobArrayGep
kgen.func @oobArrayGep() {
  // CHECK: kgen.param.constant = <2>
  // CHECK-NEXT: pop.stack_allocation 1 x array<2, index>
  // CHECK-NEXT: pop.array.gep
  %0 = kgen.param.constant = <2>
  %array = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %array[%0] : <array<2, index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @oobOffsetGep
kgen.func @oobOffsetGep() {
  // CHECK: kgen.param.constant = <2>
  // CHECK-NEXT: pop.stack_allocation 2 x index
  // CHECK-NEXT: pop.offset
  %0 = kgen.param.constant = <2>
  %alloc = pop.stack_allocation 2 x index
  %offset = pop.offset %alloc[%0] : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func @load_of_array
// MEM2REG-LABEL: kgen.func @load_of_array
kgen.func @load_of_array(%arg0: !pop.array<4, index>) -> !pop.array<4, index> {
  %array = kgen.param.constant: array<4, index> = <[0, 0, 0, 0]>
  %0 = kgen.param.constant = <3>
  %1 = kgen.param.constant = <2>
  %2 = kgen.param.constant = <1>
  %3 = kgen.param.constant = <0>
  %4 = pop.stack_allocation 1 x !pop.array<4, index>
  pop.store %array, %4 : !kgen.pointer<array<4, index>>
  %5 = pop.array.get %arg0[0] : !pop.array<4, index>
  %6 = pop.array.gep %4[%3] : <array<4, index>>
  pop.store %5, %6 : !kgen.pointer<index>
  %7 = pop.array.gep %4[%2] : <array<4, index>>
  pop.store %2, %7 : !kgen.pointer<index>
  %8 = pop.array.get %arg0[2] : !pop.array<4, index>
  %9 = pop.array.gep %4[%1] : <array<4, index>>
  pop.store %8, %9 : !kgen.pointer<index>
  %10 = pop.array.get %arg0[3] : !pop.array<4, index>
  %11 = pop.array.gep %4[%0] : <array<4, index>>
  pop.store %10, %11 : !kgen.pointer<index>
  %12 = pop.load %4 : !kgen.pointer<array<4, index>>
  kgen.return %12 : !pop.array<4, index>
}

// Check sroa has decomposed it.
// CHECK: pop.stack_allocation 1 x index
// CHECK: pop.stack_allocation 1 x index
// CHECK: pop.stack_allocation 1 x index
// CHECK: pop.stack_allocation 1 x index

// MEM2REG:(%[[IN_ARRAY:.*]]: !pop.array<4, index>
// MEM2REG: %[[INDEX:.*]] = kgen.param.constant = <1>

// MEM2REG: %[[FIRST:.*]] = pop.array.get %[[IN_ARRAY]][0] : !pop.array<4, index>
// MEM2REG: %[[THIRD:.*]] = pop.array.get %[[IN_ARRAY]][2] : !pop.array<4, index>
// MEM2REG: %[[FOURTH:.*]] = pop.array.get %[[IN_ARRAY]][3] : !pop.array<4, index>
// MEM2REG: %[[OUT:.*]] = pop.array.create [%[[FIRST]], %[[INDEX]], %[[THIRD]], %[[FOURTH]]] : !pop.array<4, index>
// MEM2REG: kgen.return %[[OUT]] : !pop.array<4, index>

// CHECK-LABEL: @offset_of_offset
kgen.func @offset_of_offset() -> !kgen.pointer<index> {
  %0 = pop.stack_allocation 2 x index
  %idx1 = index.constant 1
  %1 = pop.offset %0[%idx1] : !kgen.pointer<index>
  kgen.return %1 : !kgen.pointer<index>
}


// CHECK-LABEL: @large_array
kgen.func @large_array(%arg1: !pop.array<512, index>) -> index {
  %0 = kgen.param.constant = <0>
  %array = pop.stack_allocation 1 x !pop.array<512, index>
  // Larger arrays should not be touched.
  // CHECK: pop.stack_allocation 1 x array<512, index>
  pop.store %arg1, %array : !kgen.pointer<array<512, index>>
  %gep = pop.array.gep %array[%0] : <array<512, index>>
  %load = pop.load %gep : !kgen.pointer<index>
  kgen.return %load : index
}

// CHECK-LABEL: kgen.func @destructure_load
kgen.func @destructure_load() -> !kgen.struct<(i1, i2)> {
  // CHECK-NEXT: [[I1:%.*]] = pop.stack_allocation 1 x i1
  // CHECK-NEXT: [[I2:%.*]] = pop.stack_allocation 1 x i2
  %0 = pop.stack_allocation 1 x !kgen.struct<(i1, i2)>
  // CHECK-NEXT: [[I1V:%.*]] = pop.load [[I1]]
  // CHECK-NEXT: [[I2V:%.*]] = pop.load [[I2]]
  // CHECK-NEXT: [[S:%.*]] = kgen.struct.create([[I1V]], [[I2V]])
  %1 = pop.load %0 : !kgen.pointer<struct<(i1, i2)>>
  // CHECK-NEXT: return [[S]]
  kgen.return %1 : !kgen.struct<(i1, i2)>
}

// CHECK-LABEL: kgen.func @two_users
kgen.func @two_users() {
  %0 = pop.stack_allocation 1 x struct<(i1, i2)>
  %1 = pop.load %0 : !kgen.pointer<struct<(i1, i2)>>
  // CHECK: [[S:%.*]] = kgen.struct.create
  // CHECK-NEXT: call @use1([[S]])
  kgen.call @use1(%1) : (!kgen.struct<(i1, i2)>) -> ()
  // CHECK-NEXT: call @use2([[S]])
  kgen.call @use2(%1) : (!kgen.struct<(i1, i2)>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @lifetime_markers
kgen.func @lifetime_markers() {
  // CHECK-NEXT: [[S0:%.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: [[S1:%.*]] = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x struct<(index, index)> marked
  // CHECK-NEXT: [[F:%.*]] = pop.stack_allocation 1 x f32
  %1 = pop.stack_allocation 1 x f32 marked
  // CHECK-NEXT: lifetime.start([[F]], [[S0]], [[S1]])
  pop.stack_alloc.lifetime.start(%1, %0) : !kgen.pointer<f32>, !kgen.pointer<struct<(index, index)>>
  // CHECK-NEXT: lifetime.end([[F]], [[S0]], [[S1]])
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<struct<(index, index)>>, !kgen.pointer<f32>
  kgen.return
}

// CHECK-LABEL: kgen.func @non_integer_index_array
kgen.func @non_integer_index_array_get() {
  // CHECK: %[[ARRAY_0:.*]] = pop.stack_allocation 1 x scalar<si64> marked
  // CHECK: %[[ARRAY_1:.*]] = pop.stack_allocation 1 x scalar<si64> marked
  %0 = pop.stack_allocation 1 x array<2, scalar<si64>> marked
  // CHECK: %[[LOAD_0:.*]] = pop.load %[[ARRAY_0]] : !kgen.pointer<scalar<si64>>
  // CHECK: %[[LOAD_1:.*]] = pop.load %[[ARRAY_1]] : !kgen.pointer<scalar<si64>>
  %1 = pop.load %0 : !kgen.pointer<array<2, scalar<si64>>>
  // CHECK: %[[ARRAY_CREATE:.*]] = pop.array.create [%[[LOAD_0]], %[[LOAD_1]]] : !pop.array<2, scalar<si64>>
  // CHECK: %[[ARRAY_GET:.*]] = pop.array.get %[[ARRAY_CREATE]][*"i`2x7"] : !pop.array<2, scalar<si64>>
  %2 = pop.array.get %1[*"i`2x7"] : !pop.array<2, scalar<si64>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @unresolved_index_struct_gep<I>
// CHECK-SAME: (%[[ARG0:.*]]: !kgen.struct<(index, index)>)
kgen.generator @unresolved_index_struct_gep<I: index>(%arg0: !kgen.struct<(index, index)>) -> index {
  // CHECK-NEXT: %[[STACK_ALLOC:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  %0 = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK-NEXT: pop.store %[[ARG0]], %[[STACK_ALLOC]]
  pop.store %arg0, %0 : !kgen.pointer<struct<(index, index)>>
  // CHECK-NEXT: %[[GEP:.*]] = kgen.struct.gep %[[STACK_ALLOC]][I]
  %gep = kgen.struct.gep %0[I] : <struct<(index, index)>> -> <index>
  // CHECK-NEXT: %[[LOAD:.*]] = pop.load %[[GEP]] align<8>
  %load = pop.load %gep align<8> : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return %[[LOAD]]
  kgen.return %load : index
}


// CHECK-LABEL: @array_offset
kgen.func @array_offset(%arg1: !pop.array<4, index>, %arg2: index) -> index {
  %0 = kgen.param.constant = <0>
  %array = pop.stack_allocation 1 x array<4, index>
  // Offset from an element should not be touched.
  // CHECK: pop.stack_allocation 1 x array<4, index>
  pop.store %arg1, %array : !kgen.pointer<array<4, index>>
  %gep = pop.array.gep %array[%0] : <array<4, index>>
  // This can access an arbitrary element.
  %offset = pop.offset %gep[%arg2] : !kgen.pointer<index>
  %load = pop.load %offset : !kgen.pointer<index>
  kgen.return %load : index
}

// A struct allocation reached through its leading (offset-zero, nested) element
// GEP chain decomposes; mem2reg then forwards the stored value.
// CHECK-LABEL: @bitcast_to_leading_element
// MEM2REG-LABEL: @bitcast_to_leading_element
// MEM2REG-SAME: (%[[STRUCT:.*]]: !kgen.struct
kgen.func @bitcast_to_leading_element(%arg0: !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>) -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: pop.stack_allocation 1 x pointer<none>
  // CHECK: kgen.return

  // MEM2REG: %[[E0:.*]] = kgen.struct.extract %[[STRUCT]][0]
  // MEM2REG: %[[E1:.*]] = kgen.struct.extract %[[E0]][0]
  // MEM2REG: %[[E2:.*]] = kgen.struct.extract %[[E1]][0]
  // MEM2REG: kgen.return %[[E2]]
  %0 = pop.stack_allocation 1 x !kgen.struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>
  pop.store %arg0, %0 : !kgen.pointer<struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
  %1 = kgen.struct.gep %0[0] : <struct<(struct<(struct<(pointer<none>) memoryOnly>)>)>>
  %2 = kgen.struct.gep %1[0] : <struct<(struct<(pointer<none>) memoryOnly>)>>
  %3 = kgen.struct.gep %2[0] : <struct<(pointer<none>) memoryOnly>>
  %4 = pop.load %3 : !kgen.pointer<pointer<none>>
  kgen.return %4 : !kgen.pointer<none>
}

// The canonical round-trip pun: view a scalar slot as a wrapper aggregate and
// GEP back to it. SROA folds the pun; mem2reg then folds to `return %arg0`.
// CHECK-LABEL: @bitcast_wrapper_roundtrip
// MEM2REG-LABEL: @bitcast_wrapper_roundtrip
// MEM2REG-SAME: (%[[PTR:.*]]: !kgen.pointer<none>)
kgen.func @bitcast_wrapper_roundtrip(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[A:.*]] = pop.stack_allocation 1 x pointer<none>
  // CHECK: pop.store %{{.*}}, %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: %[[L:.*]] = pop.load %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: kgen.return %[[L]]

  // MEM2REG: kgen.return %[[PTR]]
  %index0 = kgen.param.constant = <0>
  %0 = pop.stack_allocation 1 x pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(array<1, pointer<none>>)>>
  %2 = kgen.struct.gep %1[0] : <struct<(array<1, pointer<none>>)>>
  %3 = pop.array.gep %2[%index0] : <array<1, pointer<none>>>
  pop.store %arg0, %3 : !kgen.pointer<pointer<none>>
  %4 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %4 : !kgen.pointer<none>
}

// A bitcast to a type that is not a leading element is a real reinterpretation
// and must block the decomposition.
// CHECK-LABEL: @bitcast_not_leading
kgen.func @bitcast_not_leading(%arg0: !kgen.struct<(index, index)>) -> !kgen.scalar<si64> {
  // CHECK: pop.stack_allocation 1 x struct<(index, index)>
  // CHECK: pop.pointer.bitcast
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)>
  pop.store %arg0, %0 : !kgen.pointer<struct<(index, index)>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<struct<(index, index)>> to !kgen.pointer<scalar<si64>>
  %2 = pop.load %1 : !kgen.pointer<scalar<si64>>
  kgen.return %2 : !kgen.scalar<si64>
}

// An address-space cast of the allocation is not a type pun and must be left
// alone.
// CHECK-LABEL: @bitcast_address_space_cast
kgen.func @bitcast_address_space_cast(%arg0: !kgen.struct<(index, index)>) -> !kgen.struct<(index, index)> {
  // CHECK: %[[ALLOC:.*]] = pop.stack_allocation 1 x struct<(index, index)>
  // CHECK: %[[CAST:.*]] = pop.pointer.bitcast %[[ALLOC]] : !kgen.pointer<struct<(index, index)>> to !kgen.pointer<struct<(index, index)>, 3>
  // CHECK: pop.load %[[CAST]]
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)>
  pop.store %arg0, %0 : !kgen.pointer<struct<(index, index)>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<struct<(index, index)>> to !kgen.pointer<struct<(index, index)>, 3>
  %2 = pop.load %1 : !kgen.pointer<struct<(index, index)>, 3>
  kgen.return %2 : !kgen.struct<(index, index)>
}

// A round-trip pun through struct.gep alone (no array) also folds in place.
// CHECK-LABEL: @bitcast_roundtrip_struct_only
// MEM2REG-LABEL: @bitcast_roundtrip_struct_only
// MEM2REG-SAME: (%[[PTR:.*]]: !kgen.pointer<none>)
kgen.func @bitcast_roundtrip_struct_only(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[A:.*]] = pop.stack_allocation 1 x pointer<none>
  // CHECK: pop.store %{{.*}}, %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: %[[L:.*]] = pop.load %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: kgen.return %[[L]]

  // MEM2REG: kgen.return %[[PTR]]
  %0 = pop.stack_allocation 1 x pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(pointer<none>)>>
  %2 = kgen.struct.gep %1[0] : <struct<(pointer<none>)>>
  pop.store %arg0, %2 : !kgen.pointer<pointer<none>>
  %3 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %3 : !kgen.pointer<none>
}

// A scalar slot bitcast to a *different* type is a real reinterpretation, not a
// round-trip: leave the cast alone (exercises ReplaceStack::canRun rejection).
// CHECK-LABEL: @bitcast_scalar_not_roundtrip
kgen.func @bitcast_scalar_not_roundtrip(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<si32> {
  // CHECK: pop.stack_allocation 1 x scalar<f32>
  // CHECK: pop.pointer.bitcast
  %0 = pop.stack_allocation 1 x scalar<f32>
  pop.store %arg0, %0 : !kgen.pointer<scalar<f32>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<si32>>
  %2 = pop.load %1 : !kgen.pointer<scalar<si32>>
  kgen.return %2 : !kgen.scalar<si32>
}

// If the round-trip pointer escapes (is stored as a value), it is not a pure
// slot access and must not be folded.
// CHECK-LABEL: @bitcast_roundtrip_escapes
kgen.func @bitcast_roundtrip_escapes(%out: !kgen.pointer<pointer<pointer<none>>>) {
  // CHECK: pop.stack_allocation 1 x pointer<none>
  // CHECK: pop.pointer.bitcast
  %0 = pop.stack_allocation 1 x pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<none>> to !kgen.pointer<struct<(pointer<none>)>>
  %2 = kgen.struct.gep %1[0] : <struct<(pointer<none>)>>
  pop.store %2, %out : !kgen.pointer<pointer<pointer<none>>>
  kgen.return
}

// A round-trip pun through array.gep alone folds in place.
// CHECK-LABEL: @bitcast_roundtrip_array_only
// MEM2REG-LABEL: @bitcast_roundtrip_array_only
// MEM2REG-SAME: (%[[PTR:.*]]: !kgen.pointer<none>)
kgen.func @bitcast_roundtrip_array_only(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK-NOT: pop.pointer.bitcast
  // CHECK: %[[A:.*]] = pop.stack_allocation 1 x pointer<none>
  // CHECK: pop.store %{{.*}}, %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: %[[L:.*]] = pop.load %[[A]] : !kgen.pointer<pointer<none>>
  // CHECK: kgen.return %[[L]]

  // MEM2REG: kgen.return %[[PTR]]
  %index0 = kgen.param.constant = <0>
  %0 = pop.stack_allocation 1 x pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<none>> to !kgen.pointer<array<1, pointer<none>>>
  %2 = pop.array.gep %1[%index0] : <array<1, pointer<none>>>
  pop.store %arg0, %2 : !kgen.pointer<pointer<none>>
  %3 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %3 : !kgen.pointer<none>
}

// A GEP with a non-zero index reaches a different offset, not the slot: the
// chain is not a round-trip and must not be folded.
// CHECK-LABEL: @bitcast_roundtrip_nonzero_index
kgen.func @bitcast_roundtrip_nonzero_index(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK: pop.pointer.bitcast
  %index1 = kgen.param.constant = <1>
  %0 = pop.stack_allocation 1 x pointer<none>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<pointer<none>> to !kgen.pointer<array<2, pointer<none>>>
  %2 = pop.array.gep %1[%index1] : <array<2, pointer<none>>>
  pop.store %arg0, %2 : !kgen.pointer<pointer<none>>
  %3 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %3 : !kgen.pointer<none>
}
