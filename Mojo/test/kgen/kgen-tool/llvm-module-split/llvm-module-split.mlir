// RUN: kgen-translate %s --mlir-to-llvmir -o %t
// RUN: llvm-module-split %t --per-func | FileCheck %s

llvm.mlir.global_ctors ctors = [@KGEN_EE_JIT_GlobalConstructor], priorities = [0 : i32], data = [#llvm.zero]
llvm.mlir.global_dtors dtors = [@KGEN_EE_JIT_GlobalDestructor], priorities = [0 : i32], data = [#llvm.zero]

llvm.func weak @KGEN_EE_JIT_GlobalConstructor() {
  llvm.return
}

llvm.func weak @KGEN_EE_JIT_GlobalDestructor() {
  llvm.return
}
llvm.func @f(%arg0: i64, %arg1: i64) -> i64 {
  %0 = llvm.add %arg0, %arg1  : i64
  llvm.return %0 : i64
}
llvm.func @g(%arg0: i64, %arg1: i64) -> i64 {
  %0 = llvm.mul %arg0, %arg1  : i64
  llvm.return %0 : i64
}
llvm.func @h(%arg0: i64, %arg1: i64) -> i64 {
  %0 = llvm.sub %arg0, %arg1  : i64
  llvm.return %0 : i64
}

// COM: check splitting with modules for each single function (except for coroutine exceptions.)
// CHECK: [LLVM Module Split: submodule 0]
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @KGEN_EE_JIT_GlobalConstructor, ptr null }]
// CHECK: [LLVM Module Split: submodule 1]
// CHECK: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @KGEN_EE_JIT_GlobalDestructor, ptr null }]
// CHECK: [LLVM Module Split: submodule 2]
// CHECK: define weak void @KGEN_EE_JIT_GlobalConstructor()
// CHECK: [LLVM Module Split: submodule 3]
// CHECK: define weak void @KGEN_EE_JIT_GlobalDestructor()
// CHECK: [LLVM Module Split: submodule 4]
// CHECK: define i64 @f(i64 %0, i64 %1)
// CHECK: [LLVM Module Split: submodule 5]
// CHECK: define i64 @g(i64 %0, i64 %1)
// CHECK: [LLVM Module Split: submodule 6]
// CHECK: define i64 @h(i64 %0, i64 %1)
