// RUN: not kgen %s -emit-llvm 2>&1 | FileCheck %s -DOPT=-emit-llvm -DREPL=--emit=llvm
// RUN: not kgen %s -emit-llvm=opt 2>&1 | FileCheck %s -DOPT=-emit-llvm=opt -DREPL=--emit=llvm=opt
// RUN: not kgen %s -emit-asm 2>&1 | FileCheck %s -DOPT=-emit-asm -DREPL=--emit=asm
// RUN: not kgen %s -emit-asm-verbose 2>&1 | FileCheck %s -DOPT=-emit-asm-verbose -DREPL=--emit=asm-verbose
// RUN: not kgen %s -emit-header 2>&1 | FileCheck %s -DOPT=-emit-header -DREPL=--emit=header

// CHECK: Unknown command line argument '[[OPT]]'.
// CHECK-NEXT: Did you mean '[[REPL]]'?

kgen.generator export @some_func(%arg0: f32) -> (f32, f32) {
  kgen.return %arg0, %arg0 : f32, f32
}
