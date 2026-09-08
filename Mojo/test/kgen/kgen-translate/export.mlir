// RUN: kgen-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @nothing
llvm.func @nothing() {
  llvm.return
}
