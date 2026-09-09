// RUN: kgen-opt %s -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: @loop
kgen.func @loop(%arg0: i32) {
  // CHECK-NEXT: (%arg1 loc({{.*}}) = %arg0 : i32)
  hlcf.loop (%0 = %arg0 : i32) -> () {
    hlcf.break
  }
  kgen.return
}
