// RUN: kgen %s -execute -func="my_exported_kernel:f32(f32)" | FileCheck %s -check-prefix=EXEC
// RUN: kgen %s --emit=object -o %t_my_kernel.o
// COM: Check the object file.
// RUN: llvm-objdump %t_my_kernel.o -t | FileCheck %s -check-prefix=OBJ
// COM: Check the header file.
// RUN: kgen %s -emit=header | FileCheck %s -check-prefix=HDR

kgen.func export @my_exported_kernel(%arg0: f32) cabi -> f32 {
  kgen.return %arg0 : f32
}

kgen.func @noop() {
  kgen.return
}


// EXEC: --- 'my_exported_kernel' returned 1.0

// OBJ-LABEL: {{.*}}kgen-exec-obj.mlir.{{.*}}.o:
// OBJ-DAG: my_exported_kernel

// HDR-LABEL: extern float my_exported_kernel(float);
