// RUN: kgen %s --emit=object -o %t_my_kernel.o --save-temps --temps-dir=%t_temps
// COM: Check the save tmp files.
// RUN: find $(dirname %t_temps) -type f -name "*.s" -print -quit | xargs cat | FileCheck %s -check-prefix=ASM
// RUN: find $(dirname %t_temps) -type f -name "*.pre-split.*.ll" -print -quit | xargs cat | FileCheck %s -check-prefix=PRESPLIT

kgen.func export @my_exported_kernel(%arg0: f32) cabi -> f32 {
  kgen.return %arg0 : f32
}

kgen.func export @noop() {
  kgen.return
}

// ASM-DAG: .section

// PRESPLIT-LABEL: ; ModuleID = 'kgen-save-temps.mlir'
// PRESPLIT-DAG: define dso_local float @my_exported_kernel(float noundef %0) #0
// PRESPLIT-DAG: define dso_local void @noop() #0
