// RUN: system-info --query=target-triple | FileCheck %s --check-prefix=CHECK-TRIPLE
// RUN: system-info --query=os | FileCheck %s --check-prefix=CHECK-OS
// RUN: system-info --query=arch | FileCheck %s --check-prefix=CHECK-ARCH
// RUN: system-info --query=simd-bitwidth | FileCheck %s --check-prefix=CHECK-SIMD-BITWIDTH
// RUN: system-info --query=features | FileCheck %s --check-prefix=CHECK-FEATURES
// RUN: system-info --query=core-count | FileCheck %s --check-prefix=CHECK-CORE-COUNT
// RUN: system-info --query=l1-cache-size | FileCheck %s --check-prefix=CHECK-L1-CACHE-SIZE
// RUN: system-info --query=l2-cache-size | FileCheck %s --check-prefix=CHECK-L2-CACHE-SIZE
// RUN: system-info --query=l3-cache-size | FileCheck %s --check-prefix=CHECK-L3-CACHE-SIZE
// RUN: system-info --query=l4-cache-size | FileCheck %s --check-prefix=CHECK-L4-CACHE-SIZE

// RUN: system-info | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: system-info --format json | FileCheck %s --check-prefix=CHECK-JSON
// RUN: system-info --format yaml | FileCheck %s --check-prefix=CHECK-YAML
// RUN: system-info --format json -o out.json && cat out.json | FileCheck %s --check-prefix=CHECK-JSON-OUTPUT
// RUN: system-info --format yaml -o out.txt  && cat out.txt  | FileCheck %s --check-prefix=CHECK-YAML-OUTPUT
// RUN: system-info --format json --query os --query features --query affinities | FileCheck %s --check-prefix=CHECK-JSON-QUERY


// CHECK-TRIPLE: {{.*}}
// CHECK-OS: {{.*}}
// CHECK-ARCH: {{.*}}
// CHECK-SIMD-BITWIDTH: {{.*}}
// CHECK-FEATURES: {{.*}}
// CHECK-CORE-COUNT: {{[0-9]+}}
// CHECK-L1-CACHE-SIZE: {{[0-9]+}}
// CHECK-L2-CACHE-SIZE: {{[0-9]+}}
// CHECK-L3-CACHE-SIZE: {{[0-9]+}}
// CHECK-L4-CACHE-SIZE: {{[0-9]+}}
// CHECK-DEFAULT: {{.*}}
// CHECK-JSON: {{.*}}
// CHECK-YAML: {{.*}}
// CHECK-JSON-OUTPUT: {{.*}}
// CHECK-YAML-OUTPUT: {{.*}}
// CHECK-JSON-QUERY: {{.*}}
