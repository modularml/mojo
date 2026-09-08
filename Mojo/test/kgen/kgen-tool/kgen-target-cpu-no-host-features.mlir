// REQUIRES: x86_64-linux

// Test that --target-cpu alone (no --target-triple) derives features from the
// specified CPU, not the host. This is how kgen_kernel Bazel rules invoke kgen.
// RUN: kgen %s -elaborate -S -o - --target-cpu=x86-64-v3 | FileCheck %s

// CHECK: arch = "x86-64-v3"
// CHECK-NOT: avx512
// CHECK: simd_bit_width = 256

kgen.generator export @main() {
  kgen.return
}
