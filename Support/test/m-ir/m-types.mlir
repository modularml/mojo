// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s
// RUN: support-dialect-opt -allow-unregistered-dialect -emit-bytecode %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: !M.array<32xf32>
"some.op"() {m = !M.array<32xf32>} : () -> ()

// CHECK: !M.array<256xf64>
"some.op"() {m = !M.array<256xf64>} : () -> ()

// CHECK: !M.aligned_bytes<4, align 64>
"some.op"() {m = !M.aligned_bytes<4, align 64>} : () -> ()
