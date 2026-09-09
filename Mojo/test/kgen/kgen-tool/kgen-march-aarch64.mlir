// RUN: kgen --target-triple=aarch64-unknown-linux -march armv8.2-a %s -elaborate -mcpu=neoverse-n1 -S -o - | FileCheck %s
// RUN: kgen %s -elaborate --target-triple=aarch64-unknown-linux --target-cpu=neoverse-n1 -S -o - | FileCheck %s

// CHECK: triple = "aarch64-unknown-linux", arch = "neoverse-n1", features = "+aes,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+perfmon,+ras,+rcpc,+rdm,+sha2,+spe,+ssbs", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32",
kgen.generator export @main() {
  kgen.return
}
