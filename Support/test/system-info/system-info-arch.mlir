// REQUIRES: system-darwin

// RUN: system-info --march arm64 --mcpu apple-m1 | FileCheck %s --check-prefix=CHECK-M1
// RUN: system-info --march arm64 --mcpu apple-m2 | FileCheck %s --check-prefix=CHECK-M2

// CHECK-M1: target-triple: aarch64-apple-darwin
// CHECK-M1: arch: apple-m1
// CHECK-M1: features: aes, altnzcv, ccdp, complxnum, crc, dotprod, fp-armv8, fp16fml, fptoint, fullfp16, jsconv, lse, neon, pauth, perfmon, predres, ras, rcpc, rdm, sb, sha2, sha3, specrestrict, ssbs

// CHECK-M2: target-triple: aarch64-apple-darwin
// CHECK-M2 os: macosx
// CHECK-M2: arch: apple-m2
// CHECK-M2: features: aes, bf16, complxnum, crc, dotprod, fp-armv8, fp16fml, fpac, fullfp16, i8mm, jsconv, lse, neon, pauth, perfmon, ras, rcpc, rdm, sha2, sha3, ssbs
