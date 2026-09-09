// RUN: kgen %s -elaborate -S -o - --target-triple=x86_64-unknown-linux -march=skylake-avx512+crc64 -mtune=skylake-avx512 | FileCheck %s --check-prefixes=CHECK,CHECK-TUNE
// RUN: kgen %s -elaborate -S -o - --target-triple=x86_64-unknown-linux --target-cpu=skylake-avx512 | FileCheck %s --check-prefixes=CHECK,CHECK-TARGET

// CHECK: M.target_info = #M.target<triple = "x86_64-unknown-linux", arch = "skylake-avx512",
// CHECK-TUNE-SAME: features = "+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+crc64,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves",
// CHECK-TARGET-SAME: features = "+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves",
// CHECK-SAME: data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", relocation_model = "pic", simd_bit_width = 512,
// CHECK-TUNE-SAME: tune_cpu = "skylake-avx512"

kgen.generator export @main() {
  kgen.return
}
