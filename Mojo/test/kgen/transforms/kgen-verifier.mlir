// RUN: kgen-opt --split-input-file -kgen-verifier %s --verify-diagnostics

// Non-positive, non-power-of-two SIMD lengths must not reach codegen
module {
  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<3, f32>'}}
  kgen.func @simd_non_pow2(%arg0: !kgen.simd<3, f32>) {
    kgen.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<0, f32>'}}
  kgen.func @simd_zero(%arg0: !kgen.simd<0, f32>) {
    kgen.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<-1, f32>'}}
  kgen.func @simd_neg(%arg0: !kgen.simd<-1, f32>) {
    kgen.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<65536, f32>'}}
  kgen.func @simd_too_large(%arg0: !kgen.simd<65536, f32>) {
    kgen.return
  }

  kgen.func @simd_pow2_ok(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.scalar<index>) {
    kgen.return
  }
}

