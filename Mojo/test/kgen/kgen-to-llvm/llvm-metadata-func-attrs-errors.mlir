// RUN: kgen-opt -lower-kgen-to-llvm -verify-diagnostics -split-input-file %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.target_cpu' is not allowed to be set via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @disallowed_target_cpu() attributes {
    LLVMMetadata = {llvm.target_cpu = "znver4"}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.tune_cpu' is not allowed to be set via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @disallowed_tune_cpu() attributes {
    LLVMMetadata = {llvm.tune_cpu = "znver4"}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.target_features' is not allowed to be set via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @disallowed_target_features() attributes {
    LLVMMetadata = {llvm.target_features = "+avx2,+fma"}
  } { kgen.return }
}

// -----

// `dso_local` is set by LLVMFuncOp's builder from linkage; reject overrides.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.dso_local' is not allowed to be set via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @disallowed_dso_local() attributes {
    LLVMMetadata = {llvm.dso_local = unit}
  } { kgen.return }
}

// -----

// Dashed-name LLVM attrs that don't yet have a dispatch entry are rejected
// with a "temporarily not supported" message.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.patchable_function' is temporarily not supported via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @unsupported_patchable_function() attributes {
    LLVMMetadata = {llvm.patchable_function = "prologue-short-redirect"}
  } { kgen.return }
}

// -----

// Cover a second unsupported entry that takes a unit value to exercise a
// different input shape.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.no_jump_tables' is temporarily not supported via @__llvm_metadata}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @unsupported_no_jump_tables() attributes {
    LLVMMetadata = {llvm.no_jump_tables = unit}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{invalid 'llvm.frame_pointer' value 'bogus'; expected "none", "non-leaf", "all", or "reserved"}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @bad_frame_pointer() attributes {
    LLVMMetadata = {llvm.frame_pointer = "bogus"}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.always_inline' expects no value (unit flag)}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @bad_unit_value() attributes {
    LLVMMetadata = {llvm.always_inline = "should-be-unit"}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.section' expects a string value}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @bad_string_value() attributes {
    LLVMMetadata = {llvm.section = 42 : i32}
  } { kgen.return }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{'llvm.vscale_range' expects a 2-element array (min, max)}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func export @bad_vscale_arity() attributes {
    LLVMMetadata = {llvm.vscale_range = #pop.array<1, 2, 3> : !pop.array<3, i32>}
  } { kgen.return }
}
