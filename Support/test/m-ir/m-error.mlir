// RUN: support-dialect-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics

// expected-error @below {{zero-width element type unsupported}}
"M"() {a = #M.primitives_array<i0>} : () -> ()

// -----

// expected-error @below {{expected integer, index, or float element type}}
"M"() {a = #M.primitives_array<vector<2xi2>>} : () -> ()

// -----

// expected-error @below {{expected a shaped type}}
"M"() {a = #M.dense_array<1> : i32} : () -> ()

// -----

// expected-error @below {{shaped type must have static shape}}
"M"() {a = #M.dense_array<1> : tensor<*xi32>} : () -> ()

// -----

// expected-error @below {{attribute type indicates 2 elements, but array has 1}}
"M"() {a = #M.dense_array<1> : tensor<2xi32>} : () -> ()

// -----

// expected-error@+1 {{invalid hex string for aligned_bytes}}
"M"() {a = #M.aligned_bytes<"0xg0010204", align 16>} : () -> ()

// -----

// expected-error@+1 {{alignment must be a power of two.}}
"M"() {a = #M.aligned_bytes<"0x01020304", align 15>} : () -> ()

// -----

// expected-error@+1 {{#M.device_spec_collection does not contain a device spec with the host device reference 'foo:0'.}}
"M"() {a = #M.device_spec_collection<host = <"foo", 0>,
                                     devices = [<ref = <"bar", 0>, target = <triple="x86_64-unknown-linux-gnu", arch="znver3", features="+avx2">>]>} : () -> ()

// -----

// expected-error@+1 {{#M.device_spec_collection contains duplicate device specs for the device reference 'cpu:1'.}}
"M"() {a = #M.device_spec_collection<host = <"cpu", 1>,
                                     devices = [<ref = <"cpu", 1>, target = <triple="x86_64-unknown-linux-gnu", arch="znver3", features="+avx2">>,
                                                <ref = <"cpu", 1>, target = <triple="x86_64-unknown-linux-gnu", arch="znver4", features="+avx2">>]>} : () -> ()

// -----

// expected-error@+1 {{invalid #M.inout_sig at operand 3.}}
"M"() {a = #M.inout_sig<"iioxm">} : () -> ()

// -----

// expected-error@+1 {{failed to parse M_TargetInfoAttr parameter 'relocation_model' which is to be a `llvm::Reloc::Model`}}
"M"() {a = #M.target<triple = "", arch = "", relocation_model = "">} : () -> ()

// -----

// expected-error@+1 {{failed to parse M_TargetInfoAttr parameter 'relocation_model' which is to be a `llvm::Reloc::Model`}}
"M"() {a = #M.target<triple = "", arch = "", relocation_model = "ropi-rwpi-fugazi">} : () -> ()
