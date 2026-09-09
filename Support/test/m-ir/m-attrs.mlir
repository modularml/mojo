// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s
// RUN: support-dialect-opt -allow-unregistered-dialect -emit-bytecode %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: ui7: 126, 0, 2, 20
"some.op"() {a = #M.primitives_array<ui7: -2, 0, 2, 20>} : () -> ()

// CHECK: i1: true, false
"some.op"() {a = #M.primitives_array<i1: true, false>} : () -> ()

// CHECK: bf16: 3.1{{[0-9]+}}e+00, 1.7{{[0-9]+}}e+00
"some.op"() {a = #M.primitives_array<bf16: 3.14, 1.73>} : () -> ()

// CHECK: f32: 0xFF800000
"some.op"() {a = #M.primitives_array<f32: 0xFF800000>} : () -> ()

// CHECK: primitives_array<i64>
"some.op"() {a = #M.primitives_array<i64>} : () -> ()

// CHECK: primitives_array<index: -3, 1, 3>
"some.op"() {a = #M.primitives_array<index: -3, 1, 3>} : () -> ()

// CHECK: dense_array<1, 2, 3, 4> : tensor<2x2xi32>
"some.op"() {a = #M.dense_array<1, 2, 3, 4> : tensor<2x2xi32>} : () -> ()

// CHECK: dense_array<0.{{0+}}e+00> : vector<f32>
"some.op"() {a = #M.dense_array<0.> : vector<f32>} : () -> ()

// CHECK: dense_array<65534, 1, 4> : !M.array<3xui16>
"some.op"() {a = #M.dense_array<-2, 1, 4> : !M.array<3xui16>} : () -> ()

// CHECK: dense_array<-3, 1, 3> : !M.array<3xindex>
"some.op"() {a = #M.dense_array<-3, 1, 3> : !M.array<3xindex>} : () -> ()

// CHECK: aligned_bytes<"0x01020304", align 64>
"some.op"() {a = #M.aligned_bytes<"0x01020304", align 64>} : () -> ()

// CHECK: #M.target<triple = "a", arch = "b", features = "+foo", data_layout = "p:64:64-i64:64:64", relocation_model = "pic", simd_bit_width = 128>
"some.op"() {a = #M.target<triple = "a", arch = "b", features = "+foo", data_layout = "p:64:64-i64:64:64", relocation_model = "pic", simd_bit_width = 128>} : () -> ()

// CHECK: #M.target<triple = "a", arch = "b", features = "+foo", data_layout = "p:64:64-i64:64:64", relocation_model = "pic", simd_bit_width = 128, index_bit_width = 64>
"some.op"() {a = #M.target<triple = "a", arch = "b", features = "+foo", data_layout = "p:64:64-i64:64:64", relocation_model = "pic", simd_bit_width = 128, index_bit_width = 64>} : () -> ()

// CHECK: #M.target<triple = "a", arch = "b">
"some.op"() {a = #M.target<triple = "a", arch = "b">} : () -> ()

// A non-default stdlib_plugin round-trips through both the textual and
// bytecode formats.
// CHECK: #M.target<triple = "a", arch = "b", stdlib_plugin = "metal", features = "+foo">
"some.op"() {a = #M.target<triple = "a", arch = "b", stdlib_plugin = "metal", features = "+foo">} : () -> ()

// The default stdlib_plugin ("default") is elided when printing.
// CHECK: #M.target<triple = "a", arch = "b", features = "+foo">
"some.op"() {a = #M.target<triple = "a", arch = "b", stdlib_plugin = "default", features = "+foo">} : () -> ()

// CHECK: #M.device_ref<"foo", 3>
"some.op"() {a = #M.device_ref<"foo", 3>} : () -> ()

// CHECK: #M.device_spec<ref = <"gpu", 1>, target = <triple = "nvptx64-nvidia-cuda", arch = "sm_80">>
"some.op"() {a = #M.device_spec<ref = <"gpu", 1>, target = <triple="nvptx64-nvidia-cuda", arch = "sm_80">>} : () -> ()

// CHECK: #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"gpu", 0>, target = <triple = "nvptx64-nvidia-cuda", arch = "sm_80">>, <ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "znver3", features = "+avx2">>]>
"some.op"() {a = #M.device_spec_collection<host = <"cpu", 0>,
                   devices = [<ref = <"gpu", 0>, target = <triple = "nvptx64-nvidia-cuda", arch = "sm_80">>,
                              <ref = <"cpu", 0>, target = <triple="x86_64-unknown-linux-gnu", arch="znver3", features="+avx2">>]>} : () -> ()

// CHECK: #M<multiline["a", "b", "c"]>
"some.op"() {a = #M<multiline["a", "b", "c"]>} : () -> ()

// CHECK: #M.inout_sig<"ii.mo">
"some.op"() {a = #M.inout_sig<"ii.mo">} : () -> ()

// CHECK: #M<symbols[]>
"some.op"() {a = #M<symbols[]>} : () -> ()

// CHECK: #M<symbols[@m_symbols_bytecode::@inner]>
module @m_symbols_bytecode {
  "some.op"() {a = #M<symbols[@m_symbols_bytecode::@inner]>} : () -> ()
  func.func @inner() { return }
}
