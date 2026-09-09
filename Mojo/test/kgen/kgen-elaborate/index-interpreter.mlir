// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967295, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}
}

// -----

// COM: Cmp falls back to folder when target is not specified

kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967294, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}

// -----

// COM: Subtraction

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @sub(%arg0: index, %arg1: index) -> index {
  %0 = index.sub %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @sub, 4294967295, 5)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant = <4294967290>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Shift left

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @shl(%arg0: index, %arg1: index) -> index {
  %0 = index.shl %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @shl, 1, 63)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-9223372036854775808>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Logical shift right

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.shru %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -1, 4)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <1152921504606846975>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: Arithmetic shift right

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.shrs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -1, 4)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-1>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: And

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.and %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 15, 5)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <5>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.and %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -0x80000000, 0x1f000000)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <0>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.and %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 0x80000000, 0x80000000)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <2147483648>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <5>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed ceil div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.ceildivs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.divu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed div

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.divs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned max

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.maxu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <32>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed max

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.maxs %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned min

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.minu %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <7>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed min

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.mins %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-32>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: multiplication

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.mul %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <224>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: or

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.or %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <39>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: unsigned rem

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.rems %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, 32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM: signed rem

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @test(%arg0: index, %arg1: index) -> index {
  %0 = index.rems %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  kgen.param.declare value : index = <apply(:(index, index) -> index @test, -32, 7)>
  // CHECK-NEXT: [[V0:%.*]] =  kgen.param.constant = <-4>
  %0 = kgen.param.constant: index = <value>
  // CHECK-NEXT: kgen.return [[V0]] : index
  kgen.return %0 : index
}
}

// -----

// COM Testing interpreter on 32-bit targets

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {

// COM: pop.div with uindex on 32 bit

kgen.generator @uindex_div_32(%arg0: !kgen.scalar<uindex>, %arg1: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %0 = pop.div %arg0, %arg1 : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// CHECK-LABEL: kgen.func export @testDiv
kgen.generator export @testDiv() -> !kgen.scalar<uindex> {
  kgen.param.declare value : !kgen.scalar<uindex> = <apply(:(!kgen.scalar<uindex>, !kgen.scalar<uindex>) -> !kgen.scalar<uindex> @uindex_div_32, 18446744073709551615, 10)>
  // COM: should truncate 18446744073709551615 to 4294967295, then divide
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<uindex> = <429496729>
  %0 = kgen.param.constant: !kgen.scalar<uindex> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// COM: pop.rem with uindex on 32 bit

kgen.generator @uindex_rem_32(%arg0: !kgen.scalar<uindex>, %arg1: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %0 = pop.rem %arg0, %arg1 : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// CHECK-LABEL: kgen.func export @testRem
kgen.generator export @testRem() -> !kgen.scalar<uindex> {
  kgen.param.declare value : !kgen.scalar<uindex> = <apply(:(!kgen.scalar<uindex>, !kgen.scalar<uindex>) -> !kgen.scalar<uindex> @uindex_rem_32, 18446744073709551615, 10)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<uindex> = <5>
  %0 = kgen.param.constant: !kgen.scalar<uindex> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// COM: pop.cmp with index on 32-bit

kgen.generator @pop_cmp_lt_32(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<index>) -> !kgen.scalar<bool> {
  %0 = pop.cmp lt(%arg0, %arg1) : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func export @testCmp
kgen.generator export @testCmp() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<bool> @pop_cmp_lt_32, 3000000000, 0)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}

// COM: llvm.ctlz with index on 32-bit

kgen.generator @call_llvm_ctlz(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<bool>) -> !kgen.scalar<index> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.ctlz", (%arg0, %arg1) : (!kgen.scalar<index>, !kgen.scalar<bool>) -> !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testCTLZ
kgen.generator export @testCTLZ() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<bool>) -> !kgen.scalar<index> @call_llvm_ctlz, 16, false)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <27>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// COM: pop.abs with index on 32-bit

kgen.generator @abs_index(%arg0: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.abs %arg0 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @abs_uindex(%arg0: !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> {
  %0 = pop.abs %arg0 : !kgen.simd<4, uindex>
  kgen.return %0 : !kgen.simd<4, uindex>
}

// CHECK-LABEL: kgen.func export @testAbs
kgen.generator export @testAbs() -> !kgen.simd<4, index> {
  // COM: 9223372036854775807 truncates to int32 -1 -> 1
  // COM: -9223372036854775808 truncates to int32 0 -> 0
  // CHECK: <<7, 7, 1, 0>>
  kgen.param.declare S0: simd<4, index> = <<7, -7, 9223372036854775807, -9223372036854775808>>
  kgen.param.declare S1: simd<4, index> = <apply(:(!kgen.simd<4, index>) -> !kgen.simd<4, index> @abs_index, S0)>
  %1 = kgen.param.constant: !kgen.simd<4, index> = <S1>

  // COM: Unsigned parameters are just returned, hence we don't see truncation to 32 bits here
  // CHECK: <<7, 0, 18446744073709551615, 9223372036854775807>>
  kgen.param.declare U0: simd<4, uindex> = <<7, 0, 18446744073709551615, 9223372036854775807>>
  kgen.param.declare U1: simd<4, uindex> = <apply(:(!kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> @abs_uindex, U0)>
  %2 = kgen.param.constant: !kgen.simd<4, uindex> = <U1>

  kgen.return %1 : !kgen.simd<4, index>
}

// COM: pop.floordiv with index on 32-bit

kgen.generator @floordiv_index(%arg0: !kgen.simd<4, index>, %arg1: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @floordiv_uindex(%arg0: !kgen.simd<4, uindex>, %arg1: !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<4, uindex>
  kgen.return %0 : !kgen.simd<4, uindex>
}

// CHECK-LABEL: kgen.func export @testFloordiv
kgen.generator export @testFloordiv() -> !kgen.simd<4, index> {
  kgen.param.declare S0: simd<4, index> = <<7, 7, 9223372036854775807, 9223372036854775807>>
  kgen.param.declare S1: simd<4, index> = <<3, -3, 10, -10>>
  kgen.param.declare S2: simd<4, index> = <apply(:(!kgen.simd<4, index>, !kgen.simd<4, index>) -> !kgen.simd<4, index> @floordiv_index, S0, S1)>
  // CHECK: <<2, -3, -1, 0>>
  %1 = kgen.param.constant: !kgen.simd<4, index> = <S2>

  kgen.param.declare U0: simd<4, uindex> = <<7, 7, 18446744073709551615, 18446744073709551615>>
  kgen.param.declare U1: simd<4, uindex> = <<3, 7, 10, 18446744073709551615>>
  kgen.param.declare U2: simd<4, uindex> = <apply(:(!kgen.simd<4, uindex>, !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> @floordiv_uindex, U0, U1)>
  // CHECK: <<2, 1, 429496729, 1>>
  %2 = kgen.param.constant: !kgen.simd<4, uindex> = <U2>

  kgen.return %1 : !kgen.simd<4, index>
}

// COM: pop.shr with index on 32-bit

kgen.generator @shr_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
  %0 = pop.shr %arg0, %arg1 : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShr
kgen.generator export @testShr() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <248>
  kgen.param.declare S1: scalar<index> = <4>
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shr_index, S0, S1)>
  // CHECK: = <15>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  kgen.return %0 : !kgen.scalar<index>
}

// COM: pop.shl with index on 32-bit

kgen.generator @shl_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
  %0 = pop.shl %arg0, %arg1 : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShl
kgen.generator export @testShl() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <1>
  kgen.param.declare S1: scalar<index> = <5>
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shl_index, S0, S1)>
  // CHECK: = <32>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  kgen.return %0 : !kgen.scalar<index>
}

}

// -----

// COM Testing interpreter on 64-bit targets

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// COM: pop.div with uindex on 64 bit

kgen.generator @uindex_div_64(%arg0: !kgen.scalar<uindex>, %arg1: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %0 = pop.div %arg0, %arg1 : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// CHECK-LABEL: kgen.func export @testDiv
kgen.generator export @testDiv() -> !kgen.scalar<uindex> {
  kgen.param.declare value : !kgen.scalar<uindex> = <apply(:(!kgen.scalar<uindex>, !kgen.scalar<uindex>) -> !kgen.scalar<uindex> @uindex_div_64, 18446744073709551615, 10)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<uindex> = <1844674407370955161>
  %0 = kgen.param.constant: !kgen.scalar<uindex> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// COM: pop.rem with uindex on 64 bit

kgen.generator @uindex_rem_64(%arg0: !kgen.scalar<uindex>, %arg1: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %0 = pop.rem %arg0, %arg1 : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}

// CHECK-LABEL: kgen.func export @testRem
kgen.generator export @testRem() -> !kgen.scalar<uindex> {
  kgen.param.declare value : !kgen.scalar<uindex> = <apply(:(!kgen.scalar<uindex>, !kgen.scalar<uindex>) -> !kgen.scalar<uindex> @uindex_rem_64, 18446744073709551615, 10)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<uindex> = <5>
  %0 = kgen.param.constant: !kgen.scalar<uindex> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<uindex>
  kgen.return %0 : !kgen.scalar<uindex>
}


// COM: pop.cmp with index on 64-bit

kgen.generator @pop_cmp_lt_64(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<index>) -> !kgen.scalar<bool> {
  %0 = pop.cmp lt(%arg0, %arg1) : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func export @testCmp
kgen.generator export @testCmp() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<bool> @pop_cmp_lt_64, 3000000000, 0)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}

// COM: llvm.ctlz with index on 64-bit

kgen.generator @call_llvm_ctlz(%arg0: !kgen.scalar<index>, %arg1: !kgen.scalar<bool>) -> !kgen.scalar<index> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.ctlz", (%arg0, %arg1) : (!kgen.scalar<index>, !kgen.scalar<bool>) -> !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testCTLZ
kgen.generator export @testCTLZ() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<bool>) -> !kgen.scalar<index> @call_llvm_ctlz, 16, false)>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <59>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// COM: pop.abs with index on 64-bit

kgen.generator @abs_index(%arg0: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.abs %arg0 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @abs_uindex(%arg0: !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> {
  %0 = pop.abs %arg0 : !kgen.simd<4, uindex>
  kgen.return %0 : !kgen.simd<4, uindex>
}

// CHECK-LABEL: kgen.func export @testAbs
kgen.generator export @testAbs() -> !kgen.simd<4, index> {
  // COM: abs(INT64_MIN) returns INT64_MIN again
  // CHECK: <<7, 7, 9223372036854775807, -9223372036854775808>>
  kgen.param.declare S0: simd<4, index> = <<7, -7, 9223372036854775807, -9223372036854775808>>
  kgen.param.declare S1: simd<4, index> = <apply(:(!kgen.simd<4, index>) -> !kgen.simd<4, index> @abs_index, S0)>
  %1 = kgen.param.constant: !kgen.simd<4, index> = <S1>

  kgen.param.declare U0: simd<4, uindex> = <<7, 0, 18446744073709551615, 9223372036854775807>>
  kgen.param.declare U1: simd<4, uindex> = <apply(:(!kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> @abs_uindex, U0)>
  // CHECK: <<7, 0, 18446744073709551615, 9223372036854775807>>
  %2 = kgen.param.constant: !kgen.simd<4, uindex> = <U1>

  kgen.return %1 : !kgen.simd<4, index>
}

// COM: pop.floordiv with index on 64-bit

kgen.generator @floordiv_index(%arg0: !kgen.simd<4, index>, %arg1: !kgen.simd<4, index>) -> !kgen.simd<4, index> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<4, index>
  kgen.return %0 : !kgen.simd<4, index>
}

kgen.generator @floordiv_uindex(%arg0: !kgen.simd<4, uindex>, %arg1: !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> {
  %0 = pop.floordiv %arg0, %arg1 : !kgen.simd<4, uindex>
  kgen.return %0 : !kgen.simd<4, uindex>
}

// CHECK-LABEL: kgen.func export @testFloordiv
kgen.generator export @testFloordiv() -> !kgen.simd<4, index> {
  kgen.param.declare S0: simd<4, index> = <<7, 7, 9223372036854775807, 9223372036854775807>>
  kgen.param.declare S1: simd<4, index> = <<3, -3, 10, -10>>
  kgen.param.declare S2: simd<4, index> = <apply(:(!kgen.simd<4, index>, !kgen.simd<4, index>) -> !kgen.simd<4, index> @floordiv_index, S0, S1)>
  // CHECK: <<2, -3, 922337203685477580, -922337203685477581>>
  %1 = kgen.param.constant: !kgen.simd<4, index> = <S2>

  kgen.param.declare U0: simd<4, uindex> = <<7, 7, 18446744073709551615, 18446744073709551615>>
  kgen.param.declare U1: simd<4, uindex> = <<3, 7, 10, 18446744073709551615>>
  kgen.param.declare U2: simd<4, uindex> = <apply(:(!kgen.simd<4, uindex>, !kgen.simd<4, uindex>) -> !kgen.simd<4, uindex> @floordiv_uindex, U0, U1)>
  // CHECK: <<2, 1, 1844674407370955161, 1>>
  %2 = kgen.param.constant: !kgen.simd<4, uindex> = <U2>

  kgen.return %1 : !kgen.simd<4, index>
}

// COM: pop.shr with index on 64-bit

kgen.generator @shr_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
  %0 = pop.shr %arg0, %arg1 : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShr
kgen.generator export @testShr() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <3>
  kgen.param.declare S1: scalar<index> = <63>
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shr_index, S0, S1)>
  // CHECK: = <0>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  kgen.return %0 : !kgen.scalar<index>
}

// COM: pop.shl with index on 64-bit

kgen.generator @shl_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
  %0 = pop.shl %arg0, %arg1 : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShl
kgen.generator export @testShl() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <1>
  kgen.param.declare S1: scalar<index> = <5>
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shl_index, S0, S1)>
  // CHECK: = <32>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  kgen.return %0 : !kgen.scalar<index>
}

}
