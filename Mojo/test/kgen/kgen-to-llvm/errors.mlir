// RUN: kgen-opt -lower-kgen-to-llvm -verify-diagnostics -split-input-file %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{failed to convert func signature}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func @unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    kgen.return %arg0 : tensor<4xf32>
  }
}

// -----

// expected-error @below {{could not find an enclosing target specification}}
module {
  kgen.func @no_target() {
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @capturing_parameter_closure(%arg0: !kgen.pointer<string>) capturing -> !kgen.string {
    %0 = pop.load %arg0 : !kgen.pointer<string>
    kgen.return %0 : !kgen.string
  }

  kgen.func @materializing_user() -> !kgen.generator<(!kgen.pointer<string>) capturing -> !kgen.string> {
    // TODO: Duplicate errors are because MLIR auto-legalization stuff is the wrong
    // thing to use for LowerKGENToLLVM.
    // expected-error @below {{capturing closures cannot be materialized as runtime values}}
    %struct = kgen.param.constant: struct<((!kgen.pointer<string>) capturing -> !kgen.string, index)> = <{@capturing_parameter_closure, 8}>
    // expected-error @below {{capturing closures cannot be materialized as runtime values}}
    %0 = kgen.struct.extract %struct[0] : !kgen.struct<((!kgen.pointer<string>) capturing -> !kgen.string, index)>
    kgen.return %0 : !kgen.generator<(!kgen.pointer<string>) capturing -> !kgen.string>
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {
  // expected-error @below {{'"unknown_attribute"' is not defined}}
  // expected-error @below {{failed to legalize operation 'kgen.func' that was explicitly marked illegal}}
  kgen.func export @llvm_metadata() attributes {
    LLVMMetadata = {
      unknown_attribute = #pop.array<256, 1, 4> : !pop.array<3, i32>
    }
  } {
    kgen.return
  }
}
