// REQUIRES: x86_64-linux

// COM: check mcmodel=medium and large-data-threshold=2 options.
// RUN: kgen -emit=asm --mcmodel=medium --large-data-threshold=2 %s | FileCheck %s --check-prefix=CHECK-MCMODEL-LDTHRESHOLD
// RUN: kgen -emit=object --mcmodel=medium --large-data-threshold=2 %s -o %t
// RUN: llvm-objdump %t -t | FileCheck %s --check-prefix=CHECK-OBJ

// COM: check relocation-model=pic option.
// RUN: kgen -emit=asm --relocation-model=pic --mcmodel=medium --large-data-threshold=2 %s | FileCheck %s --check-prefix=CHECK-PIC

// COM: check relocation-model=static option.
// RUN: kgen -emit=asm --relocation-model=static --mcmodel=medium --large-data-threshold=2 %s | FileCheck %s --check-prefix=CHECK-STATIC

// COM: check that string constant has a GOTOFF relocation in PIC code.
// CHECK-MCMODEL-LDTHRESHOLD: static_string 
// CHECK-MCMODEL-LDTHRESHOLD-SAME: GOTOFF 
// COM: check that string constant is in .lrodata section in the object file.
// (for any data size that's larger than large-data-threshold)
// CHECK-OBJ: .lrodata

// COM: check that PIC code uses GOTOFF for accessing the string constant.
// COM: this is the same as without having "--relocation-model=pic" since this is default value.  
// CHECK-PIC: static_string 
// CHECK-PIC-SAME: GOTOFF 

// COM: check that  static code does not use GOTOFF for accessing the string constant
// CHECK-STATIC: static_string
// CHECK-STATIC-NOT: GOTOFF
kgen.generator export @main() -> !kgen.string {
  %0 = kgen.param.constant : !kgen.string = <"I am a string.">
  kgen.return %0 : !kgen.string
}
