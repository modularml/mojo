// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// This shouldn't crash.
// https://github.com/modularml/modular/issues/2480

lit.struct.decl @FooStruct {
  lit.struct.field a : index
  lit.struct.field b : index
}

// CHECK-LABEL: lit.fn @struct_extract_fold_insert
lit.fn @struct_extract_fold_insert(%struct0: !lit.struct<@FooStruct>) -> index {
  // CHECK-NOT: lit.struct.insert
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %idx10
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct1[a] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.fn @struct_extract_no_fold_insert
lit.fn @struct_extract_no_fold_insert(%struct0: !lit.struct<@FooStruct>) -> index {
  // CHECK: lit.struct.insert
  // CHECK-NEXT: lit.struct.extract
  // CHECK-NEXT: kgen.return
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct1[b] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

lit.struct.decl @Pair register_passable_trivial {
  lit.struct.field first : !lit.struct<@Int>
  lit.struct.field second : !lit.struct<@Int>
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}
