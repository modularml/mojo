// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Single-element struct flattening
//===----------------------------------------------------------------------===//
// Tests that single-element register_passable structs are properly flattened
// and that struct field access becomes identity operations.

// Single-element struct with index field
lit.struct.decl @SingleElementIndex register_passable {
  lit.struct.field value : index
}

// Single-element struct with i32 field
lit.struct.decl @SingleElementI32 register_passable {
  lit.struct.field value : i32
}

// Multi-element struct for comparison.
lit.struct.decl @MultiElement register_passable {
  lit.struct.field first : index
  lit.struct.field second : index
}

// CHECK-LABEL: kgen.generator @single_element_index_field_access
// For single-element structs, the struct is flattened to its element type.
// Field access should become a no-op (no struct.gep needed).
// CHECK-SAME: (%arg0: !kgen.pointer<index>)
// CHECK-NEXT: kgen.return
lit.fn @single_element_index_field_access<l: !lit.origin<true>>
    (%ptr: !lit.ref<@SingleElementIndex, mut l>) {
  // This should be lowered without creating a struct.gep since the struct
  // is flattened to just `index`.
  %0 = lit.ref.struct.ger %ptr[value] : <@SingleElementIndex, mut l> -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @single_element_i32_field_access
// CHECK-SAME: (%arg0: !kgen.pointer<i32>)
// CHECK-NEXT: kgen.return
lit.fn @single_element_i32_field_access<l: !lit.origin<true>>
    (%ptr: !lit.ref<@SingleElementI32, mut l>) {
  %0 = lit.ref.struct.ger %ptr[value] : <@SingleElementI32, mut l> -> i32
  kgen.return
}

// CHECK-LABEL: kgen.generator @multi_element_field_access
// Multi-element structs should still use struct.gep.
// CHECK-SAME: (%arg0: !kgen.pointer<struct<(index, index)>>)
// CHECK-NEXT: kgen.struct.gep %arg0[1]
lit.fn @multi_element_field_access<l: !lit.origin<true>>
    (%ptr: !lit.ref<@MultiElement, mut l>) {
  %0 = lit.ref.struct.ger %ptr[second] : <@MultiElement, mut l> -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @single_element_index_access
// Index access (used by __struct_field_ref) should also work for single-element.
// CHECK-SAME: (%arg0: !kgen.pointer<index>)
// CHECK-NEXT: kgen.return
lit.fn @single_element_index_access<l: !lit.origin<true>>
    (%ptr: !lit.ref<@SingleElementIndex, mut l>) {
  %0 = lit.ref.struct.ger %ptr[idx 0] : <@SingleElementIndex, mut l> -> <index, mut l>
  kgen.return
}

// CHECK-LABEL: kgen.generator @multi_element_index_access
// Index access on multi-element structs should still use struct.gep.
// CHECK-SAME: (%arg0: !kgen.pointer<struct<(index, index)>>)
// CHECK-NEXT: kgen.struct.gep %arg0[0]
lit.fn @multi_element_index_access<l: !lit.origin<true>>
    (%ptr: !lit.ref<@MultiElement, mut l>) {
  %0 = lit.ref.struct.ger %ptr[idx 0] : <@MultiElement, mut l> -> <index, mut l>
  kgen.return
}
