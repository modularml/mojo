// RUN: kgen-opt %s -lower-lit -split-input-file -verify-parameters \
// RUN:   -kgen-print-inline-type-values | FileCheck %s

// Struct layouts whose recursion is broken by an indirection.

//===----------------------------------------------------------------------===//
// The wrapper is the immediate field type.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Ptr<:type !lit.struct<@Node>>>
}

// -----

//===----------------------------------------------------------------------===//
// A non-parametric wrapper behind an aggregate.
//===----------------------------------------------------------------------===//

lit.struct.decl @PtrToNode register_passable {
  lit.struct.field address : !kgen.pointer<:type !lit.struct<@Node>>
}

lit.struct.decl @Wrap<ty: type> register_passable {
  lit.struct.field x : !kgen.struct<(ty)>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Wrap<:type !lit.struct<@PtrToNode>>>
}

// -----

//===----------------------------------------------------------------------===//
// A by-value wrapper around the pointer wrapper. The parameter is substituted
// as a type, so the layout completes without needing a value form.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Box<ty: type> register_passable {
  lit.struct.field val : !kgen.param<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next :
      !lit.struct<@Box<:type !lit.struct<@Ptr<:type !lit.struct<@Node>>>>>
}

// -----

//===----------------------------------------------------------------------===//
// An array of the pointer wrapper. `!kgen.array` holds its element as a type,
// not as a type value, so nothing demands a value form here.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.array<2, !lit.struct<@Ptr<:type !lit.struct<@Node>>>>
}

// -----

//===----------------------------------------------------------------------===//
// A longer chain of memory-only, multi-field wrappers
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Box<ty: type> {
  lit.struct.field val : !kgen.param<ty>
  lit.struct.field tag : !kgen.scalar<index>
}

lit.struct.decl @Outer<ty: type> {
  lit.struct.field inner : !lit.struct<@Box<:type ty>>
  lit.struct.field tag : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next :
      !lit.struct<@Outer<:type !lit.struct<@Ptr<:type !lit.struct<@Node>>>>>
}

// -----

//===----------------------------------------------------------------------===//
// A parametric expression referring to the pointer wrapper. `get_alignof`
// stays unfolded through this pass - the target is still symbolic - which is
// why an unresolved size or alignment expression is not a problem here.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Aligned register_passable attributes {
  minAlignment = #kgen.param.expr<get_alignof,
    #kgen.type<!lit.struct<@Ptr<:type !kgen.scalar<index>>>> : !kgen.type,
    #kgen.param.expr<current_target> : !kgen.target> : index
} {
  lit.struct.field a : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.struct.generator @User
// CHECK-SAME:    get_alignof
lit.struct.decl @User {
  lit.struct.field f : !lit.struct<@Aligned>
}

// -----

//===----------------------------------------------------------------------===//
// A raw pointer nested inside an aggregate rather than being the whole field
// type. The pointee is erased at any depth, so these need no wrapper struct to
// break the recursion.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.struct<(!kgen.pointer<:type !lit.struct<@Node>>)>
}

// -----

//===----------------------------------------------------------------------===//
// The same, nested inside an array.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.array<2, !kgen.pointer<:type !lit.struct<@Node>>>
}
