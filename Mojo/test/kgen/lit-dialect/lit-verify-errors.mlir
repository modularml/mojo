// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

// This tests verification errors which are not enabled in production builds
// UNSUPPORTED: production

lit.fn @im_a_func() {
  kgen.return
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute type @im_a_func does not refer to a struct declaration}}
  kgen.param.constant: @im_a_func = <#lit.struct<{}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @TwoFields {
  lit.struct.field a : index
  lit.struct.field b : index
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct declaration expected 2 fields but struct attribute has 0}}
  kgen.param.constant: @TwoFields = <#lit.struct<{}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @TwoFields {
  lit.struct.field a : index
  lit.struct.field b : index
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute field name "c" at position #1 does not match the name "b" in the struct declaration}}
  kgen.param.constant: @TwoFields = <#lit.struct<{a = 1, c = 2}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.param<ty>
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute field #0 has type 'index' but corresponding struct field "a" expected 'i1'}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a = 5}>>
  kgen.return
}

// -----

lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.param<ty>
}

lit.fn @struct_attr() {
  // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "A"}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a: i1 = A}>>
  kgen.return
}
