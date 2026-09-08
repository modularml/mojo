// RUN: kgen %s -elaborate -S -o - -Dbar -D foo | FileCheck %s
// RUN: kgen %s -elaborate -S -o - -Dbar | FileCheck %s --check-prefix=UNDEF

kgen.generator export @main() -> i1 {
  // CHECK: constant: i1 = <1>
  // UNDEF: constant: i1 = <0>
  %0 = kgen.param.constant: i1 = <get_env("foo")>
  kgen.return %0 : i1
}
