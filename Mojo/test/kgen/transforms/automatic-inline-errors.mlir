// RUN: kgen-opt -split-input-file -automatic-inline -verify-diagnostics %s

kgen.func @ok_to_inline() always_inline {
  kgen.return
}

// expected-error @below {{function has recursive call to 'always_inline' function}}
// expected-note @below {{back to function here}}
kgen.func @circular.a() always_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.b() : () -> ()
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
kgen.func @circular.b() always_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.c() : () -> ()
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
kgen.func @circular.c() always_inline {
  // expected-note @below {{call here recurses}}
  kgen.call @circular.a() : () -> ()
  kgen.return
}

kgen.func @top0() {
  kgen.call @circular.a() : () -> ()
  kgen.call @ok_to_inline() : () -> ()
  kgen.return
}

// -----

// expected-error @below {{function has recursive call to 'always_inline' function}}
// expected-note @below {{back to function here}}
kgen.func @circular_inline.a() always_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular_inline.b() : () -> ()
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
kgen.func @circular_inline.b() always_inline {
  // expected-note @below {{call here recurses}}
  kgen.call @circular_inline.a() : () -> ()
  kgen.return
}
