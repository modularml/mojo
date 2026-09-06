# Reproducer: JIT SIGSEGV — cross-module `raises` factory, InlineArray struct by value

Reproduces https://github.com/modular/modular/issues/6971

`mojo run` (JIT) SIGSEGVs in libKGENCompilerRTShared at codegen when module B
calls a module-level `raises` factory in module A that returns a struct with an
`InlineArray` field by value and contains `raise Error(msg + String(x))`
(string concatenation) in its raise path. Replacing the concatenated message
with a literal stops the crash (8/8 vs 10/10+3/3).

## Run

```sh
cc -dynamiclib -o libmsstack.dylib shim.c
mojo run -I . -Xlinker ./libmsstack.dylib crash_main.mojo   # exit 139 + stack dump
# edit crash_lib.mojo: raise Error("alloc failed")           # literal
mojo run -I . -Xlinker ./libmsstack.dylib crash_main.mojo   # exit 0, "survived <addr>"
```

This directory exists solely to reproduce the linked issue for triage; it is
not intended to be merged as-is. Mojo 1.0.0b2 (2cf4d08a), macOS arm64.
