//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Minimal non-Mojo host: dlopens a Mojo-built shared library and calls its
// exported C-ABI entry point. Because the process main is not Mojo, the Mojo
// startup shim never runs, so the library itself must initialize the runtime.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef long long (*mojo_parallel_sum_t)(long long);

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <shared-lib-path>\n", argv[0]);
    return 2;
  }
  void *lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }
  mojo_parallel_sum_t fn = (mojo_parallel_sum_t)dlsym(lib, "mojo_parallel_sum");
  if (!fn) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 2;
  }
  // Call twice: runtime initialization must be idempotent across entry-point
  // invocations.
  printf("sum1=%lld\n", fn(1000));
  printf("sum2=%lld\n", fn(1000));
  return 0;
}
