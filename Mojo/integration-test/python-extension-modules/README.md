# Python -> Mojo Bindings Integration Tests

This directory contains tests for calling Mojo from Python.

## Tests

The tests can be run using Bazel:

```shell
bazel test //Mojo/integration-test:lit
```

The tests typically have the following structure:

```text
<name>
├── main.py
└── mojo_module.mojo
```

where `main.py` is the test entrypoint, which compiles and loads
`mojo_module.mojo` as a
[Python extension module](https://docs.python.org/3/extending/extending.html).

`mojo_module.mojo` should contain Mojo code written to be called from Python,
and `main.py` should test that exposed Mojo code.

## File Overview

- [./basic-raw](./basic-raw/) — a minimal low-level smoke test, calling a simple
  Mojo function from Python. Uses the low-level "raw" function bindings, which
  are more error prone.

- [./feature-overview](./feature-overview/) — This test aims to include a basic
  test for each supported Mojo language feature that is usable across the
  Python <=> Mojo interop boundary.

## Manual Test Procedure

Python extension modules are just dynamic libraries that expose a suitable
`PyInit_<module_name>()` function. To build a Mojo library into an extension
module, you can use the following command:

```shell
mojo build mojo_module.mojo --emit shared-lib -o mojo_module.so
```

Which will result in a `mojo_module.so` being built and placed alongside the
existing files:

```text
<name>
├── mojo_module.mojo
├── mojo_module.so
└── main.py
```

Running the Python `main.py` code will load and run compiled Mojo code
from `mojo_module.so`:

```shell
% python main.py
Result from Mojo 🔥: 2
```
