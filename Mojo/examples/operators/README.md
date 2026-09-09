# Implementing operators for a custom Mojo type

This directory contains an example of implementing operators for a custom Mojo
struct named `Complex`, which represents a single complex number. The
`my_complex.mojo` module defines the `Complex` struct, and `main.mojo` is a
program that imports the module and shows examples of applying the operators to
instances of the `Complex` struct. The `test_my_complex.mojo` file is a set of
unit tests using the [Mojo testing
framework](https://mojolang.org/docs/tools/testing).

Refer to
[Add operator support to custom types](https://mojolang.org/docs/manual/structs/operator-support)
in the [Mojo manual](https://mojolang.org/docs/manual/) for a complete
explanation of the implementation of the `Complex` type.

If you have [Pixi](https://pixi.sh/latest/) installed, you can
execute the example by running the following command:

```bash
pixi run mojo main.mojo
```

You can run the unit tests by running the following command:

```bash
pixi run test
```
