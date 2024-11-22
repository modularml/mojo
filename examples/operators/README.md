# Implementing operators for a custom Mojo type

This directory contains an example of implementing operators for a custom Mojo
struct named `Complex`, which represents a single complex number. The
`my_complex.mojo` module defines the `Complex` struct, and `main.mojo` is a
program that imports the module and shows examples of applying the operators to
instances of the `Complex` struct. The `test_my_complex.mojo` file is a set of
unit tests using the [Mojo testing
framework](https://docs.modular.com/mojo/tools/testing).

Refer to [An example of implementing operators for a custom
type](https://docs.modular.com/mojo/manual/operators#an-example-of-implementing-operators-for-a-custom-type)
in the [Mojo manual](https://docs.modular.com/mojo/manual/) for a complete
explanation of the implementation of the `Complex` type.

If you have [`magic`](https://docs.modular.com/magic) installed, you can
execute the example by running the following command:

```bash
magic run mojo main.mojo
```

You can run the unit tests by running the following command:

```bash
magic run test
```
