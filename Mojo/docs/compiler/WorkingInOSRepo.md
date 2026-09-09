# Working with Mojo in the Open Source Repo

Many docs in this directory describe tools and workflows used in the Modular
monorepo, which can work differently in the open source repository.

## Using Bazel

The Modular build system uses Bazel. For more information on using Bazel, see
[Using bazel](/bazel/docs/usage.md).

### Building Mojo

When building the Mojo compiler, run `bazel` with the `--config=build-mojo`
flag. Instead of adding this to every command, you can add the following to a
`local.bazelrc` file:

```text
build --config=build-mojo
```

### Bazel Aliases

Many of the Mojo compiler docs use aliases for common Bazel commands. The
following table shows the aliases and the commands they stand for, where
`<REPO_PATH>` is the path to the root of your local repository.

| Alias | Command                     |
|-------|-----------------------------|
| `bb`  | `<REPO_PATH>/bazelw build`  |
| `br`  | `<REPO_PATH>/bazelw run`    |
| `bt`  | `<REPO_PATH>/bazelw test`   |

For convenience, you can define these aliases yourself. Even if you don't,
knowing them will help you follow the Mojo compiler docs.

### Example commands

Build the Mojo compiler and standard library:

```sh
./bazelw build --config=build-mojo //Mojo:mojo
```

Run a single Mojo file with the locally built compiler:

```sh
./bazelw run --config=build-mojo //Mojo:mojo -- run main.mojo
```

Build and run the standard library tests with a locally built compiler:

```sh
./bazelw test --config=build-mojo //Mojo/stdlib/...
```

If you set the build configuration in `local.bazelrc` and define the aliases,
those three commands become:

```sh
bb //Mojo:mojo
br //Mojo:mojo -- run main.mojo
bt //Mojo/stdlib/...
```

> [!NOTE]
>
> You can't use a locally built compiler to build any of the MAX targets, so
> use the prebuilt compiler (`--config=prebuilt-mojo`) instead. Also test any
> standard library changes against the prebuilt compiler before you open a pull
> request.

## Using Mojo tools

Mojo includes several low-level tools that let you examine the intermediate
output the Mojo compiler generates, including `kgen` and `kgen-translate`.

To run these commands in the open source repo, add the `-I Mojo/stdlib` flag to
include the Mojo standard library. For example, the
[Passes and Intermediate Representations](manual/PassesAndIR.md) doc shows this
instruction for generating the `lit` dialect from a Mojo file:

```sh
br //Mojo/tools/kgen-translate -- -import-mojo main.mojo
```

The open source equivalent is:

```sh
./bazelw run //Mojo/tools/kgen-translate -- -import-mojo -I Mojo/stdlib main.mojo
```

This command doesn't build the standard library, so build it first as described
above.

## Unsupported workflows

The Modular monorepo includes several targets and workflows that currently
don't work in the open source repo. In particular, you may see references to:

- `start-modular.sh`: monorepo environment setup. There's no equivalent for the
  open source repo.

- `//:install`: build and install target. Creates a stateful development
  environment, installing build artifacts into your `PATH` so you can run tools
  directly. You can copy artifacts from the build output (`bazel-bin`) to a
  directory in your `PATH`, but you'll need to update them manually each time
  you rebuild the compiler.

## Next steps

For an overview of the compiler's architecture, see the
[Mojo Compiler Walkthrough](MojoCompilerWalkthrough.md).

Or proceed to the [Mojo Compiler Dev Manual](manual/README.md).
