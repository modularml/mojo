# Mojo documentation code examples

This directory includes code examples used in the Mojo Manual, the language
reference, and the tools documentation, to ensure those code snippets are
tested. Each subdirectory corresponds to a documentation page, and each example
has a Bazel test that runs it in CI.

## Running with Bazel

Run a single example's test from the Modular internal repo:

```sh
bt //Mojo/docs/code/manual/basics:hello_world_test
```

Or run every example under a topic (or the whole tree):

```sh
bt //Mojo/docs/code/manual/basics/...
bt //Mojo/docs/code/...
```

Use `br` to run an example directly instead of as a test:

```sh
br //Mojo/docs/code/manual/basics:hello_world
```

## Contributing

If you see something in the documentation or the code examples that is incorrect
or could be improved, we'd love to accept your contributions.

Be aware that code from this directory is **not** automatically included in the
corresponding documentation file. If you contribute a change to a code example,
please update the code example in the related documentation, as well as any
explanatory text.

Be aware that we don't provide tools to generate a preview of the website,
because the Mojo docs are built along with other content that's not included in
this repo. As such, we recommend you preview your edits in an IDE that can
render Markdown and MDX files, such as VS Code, including the
[VS Code environment in GitHub](https://github.dev/modular/modular/blob/main/).

For more information about how to contribute, see the
[Contributor Guide](../../CONTRIBUTING.md).
