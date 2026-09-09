# MAX documentation code examples

This directory holds the code examples embedded in the MAX developer
documentation, kept here as standalone, tested files so those snippets don't
rot. Each file corresponds to a code block in a MAX docs page, and each has a
Bazel test that runs it in CI.

## Running with Bazel

The examples are Python and Mojo programs, each defined as a Bazel target with
a test beside it. Run a single example's test:

```sh
bt //oss/modular/docs/code/develop/basic-ops:arithmetic_test
```

Or run every example under a topic (or the whole tree):

```sh
bt //oss/modular/docs/code/develop/basic-ops/...
bt //oss/modular/docs/code/...
```

Use `br` to run an example directly instead of as a test:

```sh
br //oss/modular/docs/code/develop/basic-ops:arithmetic
```

## Platform and resource notes

Not every example runs on every platform:

- Most example tests are marked incompatible with macOS and run on Linux; on
  Apple silicon they're skipped. They time out on remote macOS CI workers, and
  Linux CI provides equivalent coverage.
- A few `basic-ops` tests are additionally skipped on Apple GPUs, whose Metal
  compiler doesn't yet support the bf16 math intrinsics they use.
- `logit-comparison` requires a GPU and downloads a large model checkpoint, so
  it's tagged `manual` and isn't run in normal CI. Run it explicitly on a
  machine with a GPU.

## How the docs stay in sync

Each example file carries a `# DOC:` header naming the docs page that uses it,
for example:

```python
# DOC: max/develop/basic-ops.mdx
```

The
[`checkExampleDocSync`](../../../../.github/workflows/checkExampleDocSync.yaml)
GitHub workflow reads that header: when a PR changes an example here without
also editing the page it points to, the check fails and reminds the author to
update the docs. If a code change genuinely doesn't affect the prose, add
`DOCS_SYNC_SKIP` to the PR description to skip the check.

The code in this directory is **not** automatically injected into the
documentation. The docs page keeps its own copy of the snippet inline, so when
you change an example here you must also update the matching code block (and any
surrounding explanation) in the `.mdx` page.

## `docs/code/` versus `max/examples/`

Both directories hold tested MAX code, but they serve different owners:

- `docs/code/` (this directory) holds docs-owned snippets co-located with the
  documentation pages they appear in — short, focused examples that illustrate a
  single concept.
- [`max/examples/`](../../../../max/examples) holds engineer-owned, standalone
  projects that stand on their own as runnable applications; some also carry
  `# DOC:` backrefs when a docs page references them.

## Contributing

If you see something in the documentation or a code example that's incorrect or
could be improved, we'd love to accept your contributions. For more information,
see the [Contributor Guide](../../CONTRIBUTING.md).
