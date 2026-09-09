# MAX contributor guide

Welcome to the MAX community! 🧑‍🚀 We’re very excited that you’re interested in
contributing to the project. To help you get started and ensure a smooth
process, we’ve put together this contributor guide.

There are many ways to contribute to the project, from joining our
[forum](https://forum.modular.com/) or
[Discord community](https://www.discord.gg/modular), to filing bugs, to
contributing documentation, examples, or code.

## Contributing to the MAX framework and models

To ensure a streamlined process, contributors are encouraged to focus on
enhancements, bug fixes, and optimizations aligned with the overarching goals
of MAX. These guidelines aim to facilitate a collaborative
environment where contributors and the Modular team can work together
effectively toward the continued improvement of MAX.

> [!IMPORTANT]
> The library of models included in this repository is intended
> to contain high-quality models that are of broad interest to
> the community. Some models are also critical to Modular's
> roadmap and are subject to more extensive testing.
> While small contributions are welcome, it's a good idea to
> open an issue and discuss with the community before starting
> work on a new model. This helps avoid duplication of effort
> and ensures the right home for the model — either this
> repository or a separate library.

> [!IMPORTANT]
> Some parts of the codebase are under active development,
> particularly key infrastructure components such as the
> pipeline infrastructure. These areas may be under active
> design by a Modular tech lead (for example, support for new
> modalities). Contributions to these components may receive
> closer review or be superseded by internal work, so opening
> an issue to discuss first is strongly recommended. We expect
> these areas to stabilize over time.

### Areas of contribution

In general, small bug fixes are welcome throughout the stack. Changes that
are larger in nature, or changes which have a deep impact should be first
discussed. Using GitHub issues is a great way to do that. The table below
summarizes code paths and contribution status for more significant changes.

| Path                                | Description                       | Status                                                                     |
|-------------------------------------|-----------------------------------|----------------------------------------------------------------------------|
| `max/include/max/c`                 | Closed source bindings            | N/A                                                                        |
| `max/docs`                          | Developer documentation           | Contributions welcome                                                      |
| `max/examples`                      | Code examples                     | Contributions welcome                                                      |
| `max/kernels`                       | Public MAX kernels                | See `max/kernels/CONTRIBUTING.md`                                          |
| `max/python/docs`                   | Python API docs                   | Automatically generated                                                    |
| `max/python/max`                    | Python API sources                | See the table below                                                        |

For code under `max/python/max`:

| Path                                           | Description                   | Status                               |
|------------------------------------------------|-------------------------------|--------------------------------------|
| `benchmark`                                    | Benchmarking scripts          |                                      |
| `config`                                       | Core MAX Serve infrastructure | Under active development             |
| `driver`, `dtype`, `engine`, `mlir`, `support` | Low-level APIs                |                                      |
| `_entrypoints`, `profiler`                     | Tools and high-level APIs     |                                      |
| `experimental`                                 | APIs under development        | Under active development             |
| `graph`                                        | Stable MAX graph API          | Discuss before adding nontrivial ops |
| `kv_cache`                                     | LLM KV cache APIs             | Under active development             |
| `pipelines`                                    | MAX models library            | Please avoid adding new modalities   |
| `serve`                                        | MAX web server                |                                      |

### Getting started

For technical details on developing for MAX and models, see the following
document:

- [Developing the MAX framework](/max/docs/development.md) covers building,
  testing, and other information you'll need to work in the MAX framework.
  It also covers local test prerequisites (`HF_TOKEN`, model downloads,
  GPU-only targets) and a minimal test matrix for local iteration.

When validating a local MAX change, start with the minimal test matrix in the
[MAX developer guide](/max/docs/development.md#minimal-test-matrix)
instead of defaulting to `//max/...`, especially if your change does not need
HF-backed or GPU-backed integration coverage.

### Changes we *accept*

These changes are uncontroversial, easier to review, and more likely to be
accepted:

- Well-documented bug fixes submitted with code reproducing the issue in a test
  or benchmark.
- Performance improvements that don’t sacrifice code readability or
  maintainability and are accompanied by benchmarks.
- Improvements to API documentation.
- Improvements to the test coverage.
- Changes that address security vulnerabilities.

### Changes we *avoid*

Changes that don’t align with our vision and roadmap are unlikely to be
accepted. For example:

- Changes that do not align with the core principles of
  the MAX framework. When in doubt, feel free to reach out to the Modular team
  or community in our [forum](https://forum.modular.com/) or
  [Discord](https://www.discord.gg/modular) to see if a potential change would
  be well-received.
- Code without tests—especially for core primitives.
- Changes that break existing API, models, or implicit behavior semantics.
- Changes where the contributors’ favorite feature or system isn’t being used
  and they submit a change unilaterally switching the project to use it. For
  example, the contributor doesn’t like Bazel as a build system and submits a PR
  changing the repository to use their favorite build system.
- Adding support for esoteric platforms.
- Adding dependencies to the code base.
- Broad formatting or refactoring changes.
- Changes that need broad community consensus.
- Changes if contributors are not responsive to review comments.

## About pull request sizes

We ask that contributors make pull requests as small as possible. When
you are opening a pull request, check the number of lines modified in GitHub.
The smaller the better (but don't exclude the tests or docstrings). If your
pull request is over 100 lines, please try to split it into multiple pull
requests. If you make them independent, it's even better as no synchronization
will be needed for the merge.

This guideline is here for the following reasons:

- **Higher quality reviews**: It is much easier to spot a bug in a few lines
than in 1000 lines.
- **Faster overall review**: Reviewers, to approve a pull request, need to
understand every line and understand how it fits into your overall change. They
also need to go back and forth between files and functions to understand the
flow of the code. This is exponentially hard as there are more lines in the
code.
- **Avoiding blocking changes that are valid**: In a huge pull request, it's
likely that some changes are valid and some need to be reworked/discussed. If
all the changes are in the same pull request, then the valid changes will be
blocked until all discussions have been resolved.
- **Reducing the number of git conflicts**: Bigger pull request means slower
  reviews,
thus means that the pull request will be open longer and will have more git
conflicts to be resolved before being merged.
- **Parallel processing**: All programmers like to parallelize. Well, reviewers
  also
like to parallelize code reviews to merge your code faster. If you open two pull
requests that are independent, then two reviewers will be able to work on your
code.
- **Finding the time for a code review**: Doing a code review often requires
that the code is reviewed in one go, as it's hard to remember functions and code
logic from one review session to another. Thus a big pull request will require
the reviewer to allocate a big chunk of time to do the code review, which is not
always possible and might delay the review and merge of your pull request
for multiple days.

Smaller pull requests means less work for the maintainers and faster reviews
and merges for the contributors. It's a win-win!

To help break apart big projects into smaller PRs, try using [stacked pull
requests][stacked-prs] and the `gh stack` CLI, which simplify the process of
creating several PRs in a dependency chain.

[stacked-prs]: https://docs.github.com/en/pull-requests/how-tos/stacked-pull-requests

## Submitting pull requests

For details about how to submit a pull request, see the repo's
[primary contributing guide](../CONTRIBUTING.md).

Thank you for your contributions! ❤️
