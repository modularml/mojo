# Removing `mojo test`

Author: Laszlo Kindrat
Date: Oct 13, 2025
Status: Implemented.

The `mojo test` utility has not gained meaningful adoption within Modular since
its inception two years ago. In mid-2024, it was deprioritized and (aside from a
few fixes shortly after) became unmaintained. This document lays out rationale
for removing it, and how we would go about building something better.

## What `mojo test` does and does not do today

While `mojo test` used to depend on the REPL to easily introspect IR (and maybe
to one day integrate with a debugger), this made it error-prone and slow: the
REPL uses a different compilation model than `mojo build` and `mojo run`, so
what `mojo test` ran wasn’t exactly what the user wrote. For this reason (and
the consequence that it might not allow testing on GPUs) this dependency was
removed, and `mojo test` now performs test file discovery within a given path,
unit test (i.e. function) discovery within test files, and the execution of the
test cases using largely the same logic as `mojo run` (although not by directly
calling it).

This allows small external projects to quickly bring up simple test suites (no
need for a build system, or even a shell script). However, since `mojo test`
doesn’t have advanced features like result caching, parallel and remote
execution, or separation of building vs. running a test, it cannot serve the
needs of large scale projects, such as the Mojo standard library or the kernel
libraries.

## Design issues

There are a number of fundamental design issues that prevent `mojo test` from
being great:

- The implementation is in C++, and currently not open source.
  - Even when it will be open sourced, it will only be accessible to those
    familiar with C++ and MLIR.
- It is tightly coupled to the parser implementation.
- To orchestrate unit tests, `mojo test` needs to emulate a `main` function
  (yuck!), which makes controlling test orchestration difficult from within the
  suite.

## Implementation problems

Possibly due to Modular’s lack of investment into `mojo test`, it has over time
accumulated a backlog of tickets:

- <https://github.com/modular/modular/issues/4732>
- <https://github.com/modular/modular/issues/5050>
- <https://github.com/modular/modular/issues/5157>
- <https://github.com/modular/modular/issues/5379>

A particular disappointment with `mojo test` was its lack of ability to test
code examples in docs strings. Our docs team came up with a completely separate
strategy to reliably test code examples.

## What we should do instead

Since Modular does not plan to invest in `mojo test` in the short to medium
term, we propose that we remove it altogether. This will feel like ripping a
bandaid for external projects, but the standard library provides a sufficient
example for the alternative: endow test files with `main`, use the recently
introduced
[automatic test discovery](https://github.com/modular/modular/commit/4570dda790c840aa01e89f82ba3650020e2de94f)
(or register test functions explicitly in a `TestSuite`), and orchestrate using
a build system (or even just a shell script).

To actually improve the testing experience, the standard library team (who are,
notably, the maintainers of the largest Mojo test corpus) is working on a new
and improved test framework written *in Mojo*. We have reflection utilities for
automatic discovery of test functions, and some infrastructure already exists
for compiling and running Mojo from within Mojo.

We believe that existing build systems (e.g. bazel) already provide a reliable,
flexible, and scalable solution to discovery and orchestration of tests for Mojo
projects. Nonetheless, we don’t want to close any doors; as the new, library
driven testing framework matures and larger external projects start to take
advantage of it, we will reevaluate if an improved `mojo test` is worth
investing in.

## How to migrate external projects

To migrate the tests of external projects, maintainers should add the following
to their `test_*.mojo` files:

```mojo
from testing import TestSuite

# ... The existing test functions starting with `test_`.

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Then they can orchestrate the discovery of the test files using the following
shell script:

```bash
#!/bin/bash
set -e

# Get test directory from first argument
if [ -z "$1" ]; then
    echo "Error: Test directory not provided"
    echo "Usage: $0 <test_directory>"
    exit 1
fi
test_dir="$1"

any_failed=false
echo "### ------------------------------------------------------------- ###"
while IFS= read -r test_file; do
    if ! mojo run "$test_file"; then
        any_failed=true
    fi
    echo "### ------------------------------------------------------------- ###"
done < <(find "$test_dir" -name "test_*.mojo" -type f | sort)

if [ "$any_failed" = true ]; then
    exit 1
fi
```

## Steps to remove `mojo test`

We propose the deprecation of `mojo test` no later than October 17, 2025 (a
custom deprecation warning will be emitted whenever `mojo test` runs), and then
removal within two weeks, i.e. no later than October 31, 2025.
