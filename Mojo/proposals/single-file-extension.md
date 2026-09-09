# Adopt .mojo as the sole supported extension

**Written**: 26 Feb 2026

**Status**: Accepted

## Summary

Mojo permits `.🔥` as an alternative to `.mojo` for source files. While visually
distinctive, the flame emoji introduces avoidable friction across the toolchain.
It's hard to type on many keyboards, raises the question of which extension to
use, complicates search and tooling by creating two valid file types, and leads
to inconsistent third-party support. Standardizing on `.mojo` removes this
ambiguity and makes Mojo easier to use.

## Current state

Mojo source files can use either extension:

```text
my_program.mojo   # Standard extension
my_program.🔥     # Emoji extension
```

Both extensions are semantically equivalent and recognized by the compiler.

### Proposed state

```text
my_program.mojo  # Supported extension
```

## Known problems with the flame emoji extension

**User training**: Users are not told which extension to prefer or why Mojo has
two extensions instead of one. Removing the flame emoji extension makes Mojo
more predictable.

**Input difficulty**: The flame emoji is hard to type.

**Command-line tooling issues**: Searching, using `sed`, and grepping are more
complex when two different but valid extensions exist.

**File system issues**: File associations require extra configuration and care.
For example, from macOS:

```sh
(experiment) bash-3.2$ open test.🔥
No application knows how to open URL file:.../codework/experiment/test.%F0%9F%94%A5
(Error details: ..."kLSApplicationNotFoundErr: E.g. no application claims the file" ...)
```

**Cross-platform reliability issues**: Emoji rendering varies across platforms.
Some systems display different flame designs. Others show empty boxes. Terminal
file listings can break alignment because emoji width is inconsistent. The same
file may look different across machines, which weakens visual recognition in
directory listings.

**Third-party Unicode readiness**: Third-party tools don't always handle
Unicode consistently.

### Tooling changes

- The compiler team removes the emoji as a valid extension and simplifies the
  conditional logic that handles multiple extensions.
- No Mojo language features depend on the emoji extension.
- Also recommended: remove the package emoji extension.

## Impact

Minimal.

The flame emoji extension is rarely used in production codebases. Migration
requires renaming any files that use the flame extension to `.mojo`. This is a
mechanical change that can be automated. The docs will need to remove references
to the flame extension.

The compiler team will need to update the recognized extension logic.
