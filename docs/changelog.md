# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language changes

### Standard library changes

### Tooling changes

- mblack (aka `mojo format`) no longer formats non-mojo files. This prevents
  unexpected formatting of python files.

### ❌ Removed

### 🛠️ Fixed

- The Mojo Kernel for Jupyter Notebooks is working again on nightly releases.

- The command `mojo debug --vscode` now sets the current working directory
  properly.

- The Mojo Language Server doesn't crash anymore on empty **init**.mojo files.
  [Issue #3826](https://github.com/modularml/mojo/issues/3826).
