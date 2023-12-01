# Mojo Contributor Guide

Welcome to the Mojo community! 🔥 We’re very excited that you’re interested in
contributing to the project. To help you get started and ensure a smooth
process, we’ve put together this contributor guide.

## 1. First-time checklist

Before you start your first pull request, please complete this checklist:

- Read this entire contributor guide.
- Read the [Code of Conduct](CODE_OF_CONDUCT.md).

## 2. Evaluate and get buy-in on the change

We want to be sure that you spend your time efficiently and prepare changes
that aren’t controversial and get stuck in long rounds of reviews.

If your change is any one of the following, please create a pull request and we
will happily accept it as quickly as possible:

- Code improvement:
  - Bug fix
  - Performance improvement
  - Code readability improvement
  - Conformity to style improvement (TODO: publish code style guide)
- Documentation improvement:
  - Typo fix
  - Markup/rendering fix
  - Factual information fix
  - New factual information for an existing page

Before embarking on any major change, please **create an issue** or **start a
discussion**, so we can collaborate and agree on a solution.

For example, refactoring an entire code example or adding an entire new page to
the documentation is a lot of work and it might conflict with other work that’s
already in progress. We don’t want you to spend time on something that might
require difficult reviews and rework, or that might get rejected.

## 3. Create a pull request

If your change is one of the improvements described above or it has been
discussed and agreed upon by the project maintainers, please create a pull
request into the `main` branch and include the following:

- A short commit message.

- A detailed commit description that includes rationalization for the change
and/or explanation of the problem that it solves, with a link to any relevant
GitHub issues.

- A `Signed-off-by` line, as per the [Developer Certificate of
Origin](#signing-your-work).

**Note:** Documentation changes might not be visible on the website until the
next Mojo release.

Thank you for your contributions! ❤️

### Signing your work

For each pull request, we require that you certify that you wrote the change or
otherwise have the right to pass it on as an open-source patch by adding a line
at the end of your commit description message in the form of:

`Signed-off-by: Jamie Smith <jamie.smith@example.com>`

You must use your real name to contribute (no pseudonyms or anonymous
contributions). If you set your `user.name` and `user.email` git configs, you
can sign your commit automatically with `git commit -s`.

Doing so serves as a digital signature in agreement to the following Developer
Certificate of Origin (DCO):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
