# Contribution process

This page describes the process for contributing code to Mojo. See
[contribution areas](contribution-areas.md) for which parts of the codebase
accept contributions and which kinds of change they accept. If you want to
report a bug rather than contribute code, use the
[GitHub issue tracker](https://github.com/modular/modular/issues) and follow the
[issue and PR etiquette](issue-pr-etiquette.md).

## Stage 1: Prerequisites

### Check that the contribution area is open

Before you implement a change, check that the part of the codebase you want to
improve is [open to receiving contributions](contribution-areas.md).

### Signal your intent

Make sure there's a GitHub issue describing the bug you intend to fix or the
feature you intend to add. This signals your intent and keeps other contributors
aware of what's already being worked on.

Search the existing issues before you create a new one, and follow the
[issue and PR etiquette](issue-pr-etiquette.md) when you open one.

## Stage 2: Planning

### Study the existing implementation

We consider the learning opportunities that open source software development
provides to be valuable. We recommend taking some time to understand the
existing code related to what you're about to fix.

> [!NOTE]
> Feel free to take advantage of AI to study the codebase. We ask that by the
> time you open a PR, you could hold a design conversation with us about the
> changes you've proposed, unassisted.

### Create an implementation plan

We *strongly recommend* that you post an implementation plan on the GitHub issue
once you have a solution in mind. Posting a plan gives others an opportunity to
comment before you commit to a particular design. This saves everyone time: it's
much easier to review and adjust a high-level plan than a finished
implementation.

Teams at Modular add an `accepted` label when there's consensus on an approach
and they're in agreement. The `accepted` label signals that we're ready to
accept a PR.

> [!NOTE]
> You can submit a PR before an issue has been labeled `accepted`. But if your
> change is non-trivial and the associated issue hasn't been accepted, the PR is
> less likely to be reviewed or approved.

## Stage 3: Implementation

We aren't prescriptive about *how* you arrive at your changes. Please add tests
that provide reasonable coverage for the code being added or changed. PRs
without tests are very unlikely to be merged. If you're unsure how to test
something, say so in the PR ("I have not added tests, not sure how to test this
capability") and we'll happily work with you.

By the time you open a PR, you should be ready to take full responsibility for
every line of code and for the PR description you submit. See the
[issue and PR etiquette](issue-pr-etiquette.md) for details. Add
`closes #<issue-number>` to the body of your PR so we can easily see which
GitHub issue your work relates to.

## Stage 4: Review

The review process is a fantastic way to learn interactively. Anyone with an
informed opinion is welcome to review a PR, and we actively encourage
constructive review by members of the community. We appreciate your patience,
and we remind everyone to abide by the
[issue and PR etiquette](issue-pr-etiquette.md) during the review process.

> [!IMPORTANT]
> Approval by a member of the Modular team is required before any PR can be
> merged into the codebase.

## Stage 5: Merge and release

Once your PR is approved, a Modular team member comments `!sync` on it. Your
issue gets a `merged externally` label and the PR on `modular/modular` is
closed. A corresponding PR opens on Modular's internal monorepo and merges once
it passes internal CI. A `merged internally` label is added once that PR merges
into the internal monorepo, and your commit appears on `modular/modular` with
the next nightly release.
