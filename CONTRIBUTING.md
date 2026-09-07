# Modular contributor guide

Thank you for your interest in contributing to this repository!

This page explains the overall process to create a pull request (PR), from
forking the repo all the way through review and final merge.

## Submitting bugs

Reporting issues is a great way to contribute to the project.

Before opening a new issue, take a moment to search through the already
[submitted issues](https://github.com/modular/modular/issues) to avoid creating
duplicate issues for the maintainers to address.

### Writing high-quality bug descriptions

Bugs with a reproducible test case and well-written descriptions will be
considered a higher priority.

We encourage you to provide as much information about the issue as practical.
The more details you provide, the faster we can resolve the issue. The following
is a template of the information that should accompany every submitted issue.

#### Issue template

- **Summary.** A descriptive summary of the issue.
- **Description.** A detailed account of the bug, including what was expected
  and what occurred.
- **Environment details.**
  - MAX or Mojo version (run `max --version` or `mojo --version`)
  - Operating system version
  - Hardware specifications
- **Severity/frequency.** An assessment of the impact ranging from inconvenience
  to a blocker.

## Contributing changes

Before you start your first pull request, please complete this checklist:

- Read this entire contributor guide.
- Read the [Code of Conduct](./CODE_OF_CONDUCT.md).
- If you are using AI to assist with coding, read our
  [AI Tool Use Policy](./AI_TOOL_POLICY.md).

### Step 1: Evaluate and get buy-in on the change

First, consider that several parts of this repository currently do not accept
contributions. You should refer to the README or CONTRIBUTING file nearest the
code you're interested in.

We accept contributions to the following sections, each of which has its own
contribution guidelines:

- [Mojo](Mojo/CONTRIBUTING.md)
- [MAX API and models](/max/CONTRIBUTING.md)
- [MAX accelerator library](/max/kernels/CONTRIBUTING.md)
- [Code examples](/max/examples#contributing) and
  [Mojo/examples#contributing](Mojo/examples#contributing)
- [Mojo documentation](Mojo/docs#contributing)

#### Please open an issue before opening a pull request

We want your time to be well spent, and we want to be able to give your PR
the review it deserves. The best way to make both of those things happen is
to talk with us *before* you start writing code.

For anything beyond a small, obvious fix, please open a GitHub issue (or, for
larger changes, a
[proposal](Mojo/docs/contributing/proposal-process.md))
that describes the problem you're solving and the approach you have in mind. A
maintainer will respond to let you know whether the change aligns with where
we're headed, suggest alternatives, or flag anything you should know before
investing the time. Once a maintainer has signaled that the change is a good fit
— typically by adding the `accepted` label or by leaving a comment giving the
go-ahead — you're clear to start work.

What counts as a "small, obvious fix" where you can skip straight to a PR:

- Typos and grammar fixes in documentation or comments.
- One- or two-line bug fixes with an obvious root cause and a clear test.
- Small, localized documentation clarifications.

Everything else — new APIs, refactors, performance work, behavior changes,
anything touching public interfaces, or anything over ~100 lines — benefits
from a conversation first. If you're unsure which bucket your change falls
into, please err on the side of opening an issue; we're happy to help you
scope it.

If you open a non-trivial PR without a linked, maintainer-approved issue, we
may ask you to pause the PR and file an issue so we can align on the
approach. This isn't a rejection of your contribution — it's us trying to
make sure your work lands rather than stalling in review.

### Step 2: Create a pull request

If you're experienced with GitHub, here's the basic process:

1. Fork this repo.

2. Create a branch from `main`.

   If you're contributing to the Mojo standard library, see the
   [Mojo standard library developer guide](Mojo/docs/contributing/stdlib/stdlib-development.md).

3. Create a PR into the `main` branch of this repo.

4. Skip to [Step 3: PR triage and review](#step-3-pr-triage-and-review).

#### Format your changes

Please make sure your changes are formatted before submitting a pull request.
Otherwise, CI will fail in its lint and formatting checks. `bazel` setup
provides a `format` command. So, you can format your changes like so:

```bash
./bazelw run format
```

It is advised, to avoid forgetting, to set-up `pre-commit`, which will format
your changes automatically at each commit, and will also ensure that you
always have the latest linting tools applied.

To do so, install pre-commit:

```bash
pixi x pre-commit install
```

If you need to manually apply the `pre-commit`, for example, if you made a
commit with the github UI, you can do `pixi x pre-commit run --all-files`, and
it will apply the formatting to all Mojo and Python files.

You can also consider setting up your editor to automatically format
Mojo and Python files upon saving.

#### Validate your changes

Before submitting, make sure your changes are correct and complete:

- **Run the relevant tests.** If you changed code, run the tests for the
  affected area. For the Mojo standard library, see the
  [development guide](Mojo/docs/contributing/stdlib/stdlib-development.md)
  for instructions.
- **Check for regressions.** Run a broader test pass if your change touches
  shared infrastructure or has wide impact.
- **Assess quality.** Review your diff as a maintainer would. Is the logic
  clear? Are edge cases handled? Is the change well-scoped?

Contributors are responsible for the correctness and quality of their
submissions regardless of whether AI tools were used to produce them.

#### Pull request walkthrough

For more specifics, here's a detailed walkthrough of the process to create a
pull request:

1. Fork and clone this repo:

   Go to the [modular repo home](https://github.com/modular/modular) and click
   the **Fork** button at the top.

   Your fork will be accessible at `https://github.com/<your-username>/modular`.

   Clone your forked repo to your computer:

    ```bash
    git clone git@github.com:<your-username>/modular.git
    cd modular
    ```

   To clarify, you're working with three repo entities:

   - This repo (`https://github.com/modular/modular`) is known as the upstream
     repo. In Git terminology, it's the *upstream remote*.
   - Your fork on GitHub is known as *origin* (also remote).
   - Your local clone is stored on your computer.

     Because a fork can diverge from the upstream repo it was forked from, it is
     crucial to configure your local clone to track upstream changes:

     ```bash
     git remote add upstream <git@github.com>:modular/modular.git
     ```

     Then sync your fork to the latest code from upstream:

     ```bash
     git pull --rebase upstream
     ```

2. Create a branch off `main` to work on your change:

    ```bash
    git checkout -b my-fix
    ```

   Now start your work on the repo! If you're contributing to the Mojo
   standard library, see the [Mojo standard library developer
   guide](Mojo/docs/contributing/stdlib/stdlib-development.md).

   Although not necessary right now, you should periodically make sure you have
   the latest code, especially right before you create the pull request:

    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

3. Create a pull request:

   When your code is ready, create a pull request into the `main` branch.

   First push the local changes to your origin on GitHub:

    ```bash
    git push -u origin my-fix
    ```

   You'll see a link to create a PR:

    ```plaintext
    remote: Create a pull request for 'my-fix' on GitHub by visiting:
    remote:      https://github.com/[your-username]/modular/pull/new/my-fix
    ```

   You can open that URL or visit your fork on GitHub and click **Contribute**
   to start a pull request.

   GitHub should automatically set the base repository to `modular/modular` and
   the base (branch) to `main`. If not, you can select it from the drop-down.
   Then click **Create pull request**.

   Now fill out the pull request details in the GitHub UI:

   - Add a short commit title describing the change.
   - Fill out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
     In particular, link the GitHub issue that was agreed upon in
     [Step 1](#step-1-evaluate-and-get-buy-in-on-the-change) (for example,
     `Fixes #1234`) and write a clear motivation for the change.

     Click **Create pull request**.

### Step 3: PR triage and review

A Modular team member will take an initial look at the pull request and
determine how to proceed. This may include:

- **Leaving the PR as-is** (e.g. if it's a draft).
- **Reviewing the PR directly**, especially if the changes are straightforward.
- **Assigning the PR** to a subject-matter expert on the appropriate team
  (Libraries, Kernels, Documentation etc.) for deeper review.

We aim to respond in a timely manner based on the time tables in the
[guidelines for review time](#guidelines-for-review-time), below.

### Step 4: Internal review and syncing

Once a PR passes initial review and is progressing toward approval, a Modular
team member will sync it to our internal repository for further validation and
integration. This is done using an automated tool that mirrors your changes into
our internal environment.

This process is transparent to you as a contributor. You'll see a bot
(Modularbot) comment on your PR with status updates like:

- `Synced internally` - when your change has been synced internally into our
  repository
- `Merged internally` - when your change has been merged internally into our
  repository
- `Merged externally` - when your change has gone out with the latest nightly
  and is now available upstream in the `main` branch.

These messages help track the lifecycle of your contribution across our systems.

### Step 5: Review feedback and iteration

All feedback intended for you will be posted directly on the **external** pull
request. Internal discussions (e.g. security/privacy reviews or cross-team
coordination) may happen privately but won't affect your ability to contribute.
If we need changes from you, we'll leave clear comments with action items.

Once everything is approved and CI checks pass, we'll take care of the final
steps to get your PR merged.

Merged changes will generally show up in the next nightly build (or docs
website), a day or two after it's merged.

## Guidelines for review time

1. Pull Request (PR) Review Timeline

   Initial Review:
   - Maintainers will provide an initial review or feedback within 3 weeks of
     the PR submission. At times, it may be significantly quicker, but it
     depends on a variety of factors.

   Subsequent Reviews:
   - Once a contributor addresses feedback, maintainers will review updates as
     soon as they can, typically within 5 business days.

1. Issue Triage Timeline

   New Issues:
   - Maintainers will label and acknowledge new issues within 10 days of the
     issue submission.

1. Proposals

   - Proposals take more time for the team to review, discuss, and make sure
     this is in line with the overall strategy and vision for the standard
     library. These will get discussed in the team's weekly design meetings
     internally and feedback will be communicated back on the relevant proposal.
     As a team, we'll ensure these get reviewed and discussed within 6 weeks of
     submission.

### Exceptions

While we strive our best to adhere to these timelines, there may be occasional
delays due to any of the following:

- High volume of contributions.
- Maintainers' availability (e.g. holidays, team events).
- Complex issues or PRs requiring extended discussion (these may get deferred to
  the team's weekly design discussion meetings).

Note that just because a pull request has been reviewed does not necessarily
mean it will be able to be merged internally immediately. This could be due to a
variety of reasons, such as:

- Mojo compiler bugs. These take time to find a minimal reproducer, file an
  issue with the compiler team, and then get prioritized and fixed.
- Internal bugs that get exposed due to a changeset.
- Massive refactorings due to an external changeset. These also take time to
  fix - remember, we have the largest Mojo codebase in the world internally.

If delays occur, we'll provide status updates in the relevant thread (pull
request or GitHub issue). Please bear with us as Mojo is an early language.
We look forward to working together with you in making Mojo better for everyone!

### How you can help

To ensure quicker reviews:

- **Ensure your PR is small and focused.** See the
  [pull request size guidance](Mojo/docs/contributing/issue-pr-etiquette.md#keep-pull-requests-small)
  for more info.
- Write a good commit message/PR description outlining the motivation and
  describing the changes. Use the
  [pull request template](.github/PULL_REQUEST_TEMPLATE.md) as a guide.
- Use descriptive titles and comments for clarity.
- Code-review other contributor pull requests and help each other.

## Behind the scenes (FYI)

Here are a few implementation details that help us keep things running smoothly:

- We use a tool called [**Copybara**](https://github.com/google/copybara) to
  sync changes between internal and external repos.

- Your GitHub username and PR number are automatically preserved via commit
  metadata like:

    ```plaintext
    ORIGINAL_AUTHOR=username 12345678+username@users.noreply.github.com
    PUBLIC_PR_LINK=modularml/mojo#2439
    ```

- This repo is synced nightly with Modular's internal repo around 2 am ET
almost every day. This means the `main` branch may lag slightly behind our
internal repository by up to 24 hours. At times, it may be longer in case of a
(blocking) release failure in our internal CI release workflows.

## 🙌 Thanks for contributing

We deeply appreciate your interest in improving the Modular ecosystem. Whether
you're fixing typos, improving docs, or contributing core library features, your
input makes a difference.

If you have questions or need help, feel free to:

- Leave a comment on your pull request
- Join our community [forum](https://forum.modular.com/) and post a question

Let's build something great together!
