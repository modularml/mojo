# Proposal: Stdlib extensions repository

## TL,DR

There are roughly 3 maintainers currently reviewing and merging pull
requests in the Mojo stdlib.
This is because contributor's code then goes into the internal
monorepo where there is sensitive Modular code.

Most of the pull requests made are about new functions/structs.

We propose here to make a new repository at
github.com/modularml/stdlib-extensions where all new features
of the stdlib would go. The source of truth would
 be public and some maintainers would be external to Modular.
When a new feature is ready (or necessary for the internal codebase),
commits would be pulled from
`modularml/stdlib-extensions` to `modularml/mojo` through
normal pull requests.

New features can there sit there for as long as we want,
and this will take a lot of work off the shoulders of
the stdlib team while improving the contributor experience
at the same time.

## Current pain points we are trying to solve

### Current pain points for Modular employees

- Modular employees have to prioritize between internal
  work and reviewing external contribution, which
  may not be in line with current priorities internally.
- They cannot get help from external contributors since
  they should not merge into the internal codebase
  by themselves.
- Modular employees have to fix code of contributors which
  passes the external CI but not the internal one.
- For each pull request, they must ensure
  that there is no malicious code before syncing it.
- After the syncing maintainers have to wait until the
  internal CI is passing. (An immediate code review
  may be unnecessary work or incomplete
  if the CI is not green).
- Modular employees have to fix broken
  nightly releases rapidly as contributors
  cannot continue their work otherwise.
- They have to address all pull requests,
  even if they didn't approve the feature
  beforehand or if the pull request
  isn't a good fit at all for a standard library.
- Maintainers are pressured into accepting new
  features in the stdlib because there isn't a
  package manager yet,
  and it's the only place which has "Modular's blessing"
  concerning new features. This was similar to Python's
  early days where the lack of a package manager
  made the stdlib grow quickly.
- Maintainers have to fix all code in `nightly` when there are
  breaking changes, even structs and functions which
  aren't used internally.

### Current pain points for contributors

I want to emphasize that the Modular stdlib team is awesome, and
always go the extra mile for code reviews, but is victim of Mojo's success here.
There is not enough time in a work day.
So we must find a solution to take some work off their shoulders.

- Getting code reviews is slow as there are only
  3 maintainers and they have other tasks in their full-time jobs.
  New pull requests can wait multiple weeks without getting a code review.
  At the time of writing, 76 pull requests are opened and most
  of them are waiting for a code review.
- Contributors cannot get code reviews
  during weekends and holidays, which are the times
  when the contributors are the most active.
- It's not encouraged to split work as pull request
  which depends on each other can only be rebased between 12
  and 72h after the previous pull request is merged.
  This is because the nightly release is quite flaky and often fails.
  In the best case scenario, it's still 12-24 hours wait between
  merges of "chains" of pull requests.
- Contributors have no visibility on why a pull request can be blocked
  by the internal CI.
- Sometimes a pull request can even be reverted after
  being merged in nightly because it changes the IR.
- As maintainers are very busy, it is hard to get their
  opinions on issues or get directions in general. A lot
  of questions are left unanswered, even legitimate ones.
- As external contributors can't see internal pull requests,
  they cannot anticipate conflicts, or even worse,
  they can re-do a pull request which is already opened internally.

We can really say that while the contribution
process works, it is quite far from perfect and
contributors often hold back on new contributions to
let maintainers take the time to review previous ones.
External pull requests are second-class citizens
compared to internal pull requests and this is being
felt at every step of the contribution process.

### Source of those pain points

The source of those pain points is the internal codebase and the fact
that external contributors can only see part of the codebase, with a delay.
The rationale for this setup was that Modular products depends on the
features that the contributors can modify.
As such, a pull request cannot be merged if it breaks a Modular product.

But at the same time, external contributors should not be
able to see the reason of the breakage as it might leak information
about internal products.
Furthermore, the compiler isn't open-sourced, leading to the nightly system
and the synchronization of
branches once a day because contributors cannot build the compiler from a given commit.

We will assume in this document that the described setup
isn't going to change anytime soon.
Though the proposed solution still has usefullness
beyond the mentionned pain points.

### Proposed solution to reduce the pain of this setup

From looking at opened, closed and merged external contributions,
we can say that a very significant number of them are about new features.

Those new features are very unlikely to be used inside Modular products and thus,
should not create all the pain points mentionned above.
As such, they can go in another repository, until they
are needed/wanted in the internal codebase.

We will call this repository `stdlib-extensions` in this document.

This repository would have the source of truth be public, would have
a public github actions CI and would be
maintained by both internal and external developers.

In the end, pull requests targetting this repository should be
unaffected by all the pain points mentionned above.
To be clear, the standard library should not depend on `stdlib-extensions`.
`stdlib-extensions` sits on top of the stable release of the standard library.

## `stdlib-extensions`, how would it work?

### Hosting

`stdlib-extensions` should be hosted in the `modularml` organization,
at <https://github.com/modularml/stdlib-extensions>. There would be no
"internal" version of this repository. The source of truth is public.

The CI would be the public Github action for simplicity.
Unit test would run with lit with roughly the same setup
as `modularml/mojo`.

### Releases

When creating releases, Github actions should upload
`stdlib-extensions.mojopkg` to Github.
If end users want to try the functionalities of `stdlib-extensions`,
they should download the `.mojopkg` file
themselves. Modular and maintainers will NOT handle the distribution.
End users should also use the stable Mojo release with `stdlib-extensions.mojopkg`,
not the nightly. There is no garantee about backward compatibility.

### Dependencies

`stdlib-extensions` should only depend on the latest stable release of Mojo.
New dependencies will not be accepted.
Depending on the nightly release will **not** be accepted.

### Synchronization

The version of the stable release of Mojo used in the CI should
be pinned in a file.

When a new version is out, someone (maintainer or contributor)
should open a pull request to change
the version of Mojo in this file and fix all the breaking changes.
This will ensure the `main` branch
is always green.

### Governance

At first, Modular employees will be the only maintainers of `stdlib-extensions`.
They will then choose, whenever they feel confortable, contributors,
and change their status to core developers of `stdlib-extensions`.
Those core developers will only have merge right and won't be
able to touch the settings of the repository
(this would be only possible for Modular employees to do so).
Furthermore, they won't be able to bypass the CI to merge pull
requests.

The core-developers will have the status
of [outside collaborator](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-outside-collaborators/adding-outside-collaborators-to-repositories-in-your-organization)
in the `modularml` github organization.

Core developers would have no obligation of reviewing PRs,
no quota to fill, etc... just like in any other open source project.
They are just granted additional permissions.

### Requirements for merging

- 1 approval from a core developer or maintainer.
- A green CI
- The developer certificate of origin
- Squash and merge when accepting the changes in `main`.

### New features to `modularml/mojo`

Direct contributions of new features to `modularml/mojo` should be refused
and redirected to `modularml/stdlib-extensions`. Unless they are needed by
the Modular staff, for example for imminent internal use or because
it is considered a core abstraction in the standard library.

Having the feature in `modularml/stdlib-extensions` instead
will allow for quick iteration
on them without requireing Modular staff involvement (but it is still possible,
for design decisions for example).
It will also allow Modular staff to delay the choice
indefinitly of accepting or not a
feature in `modularml/mojo`, thus allowing them to focus
on what's the most important for the project at any given time.

A feature can mature inside `modularml/stdlib-extensions` before
being transferred to `modularml/mojo`.
It's also a way to judge popularity.

Accepting a new feature in `modularml/stdlib-extensions` is up to the reviewer,
which can be a core developer or Modular staff.
In case of conflict between core developers, the Modular staff can intervene
and take a decision.

### Transferring code from `modularml/stdlib-extensions` to `modularml/mojo` (graduating features)

Before the transfer, the stdlib maintainers (Modular staff) should take
a look at the feature
in `modularml/stdlib-extensions` and open an issue there for the transfer.
In this issue they can describe all the changes (design, API, etc...) that they
require. Those changes should be applied to the feature
inside `modularml/stdlib-extensions`
through regular pull requests, because it's easier to iterate on this repository
than in `modularml/mojo`.
The core developers can review and merge those pull requests to
allow faster iteration.

Once the Modular staff is satisfied with the overall code, we can proceed with
the actual transfer.

The transfer will take the form of a regular pull request
to `modularml/mojo`, targeting nightly, with all contributors to the feature
added as co-authors. The git blame/log can be used to get this list of co-authors.

The pull request to transfer code can be made by anybody as long as
there was approval from a Modular employee for this transfer beforehand.
When squashing the commit for the merge, all contributors
in the `modularml/stdlib-extensions`  commit history should
be added as co-authors of the squashed commit.

This pull request will be used as the final quality gate.
Modular staff will ensure the quality of the feature is
adequate before merging it in `modularml/mojo`.
It will also be the place to fix all breaking changes introduced in the
nightly since the latest stable release which impacted the feature's source code.

### The future of this repository when Mojo has a public source of truth

The repository can still be used a an extension to the
stdlib where we experiment with new features before accepting
them in the stdlib. This will allow to judge the usefullness
of new feature and judge the api quality.

### Amount of work needed by the Modular staff for this

The community will ask modular staff to create the repository with the right settings.
The community can then setup the CI in the repository and start
adding new features and tests for those features.

## Similar work

One can find in open-source repositories that have an "incubator"
repository, where new features are tested and
mature before pulling them in the main repository.
It's also a way to judge the popularity of new features.

The author of this proposal was personnaly involved with `tensorflow-addons`,
and believe that this was a massive improvements to the contributor's
experience compared to the Tensorflow repository.

- [Tensorflow](https://github.com/tensorflow/tensorflow) (Internal source of truth)
  -> [Tensorflow-addons](https://github.com/tensorflow/addons) (
    Public source of truth)
- [typing (python)](https://github.com/python/cpython/blob/3.12/Lib/typing.py)
  -> [typing-extensions](https://github.com/python/typing_extensions)
- [opentelemetry-java](https://github.com/open-telemetry/opentelemetry-java)
  -> [opentelemetry-java-contrib](https://github.com/open-telemetry/opentelemetry-java-contrib)
