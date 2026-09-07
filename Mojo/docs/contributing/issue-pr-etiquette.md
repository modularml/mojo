# Issue and PR etiquette

We don't restrict which tools you use to prepare a contribution, but you are
fully responsible for the work you submit, for the quality of that work, and for
how you interact with others online. Modular's [AI tool use
policy](https://github.com/modular/modular/blob/main/AI_TOOL_POLICY.md) applies
in full.

The Mojo team emphasizes and enforces the following rules to encourage genuine
discussion, understanding, and growth.

## You must be present for all online interactions

- Online interactions such as issues, reviews, and pull requests are a
  conversation. We expect you to show up as a human and respond to comments
  yourself.
- The burden is on you to demonstrate that you're present and respectful.

## Assume positive intent

- Information asymmetry almost always exists between people. Taking the time to
  understand circumstances through transparent communication is key to building
  a healthy ecosystem together.
- We're all human and we often jump to conclusions. We want people to assume
  good intentions when we do, so we extend the same courtesy.

## All output must be made for human consumption

This needs pointing out because AI output isn't always human-friendly:

- AI-written prose tends to be very verbose, often filled with unnecessary
  details and jargon.
- AI-written code tends to be overly defensive and isolated, often ignoring
  conventions, invariants, and existing utilities.

This applies to code, including comments, and to anything you post online, such
as an implementation plan. The burden is on you to demonstrate that your output
is clean and concise.

## You are solely responsible for your output

- We expect you to be able to defend every single line of change that you
  submit.
- If a mistake is pointed out to you, it's *always* your own mistake. Blaming or
  hiding behind an AI for mistakes in your diff is an explicit avoidance of your
  responsibilities.

Issues, PRs, and comments that are low-effort are likely to be closed or
ignored, whether or not they're AI-generated. Reviewers are under no obligation
to prove you're using AI.

## Keep pull requests small

We limit new contributors to two concurrent open pull requests.

We also ask that you make each pull request as small as possible. When you open
a pull request, check the number of lines modified in GitHub: the smaller the
better, though don't drop tests or docstrings to get there. If your pull request
is over 100 lines, try to split it into multiple pull requests. If you can make
them independent of each other, that's even better, because no synchronization
is needed for the merge.

This guideline exists for the following reasons:

- *Higher quality reviews.* It's much easier to spot a bug in a few lines than
  in 1000 lines.
- *Faster overall review.* To approve a pull request, reviewers need to
  understand every line and how it fits into your overall change. They also need
  to move back and forth between files and functions to follow the flow of the
  code. That gets exponentially harder as the change grows.
- *Avoiding blocking changes that are valid.* In a huge pull request, some
  changes are usually valid and some need to be reworked or discussed. If they
  all sit in the same pull request, the valid changes stay blocked until every
  discussion is resolved.
- *Fewer merge conflicts.* A bigger pull request means a slower review, which
  means the pull request stays open longer and accumulates more conflicts to
  resolve before it can merge.
- *Parallel processing.* Reviewers like to parallelize code reviews to merge
  your code faster. If you open two independent pull requests, two reviewers can
  work on your code at the same time.
- *Finding the time for a code review.* A code review often needs to happen in
  one sitting, because it's hard to hold functions and code logic in your head
  from one session to the next. A big pull request requires the reviewer to
  find a big block of time, which isn't always possible and can delay your
  review and merge by days.

Smaller pull requests mean less work for maintainers and faster reviews and
merges for contributors. It's a win-win.
