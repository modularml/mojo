# Why Bazel?

Build systems are complicated. They are what every developer interacts with when
working in a codebase, and thus it's important they they get many things
correct: speed, ease of use, correctness, etc. Bazel is one such build system,
and this doc aims to identify why we use it over other build systems.

## Emphasis on Speed and Correctness

Bazel's slogan is literally "{fast, correct}, pick two". It focuses specifically
having hermetic, reproducible builds. That is, if one person builds something in
one place, someone else should be able to build the same thing in the exact same
way, no matter what their machine is (aside from fundamental differences, like
OS or architecture). If you've ever wondered "why doesn't this work on my
machine?", Bazel does a _very_ thorough job of making sure that never happens.

To do this, Bazel is very strict about inputs and outputs. If you provide some
inputs and a function, you'll get some outputs. This also allows it to be very
fast, as by having a complete picture of the build at all times it knows exactly
when to invalidate and rebuild things. It even can leverage sandboxes to make
sure the tools are isolated from the rest of the system.

Bazel also supports remote execution and caching, which allows developers (with
access to the system) instant (if cached) or very fast (if remotely executed)
builds that aren't limited by the local machine's resources.

Generally, Bazel is so correct that `bazel clean` is rarely necessary, usually
only necessary for freeing up drive space and working around actual build system
bugs. Most developers should never need to use this.

This isn't for free, though. Build system engineers need to make sure that the
tools they use are hermetic, as Bazel assumes this for all of its optimizations.
For example, C++ compilers are very eager to use system configuration, in the
Modular codebase we use a vendored toolchain to make sure everyone compiling C++
code uses the same compiler, with the same stdlib, and the same flags. These
flags include `-isysroot` to point the compiler at a vendored sysroot, and
`-D__TIME__=` to avoid timestamps being built into binaries. Even things like
the glibc version are entirely contained within Bazel. We also enable sandboxing
as mentioned above to help safeguard against this.

## Multi-language Codebase

Our codebase is primarily three languages: Mojo, C++, and Python. C++ has its
own build systems, as does Python, but Mojo does not. This leaves us with a few
options:

- Build our own build system for Mojo
- Use either a C++ or Python build system for Mojo, and leave the others alone
- Use either a C++ or Python build system for all languages
- Use a general purpose build system for all languages

The first three options have serious drawbacks:

- Writing a build system is an incredible amount of effort, including
  maintaining the ecosystem for it.
- Mojo can be made to work with other build systems, but it forces several
  language-specific build systems to call each other in odd ways. The high level
  structure of our codebase is Python that calls either C++ or Mojo (which in
  turn also calls C++). Having a chain of build systems means that no one tool
  has a full picture of the world, and working in one area of the codebase could
  be different than working in another.
- Again, while Mojo can be adapted to use other build systems, getting every
  language to adopt a single system is an uphill battle. Consider getting Cargo
  to build Go code - probably doable via build.rs, but you lose a lot of
  visibility into what the build is doing.

This leaves us with the final option which is to use a single language-agnostic
system for all languages. Bazel can support any language (as long as someone
writes a ruleset for it), without making assumptions about specific languages.

Note that there are more languages than just those in the codebase! The more we
add to the mix, the more having something that can support anything pays off.

## Build-infrastructure-as-code

"Visibility" is important for builds, but what does that mean? As mentioned
above, having any build action be able to know exactly what is being provided
and exactly what will come out of it is extremely important for speed and
correctness, but there's more we can do with this.

Bazel uses a codified language called Starlark, also sometimes called the
BUILD language. It's (roughly) a subset of Python, which is intentional so that
it's familiar and easy to use. Config-file based build systems are nice, but
it's very common to need to do something custom, such as a genrule (running an
arbitrary script that the build system doesn't know about), writing loops or
conditionals, or even full-blown config file generation. Bazel works the other
way around: it gives you a language that can do all of that, with the ability to
wrap up the underlying plumbing into easy-to-use rules or macros. Given that
this is all code, we can lint it, format it, and version it (`.bazelversion` is
a file specifically for determining what version of Bazel to use!)

It also lets you do build graph introspection, for asking questions like "what
targets does this test depend on?" or "what targets changed between these two
commits?".

## Frequently Asked Questions

### ABC language's tool is better

It may be! `uv` and `pixi` are amazing for Python projects, `CMake`, with its
flaws, is the defacto standard C++ build system and thus gets a lot of support.
Go and Rust's tooling are very streamlined. These are all perfectly valid tools
to use, but they don't work for _our_ repo. (Though it is worth noting that
Bazel relies on some of this language-specific tooling, for example we use `uv`
to solve + generate `uv.lock`files for pulling Python dependencies into the
build.)

Bazel does not intend to be the "best build system for everything", but rather,
it is intended to be very good _overall_, and then have a unified interface so
that a C++ developer doesn't have to ask "I've modified some Python code, how do
I run it?". Instead, they just build/test the targets as they would any other.

Our goal is not to make _one_ developer 50% faster, it's to make _all_
developers 10% faster.

### Bazel is too strict/complicated/hard to use

There is some truth to this, however there are reasons behind this:

- It is strict in order to be correct. See the above note on not needing to use
  `bazel clean`. If we didn't declare inputs, we could be secretly be using
  anything for any build or test! How would we know when to rebuild/retest?
- Rules, for example, at first glance look needlessly complex. The
  implementation is split from the declaration so that Bazel can work in three
  phases, which helps with speed. This is one such example.
- Many languages use config files to build. As mentioned above, we frequently
  have the need to do something "custom" for some targets. For our codebase,
  this is acceptable, but other codebases may not, and that's okay!
- Bazel's CLI is mainly designed for build engineers, so the out-of-the-box
  experience might not be ideal for many end users. We provide many primitives
  such as `./bazelw run format` to bridge this gap.
