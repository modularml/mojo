---
title: "Mojo Manual"
sidebar_label: Introduction
description: A comprehensive guide to the Mojo programming language.
aliases:
  - /mojo/programming-manual.html
css: /static/styles/page-navigation.css
website:
  open-graph:
    image: /static/images/mojo-social-card.png
  twitter-card:
    image: /static/images/mojo-social-card.png
---

Welcome to the Mojo Manual, a complete guide to the Mojo🔥 programming language!

Mojo is designed to solve a variety of AI development challenges that no other
language can, because Mojo is the first programming language built from the
ground-up with [MLIR](https://mlir.llvm.org/) (a compiler infrastructure that's
ideal for heterogeneous hardware, from CPUs and GPUs, to various AI ASICs). We
also designed Mojo as a superset of Python because we love Python and its
community, but we couldn't realistically enhance Python to do all the things we
wanted. For a longer discussion on this topic, read [Why
Mojo](/mojo/why-mojo.html).

Beware that Mojo is still a very young language, so there's a lot that hasn't
been built yet. Likewise, there's a lot of documentation that hasn't been
written yet. But we're excited to share Mojo with you and [get your
feedback](/mojo/community.html).

## Contents

- **Get started**

  - [Get started with Mojo](get-started/index.html)
  - [Hello World!](get-started/hello-world.html)

- **Language basics**

  - [Introduction to Mojo](basics.html)
  - [Functions](functions.html)
  - [Variables](variables.html)
  - [Structs](structs.html)
  - [Modules and packages](packages.html)

- **Value ownership**

  - [Intro to value ownership](values/index.html)
  - [Value semantics](values/value-semantics.html)
  - [Ownership and borrowing](values/ownership.html)

- **Value lifecycle**

  - [Intro to value lifecycle](lifecycle/index.html)
  - [Life of a value](lifecycle/life.html)
  - [Death of a value](lifecycle/death.html)

- **Traits and parameters**

  - [Traits](traits.html)
  - [Parameterization: compile-time metaprogramming](parameters/index.html)

- **Python**

  - [Python integration](python/index.html)
  - [Python types](python/types.html)

- **Tools**

  - [Debugging](../tools/debugging.html)

- **Project information**

  - [Roadmap and sharp edges](../roadmap.html)
  - [Changelog](../changelog.html)
  - [FAQ](../faq.html)
  - [Community](../community.html)
