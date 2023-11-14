---
title: "Mojo Manual"
description: A comprehensive guide to the Mojo programming language.
css: /static/styles/page-navigation.css
website:
  open-graph:
    image: /static/images/mojo-social-card.png
  twitter-card:
    image: /static/images/mojo-social-card.png
---

Welcome to the Mojo Manual, a complete guide to the Mojo🔥 programming language!

Mojo is still a very young language, so there's a lot that hasn't been built
yet. Likewise, there's also a lot of documentation that hasn't been written yet.
In time, this Mojo Manual will include everything you need to know about Mojo.

You can read the manual in any order you want, but it is designed to
progressively built upon concepts. So if you want a complete understanding of
Mojo, we suggest reading it in the order it's presented.

## Why we created Mojo

Mojo is designed to solve a variety of AI development challenges that no other
language can, partly because Mojo is the first programming language built from
the ground-up with the [MLIR](https://mlir.llvm.org/) compiler infrastructure.
This is important because the strength of MLIR is in its ability to build
domain-specific compilers, particularly for weird domains that aren’t
traditional CPUs and GPUs, such as AI ASICs, quantum computing systems, FPGAs,
and custom silicon—all of which are important for AI deployments.

At Modular, we also wanted a single programming language to write code across
the entire AI software stack, from applications all the way to the ML operation
kernels. Currently, there's no other language that can do this (without
significant difficulty).

We designed Mojo as a superset of the Python because we love Python and its
community, but we couldn't realistically enhance Python to do all the things we
wanted.

For a longer discussion on this topic, read [Why Mojo](/mojo/why-mojo.html).

## Who it's for

We're building Mojo for everybody, including first-time programmers,
application developers, systems engineers, machine learning researchers,
kernel engineers, and everybody in between.

However, because Mojo is still very young, it's currently best suited for
people who want to explore new frontiers of what a programming language is
capable of, in terms of performance and flexibility. There is still a lot that
remains to be built, but we're excited to share what we have and listen to your
feedback.
