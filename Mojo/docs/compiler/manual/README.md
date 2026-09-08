# Mojo 🔥 Compiler Dev Manual

## Introduction

Welcome to the Mojo Compiler Dev Manual! The main goal of this is to help people
who are just getting started in modifying the Mojo compiler.

The Mojo compiler has a lot of similarities to other compilers, but also a lot
of differences. This doc will cover all of it, and link out to further reading
for the more nuanced topics.

## Overview, Intermediate Representations, and the Passes Between Them

The best way to know a compiler is to look at the various passes, and what data
goes into each pass, and what data comes out of each pass.

So we've written an entire page about it! Check out
[Passes and Intermediate Representations](PassesAndIR.md) for what our various
IR stages look like, and how our passes transform from one to the next.

## Terminology

Mojo unlocks incredible powers for abstraction. That means the compiler must
handle incredible amounts of abstraction, and have names for them all.

Some of the terminology we use here is rather non-standard, sometimes because of
historical reasons, sometimes because no such concept exists yet in the outside
world, and sometimes because we're too clever for our own good.

Check out [Terminology](Terminology.md) for all the unusual terms we use here,
to make it easier to understand Modular conversations and code.

## Common Types and Tools

There are a lot of basic types and tools that are used pervasively throughout
the compiler. Arm yourself! Read
[Common Types and Tools](CommonTypesAndTools.md).

## Mojo ↔ IR ↔ C++ Correspondence

If you want to know how Mojo code looks after it's transformed into IR, or you
want to know how to generate IR in the parser, you'll like this section.

See [Mojo ↔ IR ↔ C++ Correspondence](MojoIRCPPCorrespondence.md) for how various
given Mojo snippets compile to IR, and what C++ one would use in the compiler to
generate that same IR.

## Debugging

Things will go wrong in glorious and destructive ways, because it's a compiler.
You'll want some good tools for investigating and debugging.

Check out [Parser Debugging](ParserDebugging.md)! A lot of those tricks are
applicable to other passes as well.

Check out [Post Parser Debugging](PostParserDebugging.md) for tools and tips to
debug generated MLIR, LLVM IR, AIR or binary output.
