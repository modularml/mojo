# Mojo Actor System Proposal

* _Author_: [Reid Spencer](https://github.com/reid-spencer)
* _Prior Work_: [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782)
* _Date_: September-December 2023
* _Project Working Name_ : Moxy

## Overview
Mojo gives us the opportunity to re-imagine how an Actor System may 
work in the world of heterogeneous computing. Actor Systems have been 
actively researched since the 1970s and have been in use in a variety of
manifestations since then. This proposal wants to carry that model forward into
the 2020s and beyond.
 
We outline here a long-term view of how to tackle a very large problem.
It explores *one possible* approach to adding an actor model to Mojo. 
Importantly, we also outline how this can be done in careful iterations
over time.

## Participation
This proposal hopes to catalyze positive discussion that leads Mojicians to a
best-possible design for actors in Mojo. It is not intended to be a
specification for how actors will be implemented in Mojo but, certainly, that
specification will make reference to this Proposal. This document should
encourage participation from the community of Mojicians interested in an
actor model for Mojo and not, despite its origin, be the exclusive product of
the primary author's mind.

## Contents
* [About Actors](about-actors.md)
  * A refresher on actors and actor systems 
* [Pain Points Addressed](pain-points-addressed.md) 
  * A recitation of the software development pain points that actors alleviate
* [Actor System Features](actor-system-features.md)
  * A list of what we expect Mojo Actor System be
* [Not Actor System Features](not-actor-system-features.md)
  * A list of things we don't want the Mojo Actor System to be or have
* [Mojo Features Needed](mojo-features-needed.md)
  * The features of the Mojo compiler we think are needed to implement the
  Mojo Actor System
* [Expected Optimizations](expected-optimizations.md)
  * An exploration of some of the optimizations we expect from basing an 
  actor model on Mojo 
* [Milestones](milestones.md)
  * A rough estimate of the order in which the goals of this proposal should
    be delivered
