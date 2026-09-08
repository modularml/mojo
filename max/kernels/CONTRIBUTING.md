# MAX accelerator library contributor guide

Thank you for your interest in contributing to the MAX accelerator library!
We’re excited to welcome external contributors to help build high-performance
CPU and GPU kernels in Mojo.

## Contributing to the accelerator library

The accelerator library powers key components of MAX and Mojo, and
contributions here can have a meaningful impact across the AI ecosystem. Before
jumping in, please take a moment to review our contribution guidelines and
processes.

We welcome contributions across a wide range of functionality, but **if you’re
proposing a change that could affect the performance of the core kernel
library**, we ask that you first go through our [proposal process](#proposals).

### Changes we accept

We welcome contributions across all hardware platforms and are particularly
interested in Blackwell, Hopper, MI3xx, and other data center GPUs.

In particular, we welcome contributions of the following kernels:

- Batched matrix multiplication (BMM)
- Multi-head latent attention (MLA)
- Mixture of experts (MOE)
- Non-maximum suppression (NMS)
- Grouped matrix multiplication
- 2D convolutions
- General matrix-vector multiply (GEMV) on Hopper

## Proposals

If you’re interested in making a significant change (one that doesn’t fall into
the [list of changes we accept](#changes-we-accept) above), your first step is
a written proposal. The proposal process:

- Ensures feedback from the widest possible set of community members
- Maintains an audit trail of important design decisions
- Provides rationale for changes that impact long-term performance or
architecture

**To submit a proposal, create a GitHub Pull Request that adds a document to the
[`modular/Mojo/proposals`](https://github.com/modular/modular/tree/main/Mojo/proposals)
directory.**

Contributors are encouraged to react with a 👍 to proposal PRs if they support
the high-level direction. Proposals are reviewed and assigned to accelerator
library leads. A proposal can be merged once the lead approves, all blocking
issues are resolved, and relevant decisions are incorporated. If a lead chooses
to defer or reject the proposal, they will explain the reasoning and close the
PR. This process is inspired by successful practices in other open source
communities.

## Submitting pull requests

For details about how to submit a pull request, see the repo's
[primary contributing guide](../../CONTRIBUTING.md).

Thank you for your contributions! ❤️
