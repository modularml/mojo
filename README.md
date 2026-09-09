<!-- rumdl-disable-file MD033 MD041 MD075 -->

<div align="center">
    <img src="https://modular-assets.s3.us-east-1.amazonaws.com/images/modular-banner-github.png">

[About Modular] | [MAX docs] | [Mojo docs] | [Contributing]

</div>

---

# Modular Platform

This repo hosts open-source components of the Modular Platform,
a unified platform for AI development and deployment,
including the **MAX Framework**🧑‍🚀 and the **Mojo Language**🔥.

## Get started

To get started with the Modular Platform and serve a model using the MAX
framework, see
[the MAX quickstart guide](https://max.modular.com/get-started).

To get started with the Mojo language, see
[the Mojo quickstart guide](https://mojolang.org/docs/manual/quickstart/).

## About the repo

We're constantly open-sourcing more of the Modular Platform and you can find
all of it in here.

The main components include:

- Mojo compiler: [/Mojo](/Mojo)
- Mojo standard library: [/Mojo/stdlib](/Mojo/stdlib)
- MAX accelerator library: [/max/kernels](/max/kernels)
- MAX inference server: [/max/python/max/serve](/max/python/max/serve)
  (OpenAI-compatible endpoint)
- MAX model pipelines: [/max/python/max/pipelines](/max/python/max/pipelines)
  (Python-based graphs)
- Code examples: [/max/examples](/max/examples) +
  [/Mojo/examples](/Mojo/examples)

## Contribute

We accept contributions to the [Mojo standard library](./mojo), [MAX
accelerator library](./max/kernels), [MAX model
architectures](/max/python/max/pipelines/architectures), code examples, Mojo
docs, and more. We aren't accepting contributions to the Mojo compiler yet.

First, please read the [Contribution Guide](./CONTRIBUTING.md), and then refer
to the following documentation about how to develop in the repo:

- [`/max/docs`](/max/docs): Docs for developers working in the MAX framework
  codebase.
- [`/Mojo/docs/stdlib`](/Mojo/docs/stdlib): Docs for developers working in the
  Mojo standard library.

We also welcome your bug reports. If you have a bug, please [file an issue
here](https://github.com/modular/modular/issues/new/choose).

## License

This repository and its contributions are licensed under the Apache License v2.0
with LLVM Exceptions. See [LICENSE](LICENSE).

MAX usage and distribution are licensed under the
[Modular Community License](https://www.modular.com/legal/community).

### Third party licenses

You are entirely responsible for checking and validating the licenses of third
parties (for example, Hugging Face) for related software and libraries that are
downloaded.

## Community & Events

Get help from community members, tune in for a community meeting, or join a
local meetup.

| Channel               | Link                                            |
|-----------------------|-------------------------------------------------|
| 💬 Discord            | [discord.gg/modular][discord]                   |
| 💬 Forum              | [forum.modular.com][forum]                      |
| 📅 Meetup Group       | [meetup.com/modular-meetup-group][meetup-group] |
| 🎦 Community Meetings | [Upcoming community calls][public-com-meet-doc] |
| 📺 YouTube            | [youtube.com/@modularinc][youtube]              |

**Upcoming events** will be posted on our [Meetup page][meetup-group] and
[Discord][discord]. Community meeting recordings will be posted on our
[YouTube][youtube].

## Thanks to our contributors

<a href="https://github.com/modular/modular/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modular/modular" />
</a>

<!-- Link references -->

<!-- Header navigation links -->
[About Modular]: https://www.modular.com/
[MAX docs]: https://max.modular.com/
[Contributing]: ./CONTRIBUTING.md
[Mojo docs]: https://mojolang.org/docs/

<!-- Community & Events links -->
[discord]: https://discord.gg/modular
[forum]: https://forum.modular.com/
[meetup-group]: https://www.meetup.com/modular-meetup-group/
[youtube]: https://www.youtube.com/@modularinc
[public-com-meet-doc]: https://modul.ar/community-meeting-doc
