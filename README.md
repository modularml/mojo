<div align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/mojo_github_logo_bg.png">

  [Website][Mojo] | [Getting Started] | [API Documentation] | [Contributing] | [Changelog]
</div>

[Mojo]: https://www.modular.com/mojo/
[Getting Started]: https://docs.modular.com/mojo/manual/get-started/
[API Documentation]: https://docs.modular.com/mojo/lib
[Contributing]: ./CONTRIBUTING.md
[Changelog]: ./docs/changelog.md

# Welcome to Mojo 🔥

Mojo is a new programming language that bridges the gap between research
and production by combining Python syntax and ecosystem with systems
programming and metaprogramming features. Mojo is still young, but it is designed
to become a superset of Python over time.

This repo includes source code for:

- Mojo examples
- Mojo documentation hosted at [modular.com](https://docs.modular.com/mojo/)
- The [Mojo standard library](https://docs.modular.com/mojo/lib)

This repo has two primary branches:

- The [`main`](https://github.com/modularml/mojo/tree/main) branch, which is in
sync with the last stable released version of Mojo. Use the examples here if you’re
using a [release build of Mojo](#latest-released).

- The [`nightly`](https://github.com/modularml/mojo/tree/nightly) branch, which
is in sync with the Mojo nightly build and subject to breakage. Use this branch
for [contributions](./CONTRIBUTING.md), or if you're using the latest
[nightly build of Mojo](#latest-nightly).

To learn more about Mojo, see the
[Mojo Manual](https://docs.modular.com/mojo/manual/).

## Installing Mojo

### Latest Released

To install the last released build of Mojo, you can install the MAX SDK
or the standalone Mojo SDK:

- [Get the MAX SDK](https://docs.modular.com/engine/get-started)
- [Get the Mojo SDK](https://docs.modular.com/mojo/manual/get-started/)

Then follow the docs to [write your first Mojo
program](https://docs.modular.com/mojo/manual/get-started/hello-world).

### Latest Nightly

The nightly Mojo builds are subject to breakage and provide an inside
view of how the development of Mojo is progressing.  Use at your own risk
and be patient!

To get nightly builds, see the same instructions to [install the Mojo
SDK](https://docs.modular.com/mojo/manual/get-started/#install-mojo), but use
the command shown there to install `nightly/mojo`.

When you clone this repo, be sure you switch to the `nightly` branch, because
the `main` branch is for stable releases and might not be compatible with
nightly builds:

```bash
git clone https://github.com/modularml/mojo.git
```

```bash
git checkout nightly
```

## Contributing

When you want to report issues or request features, [please create a GitHub
issue here](https://github.com/modularml/mojo/issues).
See [here](./CONTRIBUTING.md) for guidelines on filing good bugs.

We welcome contributions to this repo on the
[`nightly`](https://github.com/modularml/mojo/tree/nightly)
branch. If you’d like to contribute to Mojo, please first read our [Contributor
Guide](https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md).

For more general questions or to chat with other Mojo developers, check out our
[Discord](https://discord.gg/modular).

## License

This repository is licensed under the Apache License v2.0 with LLVM Exceptions
(see the LLVM [License](https://llvm.org/LICENSE.txt)).

## Thanks to our contributors

<a href="https://github.com/modularml/mojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modularml/mojo" />
</a>
