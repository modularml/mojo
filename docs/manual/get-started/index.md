---
title: Get started with Mojo🔥
sidebar_label: Get Mojo
sidebar_position: 1
description: Get the Mojo SDK or try coding in the Mojo Playground.
css: /static/styles/page-navigation.css
aliases:
  - /mojo/get-started.html
  - /mojo/manual/get-started/setup.html
website:
  open-graph:
    image: /static/images/mojo-social-card.png
  twitter-card:
    image: /static/images/mojo-social-card.png
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The Mojo SDK is currently available for Ubuntu Linux systems and macOS
systems running on Apple silicon. Support for Windows is
coming soon. You can also develop from Windows or Intel macOS using a container
or remote Linux system. Alternatively, you can also experiment with Mojo using
our web-based [Mojo Playground](#develop-in-the-mojo-playground).

## Get the Mojo SDK

:::note Get Mojo in MAX!

To provide a unified toolkit for AI developers, the Mojo SDK is now included in
the [MAX SDK](/max). To install the MAX SDK, see
[Get started with MAX Engine](/engine/get-started). If you want to install the
standalone Mojo SDK, you're in the right place.

:::

The Mojo SDK includes everything you need for local Mojo development, including
the Mojo standard library and the [Mojo command-line interface](/mojo/cli/)
(CLI). The Mojo CLI can start a REPL programming environment, compile and run
Mojo source files, format source files, and more.

We've also published a [Mojo extension for Visual Studio
Code](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
to provide a first-class developer experience with features like code
completion, quick fixes, and hover help for Mojo APIs.

![](./images/mojo-vscode.png)

### System requirements

To use the Mojo SDK, you need a system that meets these specifications:

Linux:

- Ubuntu 20.04/22.04 LTS
- x86-64 CPU (with [SSE4.2 or
newer](https://www.intel.com/content/www/us/en/support/articles/000057621/processors.html))
  or AWS Graviton2/3 CPU
- Minimum 8 GiB RAM
- Python 3.8 - 3.11
- g++ or clang++ C++ compiler

Mac:

- Apple silicon (M1 or M2 processor)
- macOS Monterey (12) or later
- Python 3.8 - 3.11
- Command-line tools for Xcode, or Xcode

Support for Windows will be added in a future release.

### Install Mojo

:::tip Already have modular?

If you already have the `modular` tool,
[update](/cli/#description) to version 0.5.1 or newer, and go to step 2.

:::

1. Open a terminal and install the [`modular`](/cli/) command line tool:

    ```sh
    curl -s https://get.modular.com | sh -
    ```

2. Then install the Mojo SDK:

    ```sh
    modular install mojo
    ```

    :::note Get nightlies

    If you want the bleeding-edge (less stable) version, instead install the
    nightly build:

    ```sh
    modular install nightly/mojo
    ```

    :::

3. Set environment variables so you can access the
   [`mojo`](/mojo/cli/) CLI:

    <Tabs>
      <TabItem value="bash" label="Bash">

      If you're using Bash, run this command:

      ```sh
      MOJO_PATH=$(modular config mojo.path) \
        && BASHRC=$( [ -f "$HOME/.bash_profile" ] && echo "$HOME/.bash_profile" || echo "$HOME/.bashrc" ) \
        && echo 'export MODULAR_HOME="'$HOME'/.modular"' >> "$BASHRC" \
        && echo 'export PATH="'$MOJO_PATH'/bin:$PATH"' >> "$BASHRC" \
        && source "$BASHRC"
      ```

      </TabItem>
      <TabItem value="zsh" label="ZSH">

      If you're using ZSH, run this command:

      ```sh
      MOJO_PATH=$(modular config mojo.path) \
        && echo 'export MODULAR_HOME="'$HOME'/.modular"' >> ~/.zshrc \
        && echo 'export PATH="'$MOJO_PATH'/bin:$PATH"' >> ~/.zshrc \
        && source ~/.zshrc
      ```

      </TabItem>
    </Tabs>

Next, get started with **[Hello, world!](hello-world.html)**

If you have issues during install, check our [known
issues](/mojo/roadmap.html#mojo-sdk-known-issues).

:::note

To help us improve Mojo, we collect some basic system information and
crash reports. [Learn
more](/mojo/faq.html#does-the-mojo-sdk-collect-telemetry).

:::

### Update Mojo

Mojo is a work in progress and we will release regular updates to the
Mojo language and SDK tools. For information about each release, see the
[Mojo changelog](/mojo/changelog.html).

To check your current Mojo version, use the `--version` option:

```sh
mojo --version
```

To update to the latest Mojo version, use the `modular update` command:

```sh
modular update mojo
```

### Update the Modular CLI

We may also release updates to the `modular` tool. Run the following
commands to update the CLI on your system.

Linux:

```sh
sudo apt update
```

```sh
sudo apt install modular
```

Mac:

```sh
brew update
```

```sh
brew upgrade modular
```

## Develop in the Mojo Playground

Instead of downloading the Mojo SDK, you can also experiment with Mojo in our
online [Playground](/mojo/playground).

### What to expect

The Mojo Playground is a simple online editor where you can test out Mojo
code for yourself.

- We've included a handful of code examples to show you Mojo basics and
  demonstrate its capabilities.

- This is an online sandbox and not useful for benchmarking.

- You can download your code or share it as a gist, but there's no mechanism
  for saving code in the Playground itself. Any changes will be lost when you
  switch code examples (as well as in the event of a server refresh or update).
  If you come up with something you want to save—save it locally!

- The Playground environment doesn't include any Python packages. In the future
  we intend to make some common Python packages available to import in the
  Playground.

- There might be some bugs. Please [report issues and feedback on
  GitHub](https://github.com/modularml/mojo/issues/new/choose).

### Caveats

- The Mojo environment does not have network access, and you cannot install any
  Mojo or Python packages. You only have access to Mojo and the Mojo standard
  library.

- For a general list of things that don't work yet in Mojo or have pain-points,
  see the [Mojo roadmap and sharp edges](/mojo/roadmap.html).
