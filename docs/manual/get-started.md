---
title: Get started with Mojo🔥
sidebar_label: Get started
description: Install Mojo now and start developing
---

On this page, we'll show you how to create the classic "Hello world" starter
program with Mojo, in three different ways. If you'd rather read how to write
Mojo code beyond just printing text, see the [introduction to
Mojo](/mojo/manual/basics).

:::tip Updating?

If you already installed Mojo, see the [update guide](/max/update).

:::

## 1. Install Mojo

Mojo is now bundled with MAX, which provides everything to compile,
run, debug, and package Mojo code ([read
why](/engine/faq#why-bundle-mojo-with-max)).

To install Mojo, [see the MAX install guide](/max/install).

## 2. Run code in the REPL

Now that you've installed Mojo, let's write some code!

First, let's use the Mojo
[REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop),
which allows you to write and run Mojo code in a command prompt:

1. To start a REPL session, type `mojo` in your terminal and press
   <kbd>Enter</kbd>.

2. Then type `print("Hello, world!")` and press <kbd>Enter</kbd> twice
(a blank line is required to indicate the end of an expression).

That's it! For example:

```text
$ mojo
Welcome to Mojo! 🔥

Expressions are delimited by a blank line.
Type `:quit` to exit the REPL and `:mojo help repl` for further assistance.

1> print("Hello, world!")
2.
Hello, world!
```

You can write as much code as you want in the REPL. You can press
<kbd>Enter</kbd> to start a new line and continue writing code, and when you
want Mojo to evaluate the code, press <kbd>Enter</kbd> twice. If there's
something to print, Mojo prints it and then returns the prompt to you.

The REPL is primarily useful for short experiments because the code isn't
saved. So when you want to write a real program, you need to write the code in
a `.mojo` source file.

## 3. Run a Mojo file

Now let's write the code in a Mojo source file and run it with the
[`mojo`](/mojo/cli/) command:

1. Create a file named `hello.mojo` (or `hello.🔥`) and add the following code:

   ```mojo
   fn main():
       print("Hello, world!")
   ```

   That's all you need. Save the file and return to your terminal.

2. Now run it with the `mojo` command:

    ```sh
    mojo hello.mojo
    ```

    It should immediately print the message:

    ```text
    Hello, world!
    ```

If this didn't work for you, double-check your code looks exactly like the code
in step 1, and make sure you correctly [installed
MAX](/max/install) (it includes Mojo).

## 4. Build an executable binary

Finally, let's build and run that same code as an executable:

1. Create an executable file with the [`build`](/mojo/cli/build) command:

    ```sh
    mojo build hello.mojo
    ```

    The executable file uses the same name as the `.mojo` file, but
    you can change that with the `-o` option.

2. Then run the executable:

    ```sh
    ./hello
    ```

This creates a statically compiled binary file, so it contains all the code and
libraries it needs to run.

## 5. Install our VS Code extension (optional)

To provide a first-class developer experience with features like code
completion, quick fixes, and hover help, we've created a [Mojo extension for
Visual Studio
Code](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo).

![](./images/mojo-vscode.png)

## Next steps

- If you're new to Mojo, we suggest you learn the language basics in the
  [introduction to Mojo](/mojo/manual/basics).

- If you want to experiment with some code, clone [the Mojo
repo](https://github.com/modularml/mojo/) to try our code examples:

  ```sh
  git clone https://github.com/modularml/mojo.git
  ```

  In addition to several `.mojo` examples, the repo includes [Jupyter
  notebooks](https://github.com/modularml/mojo/tree/main/examples/notebooks#readme)
  that teach advanced Mojo features.

- To see all the available Mojo APIs, check out the [Mojo standard library
  reference](/mojo/lib).

If you have issues during install, check our [known
issues](/mojo/roadmap#mojo-sdk-known-issues).

:::note

To help us improve Mojo, we collect some basic system information and
crash reports. [Learn
more](/mojo/faq#does-the-mojo-sdk-collect-telemetry).

:::
