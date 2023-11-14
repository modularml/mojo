---
title: Hello, world!
description: Learn to run your first Mojo program.
css: /static/styles/page-navigation.css
website:
  open-graph:
    image: /static/images/mojo-social-card.png
  twitter-card:
    image: /static/images/mojo-social-card.png
---

After you [install Mojo](/mojo/manual/get-started/setup.html), you can use the
[Mojo CLI](/mojo/cli/) to build and compile Mojo programs. So let's create the
classic starter program that prints "Hello, world!"

:::{.callout-note}

**Before you start:**

You must set the `MODULAR_HOME` and `PATH` environment variables, as described
in the output when you ran `modular install mojo`. For example, if you're using
bash or zsh, add the following lines to your configuration file
(`.bash_profile`, `.bashrc`, or `.zshrc`):

```sh
export MODULAR_HOME="$HOME/.modular"
export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"
```

Then source the file you just updated, for example:

```sh
source ~/.bash_profile
```

If you have other issues during install, check our [known
issues](/mojo/roadmap.html#mojo-sdk-known-issues).

:::

## Run code in the REPL

First, let's try running some code in the Mojo
[REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop),
which allows you to write and run Mojo code directly in a command prompt:

1. To start a REPL session, type `mojo` in your terminal and press
   <kbd>Enter</kbd>.

2. Then type `print("Hello, world!")` and press <kbd>Enter</kbd> twice
(a blank line is required to indicate the end of an expression).

That's it! For example:

```text
$ mojo
Welcome to Mojo! 🔥

Expressions are delimited by a blank line.
Type `:quit` to exit the REPL and `:mojo help` for further assistance.

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

## Build and run Mojo source files

Now let's print "Hello, world" with a source file. Mojo source files are
identified with either the `.mojo` or `.🔥` file extension.

You can quickly execute a Mojo file by passing it to the `mojo` command, or you
can build a compiled executable with the `mojo build` command. Let's try both.

### Run a Mojo file

First, write the Mojo code and execute it:

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
Mojo](/mojo/manual/get-started/#install-mojo).

### Build an executable binary

Now, build and run an executable:

1. Create a stand-alone executable with the `build` command:

    ```sh
    mojo build hello.mojo
    ```

    It creates the executable with the same name as the `.mojo` file, but
    you can change that with the `-o` option.

2. Then run the executable:

    ```sh
    ./hello
    ```

The executable runs on your system like any C or C++ executable.

## Next steps

- If you're developing in VS Code, install the [Mojo
  extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
  so you get syntax highlighting, code completion, diagnostics, and more.

- If you're new to Mojo, read the [Mojo language basics](/mojo/manual/basics/).

- If you want to package your code as a library, read about
  [Mojo modules and packages](/mojo/manual/get-started/packages.html).

- If you want to explore some Mojo code, clone our repo to see some examples:

  ```sh
  git clone https://github.com/modularml/mojo.git
  ```

  Then open the `/examples` directory in your IDE to try our examples:

  - The [code examples](https://github.com/modularml/mojo/tree/main/examples/)
    offer a variety of demos with the standard library to help you
    learn Mojo's features and start your own projects.

  - The [Mojo
  notebooks](https://github.com/modularml/mojo/tree/main/examples/notebooks#readme)
  are the same Jupyter notebooks we publish in the [Mojo
  Playground](https://playground.modular.com), which demonstrate a variety of
  language features. Now with the Mojo SDK, you can also run them in VS Code or in
  JupyterLab.

- For a deep dive into the language, check out the [Mojo programming
  manual](/mojo/programming-manual.html).

- To see all the available Mojo APIs, check out the [Mojo standard library
  reference](/mojo/lib.html).

:::{.callout-note}

**Note:** The Mojo SDK is still in early development, but you can expect
constant improvements to both the language and tools. Please see
the [known issues](/mojo/roadmap.html#mojo-sdk-known-issues) and [report any
other issues on GitHub](https://github.com/modularml/mojo/issues/new/choose).

:::
