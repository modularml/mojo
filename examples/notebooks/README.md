# Mojo Jupyter notebooks

Mojo supports programming in [Jupyter notebooks](https://jupyter.org/), just
like Python.

This page explains how to get started with Mojo notebooks, and this repo
directory contains notebooks that demonstrate some of Mojo's features
(most of which we originally published on the [Mojo
Playground](https://playground.modular.com/)).

If you're not familiar with Jupyter notebooks, they're files that allow you to
create documents with live code, equations, visualizations, and explanatory
text. They're basically documents with executable code blocks, making them
great for sharing code experiments and programming tutorials. We actually wrote
the [Mojo programming
manual](https://docs.modular.com/mojo/programming-manual.html) as a Jupyter
notebook, so we can easily test all the code samples.

And because Mojo allows you to import Python modules, you can use visualization
libraries in your notebooks to draw graphs and charts, or display images. For
an example, check out the `Mandelbrot.ipynb` notebook, which uses `matplotlib`
to draw the Mandelbrot set calculated in Mojo, and the `RayTracing.ipynb`
notebook, which draws images using `numpy`.

## Get started in VS Code

Visual Studio Code is a great environment for programming with Jupyter
notebooks. Especially if you're developing with Mojo on a remote system, using
VS Code is ideal because it allows you to edit and interact with notebooks on
the remote machine where you've installed Mojo.

All you need is the Mojo SDK and the Jupyter VS Code extension:

1. Install the [Mojo SDK](https://developer.modular.com/download).

2. Install [Visual Studio Code](https://code.visualstudio.com/) and the
   [Jupyter
   extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

3. Then open any `.ipynb` file with Mojo code, click **Select Kernel** in the
   top-right corner of the document, and then select **Jupyter Kernel > Mojo**.

   The Mojo kernel should have been installed automatically when you installed
   the Mojo SDK. If the Mojo kernel is not listed, make sure that your
   `$MODULAR_HOME` environment variable is set on the system where you
   installed the Mojo SDK (specified in the `~/.profile` or `~/.bashrc` file).

   Now run some Mojo code!

## Get started with JupyterLab

You can also select the Mojo kernel when running notebooks in a local instance
of JupyterLab. The following is just a quick setup guide for Linux users with
the Mojo SDK installed locally, and it might not work with your system (these
instructions don't support remote access to the JupyterLab). For more details
about using JupyterLab, see the complete [JupyterLab installation
guide](https://jupyterlab.readthedocs.io/en/latest/getting_started/installation.html).

**Note:** You must run this setup on the same machine where you've installed
the [Mojo SDK](https://developer.modular.com/download). However, syntax
highlighting for Mojo code is not currently enabled in JupyterLab (coming soon).

1. Install JupyterLab:

    ```sh
    python3 -m pip install jupyterlab
    ```

2. Make sure the user-level `bin` is in your `$PATH`:

    ```sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

3. Launch JupyterLab:

    ```sh
    jupyter lab
    ```

4. When you open any of the `.ipynb` notebooks from this repository, JupyterLab
   should automatically select the Mojo kernel (which was installed with the
   Mojo SDK).

   Now run some Mojo code!

## Notes and tips

- Code in a Jupyter notebook cell behaves like code in a Mojo REPL environment:
  The `main()` function is not required, but there are some caveats:

  - Top-level variables (variables declared outside a function) are not visible
    inside functions.

  - Redefining undeclared variables is not supported (variables without a `let`
    or `var` in front). If you’d like to redefine a variable across notebook
    cells, you must declare the variable with either `let` or `var`.

- You can use `%%python` at the top of a code cell and write normal Python
  code. Variables, functions, and imports defined in a Python cell are available
  from subsequent Mojo code cells.
