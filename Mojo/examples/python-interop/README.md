# Calling Mojo functions from Python

Mojo now has the ability to be called from Python, with Mojo functions
representing themselves as if they were native Python functions. When a Python
script is run that references a Mojo module, that module is transparently
compiled to present an interface that can be used from Python. At present,
the interface that is presented to Python must be manually defined in Mojo
code.

A full description of this functionality can be found
[within the Mojo manual](https://mojolang.org/docs/manual/python/mojo-from-python/),
including current
[known limitations](https://mojolang.org/docs/manual/python/mojo-from-python/#known-limitations).

These examples show how to perform basic to more advanced use of Mojo code from
Python in order to progressively replace hotspots in Python code with fast
Mojo. This includes using Mojo to drive calculations on
[MAX-compatible GPUs](https://max.modular.com/packages/#gpu-compatibility).

The two examples of Mojo functions being called from Python consist of:

- **hello.py**: A literal "hello world" example, where a string is passed in
  from Python to Mojo, concatenated with a string in Mojo code, and the result
  passed back to Python to be printed.
- **mandelbrot.py**: A parallel calculation of the number of iterations to
  escape in the Mandelbrot set, performed on a MAX-compatible GPU in Mojo. The
  Python code calls into the Mojo GPU calculation and gets the results back as
  ASCII art in a string to be printed.

You can run these examples via [Pixi](https://pixi.sh):

```sh
pixi run hello
pixi run mandelbrot
```

or directly in a Python virtual environment where the `max` PyPI package has
been installed:

```sh
python hello.py
python mandelbrot.py
```
