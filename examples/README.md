# Mojo code examples

A collection of sample programs and Mojo notebooks written in the  
[Mojo](https://docs.modular.com/mojo/programming-manual.html) programming language.

## Getting Started

Access a Mojo programming environment available from the  
Mojo product [page](https://www.modular.com/mojo).

Git clone the repository of Mojo samples using the command below:

```bash
git clone https://github.com/modularml/mojo.git
```

## Running

Use the following sample command-line to run the programs:

```bash
mojo matmul.mojo
```

You can run the Mojo notebooks using [JupyterLab or Visual Studio  
Code](notebooks/README.md) with the Mojo extension available on the Marketplace.

### Mojo SDK Container

The repo also contains a Dockerfile that can be used to create a  
Mojo SDK container for developing and running Mojo programs. Use the  
container in conjunction with the Visual Studio Code devcontainers  
extension to develop directly inside the container.

The Dockerfile also sets up a `conda` environment and by default,  
starts a `jupyter` server (which you can access via the browser).

## License

The Mojo examples and notebooks in this repository are licensed  
under the Apache License v2.0 with LLVM Exceptions  
(see the LLVM [License](https://llvm.org/LICENSE.txt)).

## Contributing

Thanks for your interest in contributing to this repository!  
We are not accepting pull requests at this time, but are actively  
working on a process to accept contributions. Please stay tuned.
