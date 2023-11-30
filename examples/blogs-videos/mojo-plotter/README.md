# Mojo Plotter

## Installation

### Conda

If you don't have `conda`, install [miniconda here](https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install)

### Create conda environment

Create and acivate conda enironment:

#### General

```bash
conda env create -f environment.yaml
conda activate mojo-plotter
```

### Auto Set Mojo Environment

To automatically set Mojo to use the python environment when you activate it:

#### Macos/Linux

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
export MOJO_PYTHON_LIBRARY="$(find $CONDA_PREFIX/lib -iname 'libpython*.[s,d]*' | sort -r | head -n 1)"
echo "export MOJO_PYTHON_LIBRARY=\"$MOJO_PYTHON_LIBRARY\"" > $CONDA_PREFIX/etc/conda/activate.d/export-mojo.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset MOJO_PYTHON_LIBRARY" > $CONDA_PREFIX/etc/conda/deactivate.d/unset-mojo.sh
```

## Usage

Simply activate the environment and run the program:

```bash
conda activate mojo-plotter
mojo main.mojo
```
