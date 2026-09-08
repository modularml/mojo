# GPU functions written in Mojo

Mojo is a Python-family language built for high-performance computing. It
allows you to write custom algorithms for GPUs without the use of CUDA or other
vendor-specific libraries. The examples in this directory show a few different
ways you can compile and dispatch Mojo functions on a GPU.

These examples are complementary with the other [examples to write custom graph
ops](../custom_ops/) that run on both CPUs and GPUs.

> [!IMPORTANT]
> These examples require a [compatible
> GPU](https://max.modular.com/faq/#gpu-requirements).

The examples include the following:

- **vector_addition.mojo**: A common "hello world" example for GPU programming,
  this adds two vectors together in the same way as seen in Chapter 2 of
  [Programming Massively Parallel Processors](https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors).

- **grayscale.mojo**: The parallelized conversion of an RGB image to grayscale,
  as seen in Chapter 3 of "Programming Massively Parallel Processors".

- **naive_matrix_multiplication.mojo**: An implementation of naive matrix
  multiplication, again inspired by Chapter 3 of "Programming Massively
  Parallel Processors".

- **mandelbrot.mojo**: A parallel calculation of the number of iterations to
  escape in the Mandelbrot set. An example of the same computation performed as
  a custom graph operation can be found [here](../custom_ops/).

- **reduction.mojo**: A highly performant reduction kernel. For a detailed
  explanation see [this
  blogpost](https://veitner.bearblog.dev/very-fast-vector-sum-without-cuda/).

## Setup

1. Make sure your system includes a [compatible
GPU](https://max.modular.com/faq/#gpu-requirements).

2. If you don't have [`pixi`](https://pixi.sh/latest/), install it:

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

3. Clone this repo:

    ```bash
    git clone https://github.com/modular/modular.git
    ```

4. Navigate to these examples:

    ```bash
    cd modular/max/examples/gpu-functions
    ```

## Quickstart

You can run each of the Mojo files using `mojo` via the [`pixi
run`](https://pixi.sh/latest/reference/cli/pixi/run/) command. For example:

```bash
pixi run mojo vector_addition.mojo
```

Or enter the Pixi shell and run each directly:

```bash
pixi shell
```

```bash
mojo vector_addition.mojo
mojo grayscale.mojo
mojo naive_matrix_multiplication.mojo
mojo mandelbrot.mojo
mojo reduction.mojo
```

## Example walkthroughs

Writing individual thread-based functions in Mojo is powered by the [`gpu`
module](https://docs.modular.com/api/mojo/max/gpu/), which handles all the
hardware-specific details of allocating and transferring memory between host
and accelerator, as well as compilation and execution of accelerator-targeted
functions.

The first three examples that follow are common starting points for thread-based
GPU programming. They follow the first three examples in the popular GPU
programming textbook
[*Programming Massively Parallel Processors*](https://www.sciencedirect.com/book/9780323912310/programming-massively-parallel-processors):

- Parallel addition of two vectors
- Conversion of a red-green-blue image to grayscale
- Naive matrix multiplication, with no hardware-specific optimization

The following sections explain some of these examples a bit further.

> [!CAUTION]
> The following code in this README could be out of date. Read the corresponding
> source files for the latest example code.

### Basic vector addition

The common "hello world" example used for data-parallel programming is the
addition of each element in two vectors. Here's how it works in our
`vector_addition.mojo` example:

1. Define the vector addition function.

    The function itself is very simple, running once per thread, adding each
    element in the two input vectors that correspond to that thread ID, and
    storing the result in the output vector at the matching location.

    ```mojo
    def vector_addition(
        lhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
        rhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
        out_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    ):
        tid = thread_idx.x
        out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]
    ```

1. Obtain a reference to the accelerator (GPU) context.

    ```mojo
    ctx = DeviceContext()
    ```

1. Allocate input and output vectors.

    Buffers for the left-hand-side and right-hand-side vectors need to be
    allocated on the GPU and initialized with values.

    ```mojo
    alias float_dtype = DType.float32
    alias VECTOR_WIDTH = 10

    lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    lhs_buffer.enqueue_fill(1.25)
    rhs_buffer.enqueue_fill(2.5)

    lhs_tensor = lhs_tensor.move_to(gpu_device)
    rhs_tensor = rhs_tensor.move_to(gpu_device)
    ```

    A buffer to hold the result of the calculation is allocated on the GPU:

    ```mojo
    out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    ```

1. Compile and dispatch the function.

    The actual `vector_addition()` function we want to run on the GPU is
    compiled and dispatched across a grid, divided into blocks of threads. All
    arguments to this GPU function are provided here, in an order that
    corresponds to their location in the function signature. Note that in Mojo,
    the GPU function is compiled for the GPU at the time of compilation of the
    Mojo file containing it.

    ```mojo
    ctx.enqueue_function[vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        grid_dim=1,
        block_dim=VECTOR_WIDTH,
    )
    ```

1. Return the results.

    Finally, the results of the calculation are moved from the GPU back to the
    host to be examined:

    ```mojo
    with out_buffer.map_to_host() as host_buffer:
        host_tensor = TileTensor(host_buffer, layout)
        print("Resulting vector:", host_tensor)
    ```

To try out this example yourself, run it using the following command:

```sh
pixi run mojo vector_addition.mojo
```

For this initial example, the output you see should be a vector where all the
elements are `3.75`. Experiment with changing the vector length, the block size,
and other parameters to see how the calculation scales.

### Conversion of a color image to grayscale

The `grayscale.mojo` example shows how to convert a red-green-blue (RGB) color
image into grayscale. This uses a rank-3 tensor to host the 2-D image and the
color channels at each pixel. The inputs start with three color channels, and
the output has only a single grayscale channel.

The calculation performed is a common reduction to luminance using weighted
values for the three channels:

```mojo
gray = 0.21 * red + 0.71 * green + 0.07 * blue
```

And here is the per-thread function to perform this on the GPU:

```mojo
def color_to_grayscale(
    rgb_tensor: TileTensor[int_dtype, type_of(rgb_layout), MutAnyOrigin],
    gray_tensor: TileTensor[int_dtype, type_of(gray_layout), MutAnyOrigin],
):
    row = global_idx.y
    col = global_idx.x

    if col < WIDTH and row < HEIGHT:
        red = rgb_tensor[row, col, 0].cast[float_dtype]()
        green = rgb_tensor[row, col, 1].cast[float_dtype]()
        blue = rgb_tensor[row, col, 2].cast[float_dtype]()
        gray = 0.21 * red + 0.71 * green + 0.07 * blue

        gray_tensor[row, col, 0] = gray.cast[int_dtype]()
```

The setup, compilation, and execution of this function is much the same as in
the previous example, but in this case we're using rank-3 instead of rank-1
buffers to hold the values. Also, we dispatch the function over a 2-D grid
of block, which looks like the following:

```mojo
alias BLOCK_SIZE = 16
num_col_blocks = ceildiv(WIDTH, BLOCK_SIZE)
num_row_blocks = ceildiv(HEIGHT, BLOCK_SIZE)

ctx.enqueue_function[color_to_grayscale](
    rgb_tensor,
    gray_tensor,
    grid_dim=(num_col_blocks, num_row_blocks),
    block_dim=(BLOCK_SIZE, BLOCK_SIZE),
)
```

To run this example, run this command:

```sh
pixi run mojo grayscale.mojo
```

This will show a grid of numbers representing the grayscale values for a single
color broadcast across a simple input image. Try changing the image and block
sizes to see how this scales on the GPU.

### Naive matrix multiplication

The `naive_matrix_multiplication.mojo` example performs a very basic matrix
multiplication, with no optimizations to take advantage of hardware resources.
The GPU function for this looks like the following:

```mojo
def naive_matrix_multiplication(
    m: TileTensor[float_dtype, type_of(m_layout), MutAnyOrigin],
    n: TileTensor[float_dtype, type_of(n_layout), MutAnyOrigin],
    p: TileTensor[float_dtype, type_of(p_layout), MutAnyOrigin],
):
    row = global_idx.y
    col = global_idx.x

    m_dim = Int(p.dim[0]())
    n_dim = Int(p.dim[1]())
    k_dim = Int(m.dim[1]())

    if row < m_dim and col < n_dim:
        for j_index in range(k_dim):
            p[row, col] = p[row, col] + m[row, j_index] * n[j_index, col]
```

The overall setup and execution of this function are extremely similar to the
previous example, with the primary change being the function that is run on the
GPU.

To try out this example, run this command:

```sh
pixi run mojo naive_matrix_multiplication.mojo
```

You will see the two input matrices printed to the console, as well as the
result of their multiplication. As with the previous examples, try changing
the sizes of the matrices and how they are dispatched on the GPU.

### Calculating the Mandelbrot set fractal

The `mandelbrot.mojo` example shows a slightly more complex calculation:
[the Mandelbrot set fractal](https://en.wikipedia.org/wiki/Mandelbrot_set).
This custom operation takes no input tensors, only a set of scalar arguments,
and returns a 2-D matrix of integer values representing the number of
iterations it took to escape at that location in complex number space.

The per-thread GPU function for this is as follows:

```mojo
def mandelbrot(
    tensor: TileTensor[int_dtype, type_of(layout), MutAnyOrigin],
):
    row = global_idx.y
    col = global_idx.x

    alias SCALE_X = (MAX_X - MIN_X) / GRID_WIDTH
    alias SCALE_Y = (MAX_Y - MIN_Y) / GRID_HEIGHT

    cx = MIN_X + Float32(col) * SCALE_X
    cy = MIN_Y + Float32(row) * SCALE_Y
    c = ComplexSIMD[float_dtype, 1](cx, cy)
    z = ComplexSIMD[float_dtype, 1](0, 0)
    iters = Scalar[int_dtype](0)

    var in_set_mask = Scalar[DType.bool](True)
    for _ in range(MAX_ITERATIONS):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    tensor[row, col] = iters
```

This begins by calculating the complex number which represents a given location
in the output grid (C). Then, starting from `Z=0`, the calculation `Z=Z^2 + C`
is iteratively calculated until Z exceeds 4, the threshold we're using for when
Z will escape the set. This occurs up until a maximum number of iterations,
and the number of iterations to escape (or not, if the maximum is hit) is then
returned for each location in the grid.

The area to examine in complex space, the resolution of the grid, and the
maximum number of iterations are all provided as constants:

```mojo
alias MIN_X: Scalar[float_dtype] = -2.0
alias MAX_X: Scalar[float_dtype] = 0.7
alias MIN_Y: Scalar[float_dtype] = -1.12
alias MAX_Y: Scalar[float_dtype] = 1.12
alias SCALE_X = (MAX_X - MIN_X) / GRID_WIDTH
alias SCALE_Y = (MAX_Y - MIN_Y) / GRID_HEIGHT
alias MAX_ITERATIONS = 100
```

Try it with this command:

```sh
pixi run mojo mandelbrot.mojo
```

The result should be an ASCII art depiction of the region covered by the
calculation:

```output
...................................,,,,c@8cc,,,.............
...............................,,,,,,cc8M @Mjc,,,,..........
............................,,,,,,,ccccM@aQaM8c,,,,,........
..........................,,,,,,,ccc88g.o. Owg8ccc,,,,......
.......................,,,,,,,,c8888M@j,    ,wMM8cccc,,.....
.....................,,,,,,cccMQOPjjPrgg,   OrwrwMMMjjc,....
..................,,,,cccccc88MaP  @            ,pGa.g8c,...
...............,,cccccccc888MjQp.                   o@8cc,..
..........,,,,c8jjMMMMMMMMM@@w.                      aj8c,,.
.....,,,,,,ccc88@QEJwr.wPjjjwG                        w8c,,.
..,,,,,,,cccccMMjwQ       EpQ                         .8c,,,
.,,,,,,cc888MrajwJ                                   MMcc,,,
.cc88jMMM@@jaG.                                     oM8cc,,,
.cc88jMMM@@jaG.                                     oM8cc,,,
.,,,,,,cc888MrajwJ                                   MMcc,,,
..,,,,,,,cccccMMjwQ       EpQ                         .8c,,,
.....,,,,,,ccc88@QEJwr.wPjjjwG                        w8c,,.
..........,,,,c8jjMMMMMMMMM@@w.                      aj8c,,.
...............,,cccccccc888MjQp.                   o@8cc,..
..................,,,,cccccc88MaP  @            ,pGa.g8c,...
.....................,,,,,,cccMQOEjjPrgg,   OrwrwMMMjjc,....
.......................,,,,,,,,c8888M@j,    ,wMM8cccc,,.....
..........................,,,,,,,ccc88g.o. Owg8ccc,,,,......
............................,,,,,,,ccccM@aQaM8c,,,,,........
...............................,,,,,,cc8M @Mjc,,,,..........
```

Try changing the various parameters above to produce different resolution
grids, or look into different areas in the complex number space.

## Next Steps

- See our [tutorial to get started with GPU
programming](https://mojolang.org/docs/manual/gpu/intro-tutorial).

- Learn GPU programming by solving increasingly challenging [GPU
puzzles](https://builds.modular.com/puzzles/introduction.html).
