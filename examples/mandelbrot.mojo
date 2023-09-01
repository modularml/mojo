from benchmark import Benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota
from python import Python
from runtime.llcl import num_cores, Runtime
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

alias float_type = DType.float64
alias simd_width = simdwidthof[float_type]()

alias width = 960
alias height = 960
alias MAX_ITERS = 200

alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5


# Compute the number of steps to escape.
def mandelbrot_kernel(c: ComplexFloat64) -> Int:
    z = c
    for i in range(MAX_ITERS):
        z = z * z + c
        if z.squared_norm() > 4:
            return i
    return MAX_ITERS


def compute_mandelbrot() -> Tensor[float_type]:
    # create a matrix. Each element of the matrix corresponds to a pixel
    m = Tensor[float_type](width, height)

    dx = (max_x - min_x) / width
    dy = (max_y - min_y) / height

    y = min_y
    for j in range(height):
        x = min_x
        for i in range(width):
            m[Index(i, j)] = mandelbrot_kernel(ComplexFloat64(x, y))
            x += dx
        y += dy
    return m


def show_plot(tensor: Tensor[float_type]):
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")

    numpy_array = np.zeros((height, width), np.float32)

    for col in range(width):
        for row in range(height):
            numpy_array.itemset((row, col), tensor[col, row])

    fig = plt.figure(1, [10, 10 * height // width], 64)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
    light = colors.LightSource(315, 10, 0, 1, 1, 0)

    image = light.shade(
        numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5
    )

    plt.imshow(image)
    plt.axis("off")
    plt.savefig("out.png")
    plt.show()


fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[float_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[float_type, simd_width](0)

    var in_set_mask: SIMD[DType.bool, simd_width] = True
    for i in range(MAX_ITERS):
        if not in_set_mask.reduce_or():
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters


fn vectorized():
    let m = Tensor[float_type](width, height)

    @parameter
    fn worker(col: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](row: Int):
            """Each time we oeprate on a `simd_width` vector of pixels."""
            let cy = min_y + (row + iota[float_type, simd_width]()) * scale_y
            let cx = min_x + col * scale_x
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            m.simd_store[simd_width](
                Index(col, row), mandelbrot_kernel_SIMD[simd_width](c)
            )

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    @parameter
    fn bench[simd_width: Int]():
        for col in range(width):
            worker(col)

    let vectorized = Benchmark().run[bench[simd_width]]() / 1e6
    print("Vectorized", ":", vectorized, "ms")

    try:
        _ = show_plot(m)
    except e:
        print("failed to show plot:", e.value)


fn parallelized():
    let m = Tensor[float_type](width, height)

    @parameter
    fn worker(col: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](row: Int):
            """Each time we oeprate on a `simd_width` vector of pixels."""
            let cy = min_y + (row + iota[float_type, simd_width]()) * scale_y
            let cx = min_x + col * scale_x
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            m.simd_store[simd_width](
                Index(col, row), mandelbrot_kernel_SIMD[simd_width](c)
            )

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    with Runtime() as rt:

        @parameter
        fn bench_parallel[simd_width: Int]():
            parallelize[worker](rt, width, 5 * num_cores())

        alias simd_width = simdwidthof[DType.float64]()
        let parallelized = Benchmark().run[bench_parallel[simd_width]]() / 1e6
        print("Parallelized:", parallelized, "ms")

    try:
        _ = show_plot(m)
    except e:
        print("failed to show plot:", e.value)


fn compare():
    let m = Tensor[float_type](width, height)

    @parameter
    fn worker(col: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](row: Int):
            """Each time we oeprate on a `simd_width` vector of pixels."""
            let cy = min_y + (row + iota[float_type, simd_width]()) * scale_y
            let cx = min_x + col * scale_x
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            m.simd_store[simd_width](
                Index(col, row), mandelbrot_kernel_SIMD[simd_width](c)
            )

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    # Vectorized
    @parameter
    fn bench[simd_width: Int]():
        for col in range(width):
            worker(col)

    let vectorized = Benchmark().run[bench[simd_width]]() / 1e6
    print("Number of hardware cores:", num_cores())
    print("Vectorized:", vectorized, "ms")

    # Parallelized
    with Runtime() as rt:

        @parameter
        fn bench_parallel[simd_width: Int]():
            parallelize[worker](rt, width, 5 * num_cores())

        alias simd_width = simdwidthof[DType.float64]()
        let parallelized = Benchmark().run[bench_parallel[simd_width]]() / 1e6
        print("Parallelized:", parallelized, "ms")
        print("Parallel speedup:", vectorized / parallelized)


fn main() raises:
    compare()
