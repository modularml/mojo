# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s

from memory.unsafe import DTypePointer
from python import Python
from python.object import PythonObject

alias float_type = DType.float64
alias int_type = DType.int64


struct Matrix:
    var data: DTypePointer[DType.int32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.int32].alloc(rows * cols)
        self.rows = rows
        self.cols = cols

    fn __copyinit__(inout self, existing: Self):
        self.data = existing.data
        self.rows = existing.rows
        self.cols = existing.cols

    fn _del_old(self):
        self.data.free()

    fn __getitem__(self, row: Int, col: Int) -> Int:
        return self.data.load(row * self.cols + col).value

    fn __setitem__(inout self, row: Int, col: Int, val: Int):
        self.data[row * self.cols + col] = val

    def to_numpy(self) -> PythonObject:
        var np = Python.import_module("numpy")
        var numpy_array = np.zeros((yn, xn), np.uint32)
        for x in range(xn):
            for y in range(yn):
                numpy_array.itemset((y, x), self[x, y])
        return numpy_array


@register_passable("trivial")
struct Complex:
    var real: Float32
    var imag: Float32

    fn __init__(real: Float32, imag: Float32) -> Self:
        return Self {real: real, imag: imag}

    fn __add__(lhs, rhs: Self) -> Self:
        return Self(lhs.real + rhs.real, lhs.imag + rhs.imag)

    fn __mul__(lhs, rhs: Self) -> Self:
        return Self(
            lhs.real * rhs.real - lhs.imag * rhs.imag,
            lhs.real * rhs.imag + lhs.imag * rhs.real,
        )

    fn norm(self) -> Float32:
        return self.real * self.real + self.imag * self.imag


alias xmin = Float32(-2.25)
alias xmax = Float32(0.75)
alias xn = 1500
alias ymin = Float32(-1.25)
alias ymax = Float32(1.25)
alias yn = 1250


# Compute the number of steps to escape.
def mandlebrot_kernel(c: Complex) -> Int:
    max_iter = 200
    z = c
    for i in range(max_iter):
        z = z * z + c
        if z.squared_norm() > 4:
            return i
    return max_iter


def compute_mandlebrot() -> Matrix:
    # create a matrix. Each element of the matrix corresponds to a pixel
    result = Matrix(xn, yn)

    cnt = 0
    x = xmin
    dx = (xmax - xmin) / xn
    dy = (ymax - ymin) / yn
    for i in range(xn):
        y = ymin
        for j in range(yn):
            result[i, j] = mandlebrot_kernel(Complex(x, y))
            y += dy
        x += dx
    return result


def main():
    var python = Python()
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")

    result = compute_mandlebrot()
    dpi = 72
    width = 10
    height = 10 * yn // xn

    fig = plt.figure(1, [width, height], dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)

    light = colors.LightSource(315, 10, 0, 1, 1, 0)
    image = light.shade(
        result.to_numpy(), plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5
    )
    plt.imshow(image)
    plt.axis("off")
    plt.show()
