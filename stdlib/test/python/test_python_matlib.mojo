# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo -I %py_interop_bin_dir %s

from CPython import CPython, PyObjectPtr
from python import Python


def main():
    try:
        var python = Python()
        var np = Python.import_module("numpy")
        var plt = Python.import_module("matplotlib.pyplot")
        if not plt:
            print("matplotlib not found")

        var time = np.arange(0, 10, Float32(0.01))
        var operand = time * Float32(-0.1)
        var amplitude = np.exp(operand)
        var position = amplitude * np.sin(time * 3)

        plt.plot(time, position)
        plt.plot(time, amplitude)
        plt.plot(time, -amplitude)

        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title("Oscillations")

        plt.show()
    except:
        print("Python failed")
