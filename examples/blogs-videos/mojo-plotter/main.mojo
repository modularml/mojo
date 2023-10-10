from python import Python


fn main() raises:
    let torch = Python.import_module("torch")
    let x = torch.linspace(0, 10, 100)
    let y = torch.sin(x)
    plot(x, y)

def plot(x: PythonObject, y: PythonObject) -> None:
    let plt = Python.import_module("matplotlib.pyplot")
    plt.plot(x.numpy(), y.numpy())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Plot of y = sin(x)")
    plt.grid(True)
    plt.show()
