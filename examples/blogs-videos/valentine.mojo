import math
from memory import memset_zero, memcpy
from sys.info import simdwidthof
from algorithm import vectorize
from python import Python

struct MojoArray[dtype: DType = DType.float64](Stringable):
    var _ptr: DTypePointer[dtype]
    var numel: Int
    alias simd_width: Int = simdwidthof[dtype]()

    # Initializers
    fn __init__(inout self, numel: Int):
        self._ptr = DTypePointer[dtype].alloc(numel)
        self.numel = numel
        memset_zero[dtype](self._ptr, numel)
    
    fn __init__(inout self, numel: Int, _ptr: DTypePointer[dtype]):
        self._ptr = _ptr
        self.numel = numel

    fn __init__(inout self, *data: Scalar[dtype]):
        self.numel = len(data)
        self._ptr = DTypePointer[dtype].alloc(len(data))
        for i in range(len(data)):
            self._ptr[i] = data[i]

    fn __copyinit__(inout self, other: Self):
        self._ptr = other._ptr
        self.numel = other.numel
    
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        return self._ptr.simd_load[1](idx) 
    
    fn __neg__(self)->Self:
        return self._elemwise_scalar_math[math.mul](-1.0)

    fn __mul__(self, other: Self)->Self:
        return self._elemwise_array_math[math.mul](other)
    
    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.mul](s)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __add__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.add](s)
    
    fn __add__(self, other: Self)->Self:
        return self._elemwise_array_math[math.add](other)
       
    fn __radd__(self, s: Scalar[dtype])->Self:
        return self+s

    fn __sub__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.sub](s)

    fn __sub__(self, other: Self)->Self:
        return self._elemwise_array_math[math.sub](other)

    fn __rsub__(self, s: Scalar[dtype])->Self:
        return -self+s

    @staticmethod
    fn from_numpy(np_array: PythonObject) raises->Self:
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var numel = int(np_array.shape[0])
        var _ptr = DTypePointer[dtype].alloc(numel)
        memcpy(_ptr, npArrayPtr, numel)
        return Self(numel,_ptr)

    fn to_numpy(self) raises->PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.zeros(self.numel)
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(npArrayPtr, self._ptr, self.numel)
        return np_arr

    fn __str__(self)->String:
        var s:String = ""
        s += "["
        for i in range(self.numel):
            if i>0:
                s+=" "
            s+=self._ptr[i]
        s+="]"
        return s

    fn sqrt(self)->Self:
        return self._elemwise_transform[math.sqrt]()

    fn cos(self)->Self:
        return self._elemwise_transform[math.cos]()

    fn sin(self)->Self:
        return self._elemwise_transform[math.sin]()

    fn abs(self)->Self:
        return self._elemwise_transform[math.abs]()

    fn __pow__(self, p: Scalar[dtype])->Self:
        return self._elemwise_pow(p)

    fn _elemwise_pow(self, p: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, math.pow[dtype,dtype,simd_width](self._ptr.simd_load[simd_width](idx), SIMD[dtype,simd_width].splat(p)))
        vectorize[simd_width, tensor_scalar_vectorize](self.numel)
        return new_array

    fn _elemwise_transform[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _elemwise_array_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), other._ptr.simd_load[simd_width](idx)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

def main():
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    np_arr = np.arange(-2,2,0.01)
    x = MojoArray.from_numpy(np_arr)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])   

    a = MojoArray.from_numpy(np.linspace(0,20,100))
    for i in range(a.numel):
        y = (x**2)**(1/3.) - 0.9*((3.3-(x*x)).sqrt())*(a[i]*3.14*x).sin()
        ax.cla()
        title = ax.set_title("Mojo ❤️ Python")
        title.set_fontsize(20)
        ax.set_axis_off()
        ax.plot(x.to_numpy(),y.to_numpy(),'r')
        plt.pause(0.1)
        plt.draw()