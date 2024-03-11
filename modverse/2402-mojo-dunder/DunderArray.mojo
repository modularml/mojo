import math, random
from algorithm import vectorize
from algorithm.functional import elementwise
from sys.intrinsics import _mlirtype_is_eq
from python import Python

struct DunderArray[dtype: DType = DType.float64](Stringable, Intable, CollectionElement, Sized):
    var _ptr: DTypePointer[dtype]
    var numel: Int

    # Initializers
    fn __init__(inout self, numel: Int):
        self._ptr = DTypePointer[dtype].alloc(numel)
        self.numel = numel
        memset_zero[dtype](self._ptr, numel)

    fn __init__(inout self,np_array: PythonObject) raises:
        var npArrayPtr = DTypePointer[dtype](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, dtype.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        self.numel = int(np_array.shape[0])
        self._ptr = DTypePointer[dtype].alloc(self.numel)
        memcpy(self._ptr, npArrayPtr, self.numel)

    
    fn __init__(inout self, numel: Int, _ptr: DTypePointer[dtype]):
        self._ptr = _ptr
        self.numel = numel

    fn __init__(inout self, *data: Scalar[dtype]):
        self.numel = len(data)
        self._ptr = DTypePointer[dtype].alloc(len(data))
        for i in range(len(data)):
            self._ptr[i] = data[i]

    @staticmethod
    fn rand(numel: Int)->Self:
        let _ptr = DTypePointer[dtype].alloc(numel)
        random.rand(_ptr, numel)
        return Self(numel, _ptr)

    fn __copyinit__(inout self, other: Self):
        self._ptr = other._ptr
        self.numel = other.numel
    
    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self.numel = existing.numel
        existing.numel = 0
        existing._ptr = DTypePointer[dtype]()

    fn __del__(owned self):
        self._ptr.free()

    # Getter and setters
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        return self._ptr.simd_load[1](idx) 

    fn __setitem__(inout self, elem: Int, val: Scalar[dtype]):
        return self._ptr.simd_store[1](elem, val)

    # Unary arithmetic operators
    fn __neg__(self)->Self:
        return self*(-1.0)
    
    fn __pos__(self)->Self:
        return self*(1.0)

    fn __invert__(self) -> Self:
        if _mlirtype_is_eq[Scalar[dtype], Bool]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], ~self._ptr.simd_load[simd_width](idx[0]))
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only invert Bool arrays')
            return self

    # Normal comparison operators
    fn __lt__(self, other: Self) -> Bool:
        return (self**2)._reduce_sum()[0] < (other**2)._reduce_sum()[0]

    fn __le__(self, other: Self) -> Bool:
        return (self**2)._reduce_sum()[0] <= (other**2)._reduce_sum()[0]

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr
    
    fn __ne__(self, other: Self) -> Bool:
        return self._ptr != other._ptr

    fn __gt__(self, other: Self) -> Bool:
        return (self**2)._reduce_sum()[0] > (other**2)._reduce_sum()[0]

    fn __ge__(self, other: Self) -> Bool:
        return (self**2)._reduce_sum()[0] >= (other**2)._reduce_sum()[0]
    
    # Normal, reflected and inplace arithmetic operators
    fn __add__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.add](s)
    
    fn __add__(self, other: Self)->Self:
        return self._elemwise_array_math[math.add](other)
       
    fn __radd__(self, s: Scalar[dtype])->Self:
        return self+s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self+s

    fn __sub__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.sub](s)
    
    fn __sub__(self, other: Self)->Self:
        return self._elemwise_array_math[math.sub](other)

    fn __rsub__(self, s: Scalar[dtype])->Self:
        return -(self-s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self-s

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.mul](s)

    fn __mul__(self, other: Self)->Self:
        return self._elemwise_array_math[math.mul](other)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self*s

    fn __matmul__(self, other: Self)->Self:
        return self._elemwise_array_math[math.mul](other)._reduce_sum()

    fn __imatmul__(inout self, other: Self):
        self = self.__matmul__(other)

    fn __truediv__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.div](s)

    fn __truediv__(self, other: Self)->Self:
        return self._elemwise_array_math[math.div](other)

    fn __rtruediv__(self, s: Scalar[dtype])->Self:
        return self._r_elemwise_scalar_math[math.div](s)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self/s

    fn __floordiv__(self, s: Scalar[dtype])->Self:
        return (self/s)._elemwise_transform[math.floor]()

    fn __floordiv__(self, other: Self)->Self:
        return (self/other)._elemwise_transform[math.floor]()

    fn __rfloordiv__(self, s: Scalar[dtype])->Self:
        return (s/self)._elemwise_transform[math.floor]()

    fn __ifloordiv__(inout self, s: Scalar[dtype]):
        self = self.__rfloordiv__(s)

    fn __mod__(self, s: Scalar[dtype])->Self:
        return self._elemwise_scalar_math[math.mod](s)

    fn __mod__(self, other: Self)->Self:
        return self._elemwise_array_math[math.mod](other)

    fn __rmod__(self, s: Scalar[dtype])->Self:
        return self._r_elemwise_scalar_math[math.mod](s)

    fn __imod__(inout self, s: Scalar[dtype]):
        self = self.__mod__(s)

    fn __pow__(self, p: Int)->Self:
        return self._elemwise_pow(p)

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn __lshift__(self, p: Int) -> Self:
        if _mlirtype_is_eq[Scalar[dtype], Int32]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], self._ptr.simd_load[simd_width](idx[0])<<p)
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only shift int arrays')
            return self

    fn __rshift__(self, p: Int) -> Self:
        if _mlirtype_is_eq[Scalar[dtype], Int32]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], self._ptr.simd_load[simd_width](idx[0])>>p)
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only shift int arrays')
            return self

    fn __ilshift__(inout self, p: Int):
        self = self.__lshift__(p)

    fn __irshift__(inout self, p: Int):
        self = self.__rshift__(p)

    fn __and__(self, other: Self) -> Self:
        if ~_mlirtype_is_eq[Scalar[dtype], Float64]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], 
                    self._ptr.simd_load[simd_width](idx[0])&other._ptr.simd_load[simd_width](idx[0]))
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only AND int or bool arrays')
            return self

    fn __iand__(inout self, other: Self):
        self = self.__and__(other)

    fn __or__(self, other: Self) -> Self:
        if ~_mlirtype_is_eq[Scalar[dtype], Float64]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], 
                    self._ptr.simd_load[simd_width](idx[0])|other._ptr.simd_load[simd_width](idx[0]))
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only AND int or bool arrays')
            return self

    fn __ior__(inout self, other: Self):
        self = self.__or__(other)

    fn __xor__(self, other: Self) -> Self:
        if ~_mlirtype_is_eq[Scalar[dtype], Float64]():
            let new_array = Self(self.numel)
            @parameter
            fn wrapper[simd_width:Int,rank:Int=1](idx: StaticIntTuple[rank]):
                new_array._ptr.simd_store[simd_width](idx[0], 
                    self._ptr.simd_load[simd_width](idx[0])^other._ptr.simd_load[simd_width](idx[0]))
            elementwise[1, simdwidthof[dtype](), wrapper](self.numel)
            return new_array
        else:
            print('Error: You can only AND int or bool arrays')
            return self

    fn __ixor__(inout self, other: Self):
        self = self.__xor__(other)

    fn __len__(self) -> Int:
        return self.numel

    fn __int__(self) -> Int:
        return self.numel

    fn __bool__(self) -> Bool:
        return self.dtype == DType.bool

    fn __str__(self) -> String:
        var printStr:String = "["
        var prec:Int=4
        for i in range(self.numel):
            var val = self[i]
            @parameter 
            if _mlirtype_is_eq[Scalar[dtype], Float64]():
                var s: String = ""
                let int_str: String
                int_str = String(math.trunc(val).cast[DType.int32]())
                if val < 0.0:
                    val = -val
                let float_str: String
                if math.mod(val,1)==0:
                    float_str = "0"
                else:
                    float_str = String(math.mod(val,1))[2:prec+2]
                s = int_str+"."+float_str               
                if i==0:
                    printStr+=s
                else:
                    printStr+="  "+s
            else:
                if i==0:
                    printStr+=str(val)
                else:
                    printStr+="  "+str(val)

        printStr+="]\n"
        printStr+="Length:"+str(self.numel)+","+" DType:"+str(dtype)
        return printStr
    
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

    fn _reduce_sum(self) -> Self:
        var reduced = Self(1)
        alias simd_width: Int = simdwidthof[dtype]()
        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced[0] += self._ptr.simd_load[simd_width](idx).reduce_add()
        vectorize[simd_width,vectorize_reduce](self.numel)
        return reduced

    fn _elemwise_transform[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        let new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        let new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), SIMD[dtype, simd_width](s)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _r_elemwise_scalar_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        let new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._ptr.simd_load[simd_width](idx)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _elemwise_array_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        let new_array = Self(self.numel)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, func[dtype, simd_width](self._ptr.simd_load[simd_width](idx), other._ptr.simd_load[simd_width](idx)))
        vectorize[simd_width, elemwise_vectorize](self.numel)
        return new_array

    fn _elemwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        let new_array = Self(self.numel)
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.simd_store[simd_width](idx, math.pow(self._ptr.simd_load[simd_width](idx), p))
        vectorize[simd_width, tensor_scalar_vectorize](self.numel)
        return new_array
    