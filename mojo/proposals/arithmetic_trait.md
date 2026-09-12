# Abstracting Over Arithmetic Types

## Motivation

As Mojo becomes more widely used across scientific computing domains, giving
library writers the ability to effectively abstract over arithmetic types will
be crucial to its success in those areas. As Mojo intends to give users
low-level control over exotic hardware, organizing such an abstraction quickly
becomes non-trivial as we must consider operations where the output type is
different than the inputs. Including dependent type operations where the
output type becomes a function of the input types.

In this proposal I have taken some liberties assuming language features that
do not exist, nor are they planned to be implemented in the near future. The
exact syntax isn't particularly important, but I have added them simply as a way
of representing the sort of solution that I would find to be elegant.

Currently if library writers want to define functions that operate on all the
common numeric types, they must provide individual overloads for
each of `SIMD`, `Int`, `UInt`, and any other types they may want to support.

```mojo
fn sum(*args: Int) -> Int:
    ...

fn sum(*args: UInt) -> UInt:
    ...

fn sum(*args: SIMD) -> __type_of(args):
    ...
```

## The Arithmetic trait

The solution is to build a heirachy of traits that are each responsible for
defining one particular operation. The most trivial form would look like the
following:

```mojo
trait Add:

    fn __add__(self, other: Self) -> Self:
        ...

trait Sub:

    fn __sub__(self, other: Self) -> Self:
        ...

trait Arithmetic(Add, Sub):
    ...
```

Since in-place operations like `__iadd__` are impossible to implement for
dependent types due to the result always being different than `Self`.
They must be kept separate from the non-mutating operations that can return a
new type.

```mojo

trait IAdd:

    fn __iadd__(self, other: Self):
        ...

trait IArithmetic(IAdd):
    ...
```

We can handle certain special cases using associated aliases. Such as when an
operation produces a concrete result that doesn't match the input types,
a good example being `Int.__truediv__` which returns a `Float64`.
It also might be nice to consider that associated aliases could have default
values to save a lot of boilerplate in the common case of simply returning
`Self`.

```mojo

trait Add:
    alias AddOutput: AnyType = Self

    fn __add__(self, other: Self) -> AddOutput:
        ...

trait TrueDiv:
    alias TrueDivOutput: AnyType = Self

    fn __truediv__(self, other: Self) -> TrueDivOutput:
        ...

struct Int(TrueDiv, Add, ...):

    alias TrueDivOutput = Float64

    fn __truediv__(self, other: Self) -> TrueDivOutput:
        ...

    fn __add__(self, other: Self) -> AddOutput:
        ...
```

However, with such a simple approach we immediately encounter two significant
issues.

### We cannot express that a type can be added with something other than itself

This could partially be remedied by adding another associated alias to the
trait definition, but we still can't express types that can be added with more
than one type. So parametric traits would likely be the proper solution.

```mojo
trait Add[OtherT: AnyType, Result: AnyType]:

    fn __add__(self, other: OtherT) -> Result:
        ...

fn addableWithInt[Result: AnyType, T: Some[Add[Int, Result]]](a: T, i: Int) -> Result:
    return a + i
```

### We cannot make dependent types conform to this trait

To support dependent types we must devise a way to represent that the output
of an operation is not a static type, but a function of the two input types.
Similar to the case of parametric traits, Mojo does not have the ability to
express this concept to the extent we require.

I can conceive of a way to accomplish this in C++ using concepts and struct
template specialization.

```c++
template<int V>
struct Foo;

template <typename L, typename R>
struct OutputType
{};

template <typename L, typename R=L>
concept Addable = requires(L a, R b)
{
    {a + b} -> std::same_as<typename OutputType<L, R>::T>;
};

template<int L, int R>
struct OutputType<Foo<L>, Foo<R>>
{
    using T = Foo<L +R>;
};

template<int V>
struct Foo
{
    template<int O>
    typename OutputType<Foo<V>, Foo<O>>::T operator+(Foo<O> other)
    {
        return{};
    }
};

template<Addable L, Addable R>
typename OutputType<L, R>::T doAdd(L l, R r)
{
    return l + r;
}
```

Now Mojo does not have parameter specialization, but I believe we could
accomplish a similar effect by allowing parameterized alias overloads.

```mojo

# default: All matching types
alias MulResultOf[l: AnyType, r: __type_of(l)] = __type_of(l)

alias MulResultOf[l: IntLiteral, r: IntLiteral] = __type_of(l, r)
alias MulResultOf[l: Int, r: Int] = Int


trait Mul[OtherT: AnyType = Self]:

    fn __mul__(self, other: OtherT) -> ResultOf[Self, OtherT]:
        ...

# Needs to be able to accept unbound type parameters
struct IntLiteral[...](Mul[IntLiteral]):

    fn __mul__(self, other: IntLiteral) -> ResultOf[Self, __type_of(other)]:
        ...

struct Int(Mul):

    fn __mul__(self, other: Int) -> ResultOf[Int, Int]:
        ...
```

Using parametric aliases we can also for instance express that a type is addable
with multiple different types.

```mojo

alias MulResultOf[l: String, r: Int] = String
alias MulResultOf[l: String, r: UInt] = String


struct String(Mul[Int], Mul[UInt]):

    fn __mul__(self, other: Int) -> MulResultOf[String, Int]:
        ...
    
    fn __mul__(self, other: UInt) -> MulResultOf[String, UInt]:
        ...
```

### Putting it all together

A complete implementation might look something like this. Types that support
all operations for a particular input type can use the `Arithmetic/IArithmetic`
traits, and special cases for particular operations can be added as well.

```mojo

trait Arithmetic(
    Add[Other], 
    Sub[Other],
    Div[Other],
    Mult[Other]
):
    ...

# Reverse operations
trait RArithmetic(
    RAdd[Other], 
    RSub[Other],
    RDiv[Other],
    RMult[Other]
):
    ...

trait IArithmetic(
    IAdd,
    ISub,
    IDiv,
    IMult
):
    ...

struct Int(
    Arithemtic, # defaults to Arithmetic[Self],
    IArithemetic,
    Arithmetic[SIMD],
    RArithmetic[SIMD]
    IArithmetic[SIMD],
    ...,
):
    ...

struct String(
    Add,
    Mult[SIMD],
    Mult[Int],
    RMult[SIMD],
    ...
):
    ...
```

I believe such an approach offers sufficient flexibility, while minimizing
boilerplate and complexity. Though it invents a few compiler capabilities
that we do not currently have. We could implement a simpler subset of
the complete functionality using assocated aliases, depending on how
tolerant we wish to be about performing migrations in the future once
more powerful language features are available to us.
