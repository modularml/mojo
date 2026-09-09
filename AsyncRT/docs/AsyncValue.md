# The `M::AsyncRT::AsyncValue` family of types

This document explains some of the concepts behind `AsyncValue` and related
types like `AsyncValueRef<T>`.

## `AsyncValue`

[`AsyncValue`](../include/AsyncRT/Runtime/AsyncValue.h) is conceptually similar
to [std::future](https://en.cppreference.com/w/cpp/thread/future), except that
`AsyncValue` does not let callers wait/block until the value becomes available.
Instead, the caller enqueues a closure that uses the value with
`AsyncValue::andThenSync`. `AsyncValue::emplace` will run any enqueued closures
when the value becomes available. This approach is similar to
[continuation passing](https://en.wikipedia.org/wiki/Continuation-passing_style).

Another major difference is that `AsyncValue` has built in support for error
handling: in addition to being completed by a future value, they may also be
completed by an error value (which tracks location information as well). All
clients are expected to cope with (and propagate) errors in a correct way.

`AsyncValue`s are heap allocated and reference counted. You should use them with
the [`RCRef`](/Support/include/Support/RCRef.h) and
[`AsyncValueRef<T>`](../include/AsyncRT/Runtime/AsyncValueRef.h)
classes whenever possible to maintain their lifetime.

### Types and type erasure

An `AsyncValue` will eventually resolve to hold a value of some C++ type, but
this is dynamic and can happen after construction. The `AsyncValue` type itself
is therefore type-erased: users can manipulate an `AnyAsyncValueRef` without
knowing what type it will ultimately contain. For example, you can enqueue a
closure with `AsyncValue::andThenSync()` without knowing the actual type that
will ultimately be contained in the `AsyncValue`. Type information is only
needed when *accessing* the contained data, for example with
`AsyncValue::get<T>()` or `AsyncValue::emplace<T>()`.

`AnyAsyncValueRef` is used when working with a type-erased
`AsyncValue` and `AsyncValueRef<T>` is used when you know the element type `T`
that is stored in the `AsyncValue`. It is preferable to use strong types if
you know them, but dynamic type-generic code sometimes doesn't.
`AsyncValueRef<T>` implicitly converts to `AnyAsyncValueRef`.

`AsyncValue` can hold any C++ type, including move-only and even non-movable
types, but all types need to be registered before use with
`AsyncValue::registerType<T>()`. This registration logic is allows dense
storage of the payloads and data, and allows limited type reflection with the
`->isType<T>()` predicate.

### Access to the `M::AsyncRT::CPUDevice` for an `AsyncValue`

The AsyncRT runtime is designed to support multiple instances of a runtime in a
process at the same time, so some things (for example allocating a new
`AsyncValue`) require an `M::AsyncRT::CPUDevice&` to be handy and around. This
can be awkward, because (like an `MLIRContext`) it is almost global state, and
it is a pain to pass it around everywhere.

Fortunately, `AsyncValue` instances is that they always know what
`M::AsyncRT::CPUDevice` they came from. You can access this through the
`asyncVal->getRuntime()` method which returns a
[`CompactCPUDevicePtr`](../include/AsyncRT/Runtime/CompactCPUDevicePtr.h).
A `CompactCPUDevicePtr` is a specialized class that can be used interchangeably
with `CPUDevice&`.

The consequence of this is that having an `AsyncValue` at hand gives you access
to the `CPUDevice&` that you need.

### Chaining work together with `andThenSync`

One of the most common things to do when building a series of asynchronous
computations is to enqueue work that occurs when a value becomes available.
`AsyncValue` makes this very easy through the `andThenSync` method:

```c++
void printWhenReady(AnyAsyncValueRef input) {
  input->andThenSync([]() {
    // This prints whenever `input` becomes ready.
    printf("input is ready!");
  });
}
```

If the `AsyncValue` is already ready when the `andThenSync` is executed, then
the lambda is immediately executed. Otherwise it is enqueued and run when the
value becomes available.

The nice thing about this pattern is that it provides the direct ability to
capture arbitrary state in the lambda's capture list, and that capture list
is kept alive for the duration of the lambdas execution. This means that any
other `RCRef` you capture will be alive for the duration as well:

```c++
/// When the specified int32_t becomes available, add it to the refcounted
/// table.
void addToTableWhenReady(AsyncValueRef<int32_t> input,
                         RCRef<TableOfValues> tablePtr) {
  // Watch out for order of evaluation, std::move will corrupt our `input`
  // argument.
  AsyncValue *inputPtr = input.getPointer();
  inputPtr->andThenSync([input = std::move(input),
                     tablePtr = std::move(tablePtr)]() {
    tablePtr->addValue(input.get());
  });
}
```

This is extremely handy for capturing and working with values, but you'll note
that there is a footgun here due to C++'s lack of order of evaluation rules. We
can't just use `input->andThenSync([input = std::move(input), ...` because the
compiler might evaluate the `std::move` before the load of input for the base
expression.

Another downside of this style of `andThenSync` is that it is capturing a
pointer to the value being waited on. This can increase the size of the lambda
and increase the chances of an out-of-line representation for the function. To
address both of these problems, you can use a form of `andThenSync` that gets
passed in a reference to the value when it is available. The same thing as
above can be expressed as:

```c++
/// When the specified int32_t becomes available, add it to the refcounted
/// table.
void addToTableWhenReady(AsyncValueRef<int32_t> input,
                         RCRef<TableOfValues> tablePtr) {
  // Note that use of `input.` vs `input->`:
  input.andThenSync([tablePtr = std::move(tablePtr)]
                (const AsyncValueRef<int32_t> &input) {
    tablePtr->addValue(input.get());
  });
}
```

Now we're not moving away from the `input` argument, we're introducing a shadow
of it within the lambda. This reduces the size of the capture list and removes
a footgun. You may take the argument in this way as `const
AsyncValueRef<int32_t> &` or `const AnyAsyncValueRef &` depending on whether
you have an `AsyncValueRef` or just an untyped `RCRef`. Because these are
passed in as a const reference, you will need to `.copy()` them if you want
to extend the lifetime of the reference.

### The states of `AsyncValue`

`AsyncValue` may be in four possible states: "unconstructed", "unconstructed
(with inline waiter)",
"value available" and "error". The final two states are considered to be
"ready" states - they happen when the future is resolved (either to a value or
an error) - all waiters are notified transitioning to a ready state, and you
cannot transition an `AsyncValue` back out of a ready state.

**"Unconstructed":** An `AsyncValue` in unconstructed state is obtained from the
`AsyncValue::allocate<T>` or `AsyncValueRef<T>::allocate` static method. In
this state, any `andThenSync` requests are queued up until the value transitions
into a ready state.

**"Unconstructed (with inline waiter)":** An `AsyncValue` in unconstructed state
works exactly like an `kUnconstructed` one, but has its first waiter held in the
payload field. Clients of `AsyncValue` will never have to worry about this.

**"Value Available":** This is the state that most `AsyncValue`s achieve where
they hold a completed C++ value and where all `andThenSync` waiters are
notified. You can directly create an `AsyncValue` in this state with
`AsyncValue::createReady<T>` or `AsyncValueRef<T>::createReady`, but most cases
will create one in unconstructed and transition to this state with the
`emplace(...)` method.

**"Error":** This state indicates that the computation creating the value had
an error. You may create an `AsyncValue` directly in this state with the
`createError` method, but a more typical usage is to determine that an
unconstructed `AsyncValue` had a problem, and transition it to this state with
the `setToError` method.

### Indirect Async Values

Beyond these four core states, you may run into a situation where you need to
create an `AsyncValue` before knowing what C++ type it will contain. In this
case, you can create a special "indirect AsyncValue" with the
`AsyncValue::createIndirect`, and resolve it with `resolveIndirect` method. As
the name implies, this adds a level of indirection that allows you to create an
AsyncValue, and then fulfill it with another AsyncValue of concrete type later.

For example, you might have some type generic code that resolves the type
depending on the input types:

```C++
// This works with both integer and string values forming "x+x" or "concat(x,x)"
// depending on what the argument resolves to.
AnyAsyncValueRef genericAsyncDouble(AnyAsyncValueRef input) {
  // Must create this value before knowing what type `input` is.
  AnyAsyncValueRef result = AsyncValue::createIndirect(input->getRuntime());

  input.andThenSync([result = result.copy()](const AnyAsyncValueRef &input) {
    AnyAsyncValueRef newVal;
    if (input.isType<int32_t>())
      newVal = AsyncValue::createReady<T>(input.get<int32_t>()*2);
    else {
      assert(input.isType<std::string>() && "unexpected type");
      const std::string &str = input.get<std::string>();
      newVal = AsyncValue::createReady<T>(str+str);
    }
    result->resolveIndirect(std::move(newVal));
  });

  return result;
}
```
