# Cache

## Glossary of Terms

`CAS` - **C**ontent **A**ddressible **S**tore # spellchecker:disable-line

`delegate` - Higher-level cache backend to fault to. For example, the delegate
for the in-memory backend is often the filesystem backend.

## Introduction

The Cache module is Modular's CAS system. It supports a hierarchical backend
system based on a delegation model. It is fully asynchronous. It contains APIs
for caching IR and generic (serializable) transformations. It has some MLIR
components (Attributes, custom ops) used for supporting the various caching
APIs.

## Buffer

In order to provide an asynchronous cache, we need a ref-counted buffer. If
a user provides data to the cache to be stored, we have to keep that data alive
until it has been stored.

This ref-counted buffer must be writeable, and for convenience we'd like it to
implement the `llvm::MemoryBuffer` and `llvm::raw_pwrite_stream` APIs. The
`Buffer` class provides the `llvm::MemoryBuffer` APIs, and the
`WriteableBuffer` class provides the `llvm::raw_pwrite_stream` APIs. As
usual with the AsyncRT ref-counting infrastructure, `BufferRef` is a
`RCRef` of the `Buffer` class, same for the `WriteableBuffer` class.

`Buffer` interns any data passed to it on construction - it's important to
ensure that any references to it actually extend the lifetime of the underlying
memory, which means that the class itself should own the memory. If a string
is passed in with `get`, then `Buffer` will copy the data into its own internal
buffer. If a file is opened with `getFile`, on success `Buffer` will mmap the
file read-only from the file system. The file will be automatically unmapped
when the last reference is destroyed.

`WriteableBuffer` works the same way as `Buffer` - it owns any data passed to
it. For a buffer allocated with `get`, anything written to it is copied into
its internal buffer, which is dynamically resized as necessary. For a buffer
allocated with `getFile`, the file is mapped read/write and can be written into
by the user. Care should be taken with `getFile` - simply streaming data to a
mapped file will result in writing data to the *end* of the file - an invalid
operation. The user should instead use the `pwrite` API to write data to a
particular offset.

```c++
using namespace Cache;
...

BufferRef buffer = Buffer::get("hello");
llvm::outs() << buffer->getBuffer() << "\n"; // prints "hello"
...

BufferRef fileBuf = Buffer::getFile("foo.txt");
llvm::outs() << buffer->getBuffer() << "\n"; // prints the contents of foo.txt
...

WriteableBufferRef writeableBuf = WriteableBuffer::get();
*writeableBuf << "hello" << 123;
// Convert the writeable reference to a read-only reference.
BufferRef buf = std::move(writeableBuf);
llvm::outs() << buf->getBuffer() << "\n"; // prints "hello123"
...

ErrorOr<WriteableBufferRef> writeOr = WriteableBuffer::getFile("tmpFile", /*size=*/5, /*offset=*/0);
WriteableBufferRef write = std::move(*writeOr);
// pwrite because we want to write to a particular offset.
char hello[] = "hello";
write->pwrite(hello, 5, 0);
```

## BlobCache

The BlobCache is made up of 3 essential components, the backend, the KeyInfo
struct, and the BlobCache itself.

### BlobCacheBackend

The [BlobCacheBackend](../include/Cache/BlobCache.h) is the part of the system
responsible for providing an interface to the storage backends. Backends exist
as a linked-list, where the next node in the list is known as the delegate. The
base BlobCacheBackend class handles asynchrony so that the individual backends
can implement synchronous operations for the API functions. Backends should
generally be simple to construct and should generally not require data to be
copied. For an example, look at the [FileSystemBackend](../lib/BlobCache.cpp).
It simply reads from and writes to a particular directory - this means that
multiple `FileSystemBackend` instances can share the same cache directory.
This is a property we should strive to maintain.

`BlobCacheBackend` is a virtual base class, and implementations need to provide
the following:

```c++
/// Subclasses should use this to provide the implementation of actually
/// storing an item.
virtual ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) = 0;
/// Subclasses should use this to provide the implementation of checking if an
/// item exists.
virtual bool containsImpl(StringRef keyHash) const = 0;
/// Subclasses should use this to provide the implementation of getting an
/// item from storage.
virtual CacheFindResult findImpl(StringRef keyHash) const = 0;
```

### KeyInfo

The KeyInfo struct is the template field of the BlobCache. The user of the
BlobCache must provide this structure to allow the BlobCache to hash the cache
keys. It should be of the form:

```c++
struct KeyInfo {
  using KeyTy = SomeT;
  static std::string hashKey(KeyTy key);
}
```

### BlobCache

The [BlobCache](../include/Cache/BlobCache.h) has an API like `llvm::DenseMap`
with `insert`, `contains`, and `find`. It is templated on KeyInfo, which
enables the user to use any C++ type as the key. It uses a linked-list of
backends to provide its storage hierarchy. The BlobCache API is fully async,
enabling the user to kick off a cache operation and chain other computation off
its result. This is important because in the future this may kick off a network
request, which we should not synchronously wait on.

The `BlobCache` assumes overwrite semantics on `insert`, so it is incumbent on
the user to provide a strong hash function - the region and transform caches
use SHA-256 as a hash function currently.

## Transform Caching

The principle behind transform caching is also quite simple - as long as the
input and output of a transform can be serialized and the transform is
deterministic, we can use the inputs as a cache key and lookup the outputs
directly. For a concrete example, take a pass pipeline given by a
`PassManager`.

```c++
mlir::PassManager pm(&ctx);
// Set up the pass manager.

// Parse the source string.
mlir::OwningOpRef<ModuleOp> module1 =
  mlir::parseSourceString<ModuleOp>(mlirString, ParserConfig{&ctx});
// Run the pass manager on the module.
auto xform = cachedTransform(*module1, transformCache, std::move(readyChain),
                             pm);
```

This example runs a set of provided passes, caches the result, and at the end,

The most complex part of caching a transform is understanding what the cache
key needs to include. In the case of the MLIR pass pipeline, the cache key
should contain the pass pipeline itself, the input IR, and a version indication
of the code that implements the passes that will be run. Users should take care
to understand all the dependencies of a given transform before caching it;
false cache hits are very difficult to debug!
