# Stdlib Proposal for mmap

A `mmap` module has mentioned by the community and welcomed by the core team, this proposal walks through a proposed basic API along with a brief proof of concept.

## What is Memory Mapping and how is it used?

`mmap` creates a new mapping in the virtual address space of the program. It allows for files or devices to be mapped directly into memory, enabling faster IO operations. It has a few primary advantages: it is efficient for random access on large files, it allows multiple processes to share memory, it can improve performance by reducing system calls and copy operations.

It uses demand paging, which means that file contents are not immediately read from disk. The OS manages this mapping and returns additional pages of data as needed.

Modules to access underlying OS technology surrounding memory mapped files, are available in most programming languages, and have become incredibly useful when deal with large datasets in the machine learning space.

## Proposal

The introduction of a `mmap` module, with two primary objects:

A `MmapMode`, an enum struct, identifying the four initial `mmap` configurations, for Read-Only, Write Only, Exec, and Copy-On-Write. 

A `Mmap` object, containing the functionality for construction/destruction and interaction with memory mapped files.

### High Level Usage

An initial implementation of `Mmap` in a read-only context can be used as follows. Note, the API has been meant to provide similarity with the existing `FileHandle` API.

```mojo
    var length: Int32 = 24;
    var fd: FileDescriptor = open("./test.txt", "r");

    # Initialize a mmap object for the particular file, with read and shared
    # permissions
    var mmap = Mmap(fd=fd, length=length, offset=0, mode=MmapMode.READ)

    # Print out the entire file as a String
    print(String(mmap.read_bytes()))

    # Memory mapped files must be closed once they are finished
    mmap.close()
```

Whereas an initial implementation of `Mmap` in a read/write context can be used as follows:

```mojo
    var length: Int32 = 24;
    var fd: FileDescriptor = open("./test.txt", "r+")

    # Initialize a mmap object for the particular file, with read/write and shared permissions
    var mmap = Mmap(fd=fd, length=length, offset=0, mode=MmapMode.WRITE)

    # Print out the entire file as a String
    print(String(mmap.read_bytes()))

    # Write bytes to mmap
    mmap.write("hello world")

    # Memory mapped files must be closed once they are finished
    mmap.close()
```
### Initial API

An initial API for `MmapMode` and `Mmap` could look like the below. 

```mojo
    struct MmapMode:
        alias READ: String = "READ"
        alias WRITE: String = "WRITE"
        alias EXEC: String = "EXEC"

    struct Mmap:
        var pointer: UnsafePointer[UInt8];
        var length: Int32;
        var offset: Int32;
        var mode: String;

    @staticmethod
    fn _get_prot(mode: String) raises -> Int32:
        """Helper method to translate between `MmapMode` and OS accepted prot values"""
        ...

    @staticmethod
    fn _get_flags(mode: String) raises -> Int32:
        """Helper method to translate between `MmapMode` and OS accepted flags values"""
        ...

    fn __init__(inout self, fd: FileDescriptor, length: Int32, offset: Int32, mode: String, address: UnsafePointer[UInt8]) raises:
        """Given a FileDescriptor, an expected length and offset, mmap configuration, and an initial address, initialize a `mmap` object"""
        ...

    fn __init__(inout self, length: Int32, offset: Int32, mode: String, address: UnsafePointer[UInt8]) raises:
        """Given an expected length and offset, mmap configuration and an initial address, initialize an anonymous `mmap` object"""
        ...

    fn close(inout self):
        """Close the existing `mmap` leveraging the OS `munmap` command."""
        ...

    fn read_bytes(inout self, owned size: Int32 = -1) raises -> List[UInt8]:
        """Reads data from the mapped file and sets the seek position. 
        If size is left as default of -1, it will read to the length provided during initialization."""
        ...

    fn write(inout self, data: String) raises:
        """Write data to the memory mapped file. Will error if opened in a MmapMode.READ configuration."""
        ...

    fn seek(inout self, size: Int32) raises:
        """Move the byte cursor forward in the memory mapped file."""
        ...

```

### Proof of Concept

An initial proof of concept, which provides for read-only memory mapped files is provided below:

```mojo
    alias PROT_READ: Int32 = 1;
    alias MAP_SHARED: Int32 = 0x01;

    fn page_size() -> Int32:
        return external_call["getpagesize", Int32]()

    struct MmapMode:
        alias CLOSEDD: String = "CLOSED"
        alias READ: String = "READ"

    struct Mmap:
        var pointer: UnsafePointer[UInt8]:
        var length: Int32;
        var offset: Int32;
        var mode: String;

        @staticmethod
        fn _get_prot(mode: String) raises -> Int32:
            if mode == MmapMode.READ:
                return PROT_READ
            else:
                raise "mode provided is not valid: " + mode + ", available options: ('READ')"

        @staticmethod
        fn _get_flags(mode: String) raises -> Int32;
            if mode == MmapMode.READ:
                return MAP_SHARED
            else:
                raise "mode provided is not valid: " + mode + ", available options: ('READ')"

        fn __init__(inout self, fd: FileDescriptor, length: Int32, offset: Int32, mode: String, address: UnsafePointer[UInt8] = UnsafePointer[UInt8]()) raises:

            # Calculate appropriate arguments
            var alignment = offset % page_size();
            var aligned_offset = offset - alignment;
            var aligned_len = length + alignment;

            var pointer = external_call["mmap", UnsafePointer[UInt8], UnsafePointer[UInt8], Int32, Int32, Int32, Int32, Int32](
                            address,
                            aligned_len,
                            self._get_prot(mode),
                            self._get_flags(mode),
                            fd.value,
                            aligned_offset
                            )

            if pointer < UnsafePointer[UInt8]():
                raise "unable to mmap file"

            self.pointer = pointer
            self.length = aligned_len
            self.offset = offset
            self.mode = mode

        fn close(inout self) raises:
            var err = external_call["munmap", Int32, UnsafePointer[UInt8], Int32](self.pointer, self.length)
            if err != 0:
                raise "unable to close mmap"
            else:
                self.mode = MmapMode.CLOSED

        fn seek(inout self, size: Int32):
            self.offset += size

        fn read_bytes(inout self, owned size: Int32 = -1) raises -> List[UInt8]:

            if self.mode == "CLOSED":
                raise "mmap already closed"

            var bytes = List[UInt8]()

            if size == -1:
                size = self.length

            for i in range(self.offset, self.offset + size):
                bytes.append((self.pointer + i)[])

            self.offset += size

            return bytes
```

## Background and Alternatives Considered

### How should the memory map be constructed?

The [mmap Man page for MacOS](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html), walks through mmap in detail. However, at a high level, `mmap` takes in, an address to map to, an expected size of the file, file protections, mapping options via flags, a file descriptor, and a byte offset. If successful, it returns a pointer to the start of the mapped region.

Few important pieces to consider:
* While the flags/protections can be different the vast majority of code for initializing a mmap is the same.
* The file descriptor only needs to remain open while the mmap is being initialized. If so, we may want to handle the context window surrounding the creation/deletion of the FileDescriptor in initialization as well.
* A mapped file in memory, must be closed, otherwise it can lead to memory leaks.

Given this information, there are two primary different ways we could initialize a Mmap object. The first way could parameterize a 'mode' which should be shorthand for the `prot` and `flags` fields:

```mojo
    struct MmapMode:
        alias READ: String = "READ"
        alias WRITE: String = "WRITE"
        alias EXEC: String = "EXEC"
        alias COW: String = "COW"

    struct Mmap:

        @staticmethod
        fn _get_prot(mode: String) -> Int32:
            ...

        @staticmethod
        fn _get_flags(mode: String) -> Int32:
            ...

        fn __init__(self, fd: FileDescriptor, length: Int32, offset: Int32, address: UnsafePointer[UInt8] = UnsafePointer[UInt8](), mode: String = MmapMode.READ) raises -> Self
            ...

```

There are a few pieces to clarify in this first method:

With the mode shorthand method, we would need two small helper methods to translate between the mmap mode, and prot/flags integers, provided to the OS.

Additionally, the address in the initialization function provides a hint for the OS to try to place the map within that address. We can default it to a null pointer, which would tell the OS to place the map wherever it would like, with the ability for the user to overwrite this value if they see fit.

An additional way this could be done is similar to other implementations in other languages (namely [memmap2-rs](https://github.com/RazrFalcon/memmap2-rs)). Instead of a enumerated mode, individually named factory functions are provided to configure the memory map. It would look something like this, with an additional constructor provided for each 'mode':

```mojo
    struct Mmap:

        @staticmethod
        fn map(self, fd: FileDescriptor, length: Int32, offset: Int32, address: UnsafePointer[UInt8]()) raises -> Self:
            ...

        @staticmethod
        fn map_mut(self, fd: FileDescriptor, length: Int32, offset: Int32, address: UnsafePointer[UInt8]()) raises -> Self:
            ...

        ...

```

Ultimately, I tend to prefer the first option, as it provides a few benefits. Firstly, the constructor is clearly identified, there is little ambiguity how this class is constructed. 

Secondly, it is simpler in the off chance, we wanted to provide the user with a context managed mmap, (one in which the `__enter__` and `__exit__` methods, both open and close the map automatically).

### Aside: Should we provide a `FileDescriptor` directly?

With the API above, the user is expected to open a FileDescriptor directly, and pass this to the Mmap constructor. Usage would look something like this:

```mojo
    # This only opens the file handle
    var fd: FileDescriptor = open("./test.txt", "r")
    var mmap = Mmap(fd=fd, length=15, offset=0)

    # We no longer need the file descriptor
    # thus we should close it.

    # We can then continue to use the mmap object as needed...
    print(String(mmap.read_bytes()))
```

Given this usage pattern, there are two outstanding thoughts, I had:

Firstly, Does the automatic conversion from `FileHandle` to `FileDescriptor` in the above, automatically close the `FileHandle`? If this is not the case, and the `FileHandle` needs to be deconstructed individually, we may want to manage this for the user directly.

Secondly, this puts the responsibility for aligning between file permissions and `MmapMode` on the user. Which may be unnecessary burdensome, given that the `mode` provided during `open` must be aligned with the `MmapMode`.

If we were to incorporate this responsibility in the initialization of the `Mmap` object, we could do something like this:

```mojo
    struct Mmap:
        fn __init__(inout self, path: String, length: Int32, offset: Int32, mode: String, ...) raises:

            # In this scenario, we would infer the file permissions
            # from the mode provided above.
            var fd: FileDescriptor = open(path, "r")

            # Initialize mmap object as before...

            # This is not a valid method on FileDescriptor
            # but suggested in the examples for the method
            fd.close()

```

