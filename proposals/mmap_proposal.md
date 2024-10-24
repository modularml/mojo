# Stdlib Proposal for mmap

A `mmap` module has mentioned by the community and welcomed by the core team.
This proposal walks through a proposed basic API along with a brief proof of concept.

## What is Memory Mapping and how is it used?

`mmap` creates a new mapping in the virtual address space of the program.
It allows for files to be mapped directly into memory, enabling faster IO operations.

It has a few primary advantages:

- It is efficient for random access on large files.
- It allows multiple processes to share memory.
- It can improve performance by reducing system calls and copy operations.

It uses demand paging, which means that files are not immediately read from disk.
The OS manages this mapping and returns additional pages of data as needed.

## Proposal

The introduction of a `mmap` module, with two primary objects:

A `MmapMode`, an enum struct, identifying the four initial `mmap` configurations:

- Read Only
- Write Only
- Exec
- Copy On Write

A `Mmap` object, responsible for creating and working with mmap files.

### High Level Usage

An initial implementation of `Mmap` in a read-only context can be used as follows.
Note, the API will provide functionality similar to the existing `FileHandle` API.

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

Whereas an initial implementation of `Mmap` in a write context can be used as follows:

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

An initial proof of concept, for read-only memory mapped files is provided below:

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

The mmap Man page for MacOS walks through mmap in detail.
However, at a high level, `mmap` takes in:

- an address to map to
- an expected size of the file
- file protections
- mapping options via flags
- a file descriptor
- a byte offset

If successful, it returns a pointer to the start of the mapped region.

Few important pieces to consider:

- While the configurations vary the majority of code for creating a mmap is the same.
- The file descriptor only needs to remain open while the mmap is being initialized.
- A mapped file in memory, must be closed, otherwise it can lead to memory leaks.

Given this, there are two primary different ways we could initialize a Mmap object.
The first way would include a 'mode', which stands in for 'prot' and 'flags' options:

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

        fn __init__(self, 
                    fd: FileDescriptor, 
                    length: Int32, 
                    offset: Int32, 
                    address: UnsafePointer[UInt8] = UnsafePointer[UInt8](), 
                    mode: String = MmapMode.READ) raises -> Self
            ...

```

There are a few pieces to clarify in this first method:

With this method, we would need helper methods to translate between the mode provided,
and OS flags.

Additionally, the address in the init function is only a hint for the OS.
If provided with a null pointer, the OS will choose the map location.
We've set a sane default, while allowing the user to change this value if needed.

An additional way this oculd be done is similar to implementations in other languages.
Instead of an enumerated mode, individually named factory functions could be used.
Each function would provided a specifically configured map.
It would look something like this:

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

I tend to prefer the first option, as it provides a few benefits:

- The constructor is clearly identified.
- Simpler, in case we want to provide the user with a context managed mmap object.

### Aside: Should we provide a `FileDescriptor` directly?

With the API above, the user is expected to open a FileDescriptor directly prior.
This is then passed to the `Mmap` constructor.

Usage would look something like this:

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

Does the conversion from `FileHandle` to `FileDescriptor`, close the `FileHandle`?
If the `FileHandle` needs to be closed, we can manage this for the user directly.

Secondly, file permissions and `MmapMode` are related.
The above, leaves the responsibility for this with the user.
This burden can be alleviated by managing the `FileHandle` ourselves.

To incorporate this responsibility during initialization instead, we could do this:

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
