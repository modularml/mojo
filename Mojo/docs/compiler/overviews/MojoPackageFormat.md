# Mojo Package Format (🔥📦)

## Overview

The Mojo Package format is a binary format consisting of a short header section
followed by serialized MLIR bytecode.

Unless otherwise specified, everything below pertains to version 1 of this
format.

### Header Section

The header contains any necessary information that is unsuitable or impossible
to encode in the MLIR.

The header consists of the 4-byte "magic bytes", a version number, and three
currently unused bytes.

Two versions are encoded in the header: the first is the version of the Mojo
language that was used to create the package; the second is the version of the
Modular platform used to create the package.

Each version consists of 4 components: major, minor, patch, and label. The
major and minor components are a single byte each; the patch is a 2-byte
integer. Each integer is interpreted as an unsigned number. Each version also
encodes a nul-terminated string as the "label".

The third component is a nul-terminated checksum string that represents the
internal version of the MLIR bytecode used to encode the rest of the package.

The header is then aligned up to a multiple of 8 bytes in size. The padding
bytes are ignored when read and carry no meaning.

``` sh
header {
  magic: "MKPG"
  encoding_version: uint8_t
  reserved: uint8_t[3]

  // Mojo Version
  mojo_ver_major : uint8_t
  mojo_ver_minor : uint8_t
  mojo_ver_patch : uint16_t
  mojo_ver_label : string

  // Modular Version
  modular_ver_major : uint8_t
  modular_ver_minor : uint8_t
  modular_ver_patch : uint16_t
  modular_ver_label : string

  mlir_checksum : string

  alignment_padding : uint8_t[*]
}
```

### MLIR Section

This part of the package (and indeed, the rest of the package) is serialized
MLIR bytecode as emitted by LLVM.
