//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

/// Define a representation for .svg files, that allows for importing them as
/// strings.
declare module '*.svg' {
  const script: string;
  export default script;
}
