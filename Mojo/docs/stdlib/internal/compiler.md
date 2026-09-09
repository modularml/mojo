## WARNING

Everything in this file is subject to revision on any bugfix or security
update. We (the stdlib team and contributors), reserve the right to remove,
change the API contracts of, rename, or cause to instantly crash the program,
any operation described in here. These are **PRIVATE** APIs and implementation
details for the Mojo stdlib and for MAX to use. **WE WILL CHANGE IT WHENEVER
WE FIND IT CONVENIENT TO DO SO WITHOUT WARNING OR NOTICE**.

## Compiler Docs

### MLIR Interpreter

The MLIR Interpreter is the mechanism by which Mojo evaluates code at compile
time.

#### Current Limitations

From:
[Chris on Discord](https://discord.com/channels/1087530497313357884/1339917438372020264)

- No access to target information (much of `sys.info` doesn't work)
- Runs before elaboration, meaning some information is not available.
