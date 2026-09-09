# MojoLLDB

**Note: the intended audience for this document are MojoLLDB developers and not
end users.**

MojoLLDB is the LLDB plugin that provides support for debugging Mojo programs
and that powers the Mojo REPL.

It includes the following core components:

- MojoTypeSystem, which translates MLIR into LLDB entities.
- MojoDWARFParser, which translates DWARF into MLIR.
- REPL, which sets up the `mojo repl` tool.
- ExpressionParser, which allows expression evaluation in Mojo.
- MojoLanguage, which defines data formatters and synthetic providers for nicer
  variable printing.

## Error handling

LLDB's philosophy on error handling is that LLDB must never crash in a release
build. The
[official documentation](https://lldb.llvm.org/resources/contributing.html#error-handling-and-use-of-assertions-in-lldb)
elaborates on this topic. As a useful note, the last phrase in that document is
of utmost relevance:

> Please keep in mind that the debugger is often used as a last resort, and a
crash in the debugger is rarely appreciated by the end-user.

What does this mean for us? It means that, for the end user, MojoLLDB shouldn't
crash even if the compiler produces the wrong input. That is impossible to
enforce in all cases, but we should do our best incrementally.

### Error messages with `EMIT_BUG_REPORT_MESSAGE`

We want errors to be actionable, which in practice translates into bug reports.
MojoLLDB developers are expected to print error messages with
`EMIT_BUG_REPORT_MESSAGE`, which besides letting the user know of an error in
the debugger, it asks them to submit a bug report. Eventually we could add more
automation where the bug report is pre-filled for them, as well as integrating
this with the IDE.

Special care must be given to making sure that the error messages are not too
low level, which helps users provide good error reports that they can
understand.

An interesting detail is that this utility prints right away, even if it shows
up in the middle of other output in the terminal. This behavior is non-standard
in LLDB, which almost always collects error messages and returns them as a
secondary result of a command. LLDB's rationale is to avoid mixing error
messages with the actual command output in the terminal. This makes sense in the
context of `LLDB CLI`, but it doesn't in the context of`lldb-dap`, where the
`Debug Console` is nice enough to display error messages and regular outputs in
different ways. Besides that, `lldb-dap` users don't really execute commands,
they instead execute GUI actions that happen to be backed by a series of
commands. As we want the `lldb-dap` experience to be our main focus, we can
allow ourselves to be intrusive in terms of error messaging, for the sake of
gathering as much feedback as possible and making errors actionable as soon as
they happen.
