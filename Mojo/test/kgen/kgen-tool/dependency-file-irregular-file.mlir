// Dependency file generation fails when artifacts are not saved in
// regular files.
// RUN: echo "" | not kgen - --emit=object -d=%t -o /dev/null

// `/dev/null` may be considered an ordinary file on Windows.
