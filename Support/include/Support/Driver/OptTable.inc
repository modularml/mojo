//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file uses x-macros to define an options enum and `llvm::opt::OptTable`
// based on the option definitions in a separate file, which must be located at
// `DRIVER_OPTIONS_PATH`.
//
//===------------------------------------------------------------*- C++ -*-===//

#ifndef DRIVER_OPTIONS_PATH
#error "DRIVER_OPTIONS_PATH must be defined before including this header"
#endif

namespace {
namespace options {
enum ID {
  OPT_INVALID = 0,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTFORVARIANTS, METAVAR,      \
               VALUES, SUBCOMMANDIDSOFFSET)                                    \
  OPT_##ID,
#include DRIVER_OPTIONS_PATH
#undef OPTION
  LastOption
};
} // namespace options
} // namespace

#define OPTTABLE_STR_TABLE_CODE
#include DRIVER_OPTIONS_PATH
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include DRIVER_OPTIONS_PATH
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_PREFIXES_UNION_CODE
#include DRIVER_OPTIONS_PATH
#undef OPTTABLE_PREFIXES_UNION_CODE

using llvm::opt::DefaultVis;
using llvm::opt::HelpHidden;
static constexpr llvm::opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTFORVARIANTS, METAVAR,      \
               VALUES, SUBCOMMANDIDSOFFSET)                                    \
  {PREFIX,                                                                     \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   HELPTEXTFORVARIANTS,                                                        \
   METAVAR,                                                                    \
   options::OPT_##ID,                                                          \
   llvm::opt::Option::KIND##Class,                                             \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   VISIBILITY,                                                                 \
   options::OPT_##GROUP,                                                       \
   options::OPT_##ALIAS,                                                       \
   ALIASARGS,                                                                  \
   VALUES,                                                                     \
   SUBCOMMANDIDSOFFSET},
#include DRIVER_OPTIONS_PATH
#undef OPTION
};

#undef DRIVER_INCLUDE
#undef DRIVER_OPTIONS_PATH
