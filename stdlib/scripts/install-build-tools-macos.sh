# Install `lit` for use in the tests
brew install lit

# Ensure `FileCheck` from the pre-installed LLVM 15 package is visible
echo $(brew --prefix llvm@15)/bin/ >> $GITHUB_PATH
