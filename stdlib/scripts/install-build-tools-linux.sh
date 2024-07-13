LLVM_VERSION=17
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh $LLVM_VERSION
rm llvm.sh

# Make common LLVM binaries (including FileCheck) in our PATH so they work when used in an unversioned context
# For example, instead of saying `FileCheck-17` which exists in `/usr/bin`, this allows us to just call
# FileCheck unqualified.
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$LLVM_VERSION 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$LLVM_VERSION 100
sudo update-alternatives --install /usr/bin/lld lld /usr/bin/lld-$LLVM_VERSION 100
sudo update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-$LLVM_VERSION 100
sudo update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-$LLVM_VERSION 100
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-$LLVM_VERSION 100

python3 -m pip install lit
