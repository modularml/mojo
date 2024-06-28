
# remove potentially appended $HOME/.local/bin from PATH
PATH="${PATH%:$HOME/.local/bin}"

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] && [[ "$PATH" != *"$HOME/bin"* ]] ; then
    PATH="$HOME/bin:$PATH"
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] && [[ "$PATH" != *"$HOME/.local/bin"* ]] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
