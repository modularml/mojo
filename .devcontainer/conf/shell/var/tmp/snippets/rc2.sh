
# enable auto-completion for magic
if [ -n "$ZSH_VERSION" ]; then
    eval "$(magic completion --shell zsh)"
elif [ -n "$BASH_VERSION" ]; then
    eval "$(magic completion --shell bash)"
fi
