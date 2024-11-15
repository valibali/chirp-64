#!/bin/bash
set -ex

##
## Create some aliases
##
echo 'alias ll="ls -alF"' >> $HOME/.bashrc
echo 'alias la="ls -A"' >> $HOME/.bashrc
echo 'alias l="ls -CF"' >> $HOME/.bashrc

# Convenience workspace directory for later use
WORKSPACE_DIR=$(pwd)

# Change some Poetry settings to better deal with working in a container
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry config cache-dir ${WORKSPACE_DIR}/.cache
poetry config virtualenvs.in-project true

# Optional extra pip repositry to add to poetry
# Make sure this is reflecting whats in the pyproject.toml!
#poetry config http-basic.my-extra-pip-repo username password

# Now install all dependencies
poetry install

echo "Done!"