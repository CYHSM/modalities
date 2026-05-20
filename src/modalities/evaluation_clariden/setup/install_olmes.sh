#!/bin/bash
set -u

# Run on a login node or interactive session
MY_ROOT="/users/markusfrey"
OLMES_VENV="${MY_ROOT}/venvs/olmes"

mkdir -p "${MY_ROOT}/venvs"

# We MUST recreate this venv using --system-site-packages so it adopts the exact uenv Torch operators
if [ ! -d "$OLMES_VENV" ]; then
    python3 -m venv --system-site-packages "$OLMES_VENV"
fi

# Clone olmes somewhere stable if it does not exist
cd "$MY_ROOT"
if [ ! -d "olmes" ]; then
    git clone https://github.com/allenai/olmes.git
fi
cd olmes

# Install into the olmes venv natively
source "${OLMES_VENV}/bin/activate"
pip install --upgrade pip setuptools wheel
pip install .

echo "OLMES securely installed into dedicated venv at ${OLMES_VENV}"