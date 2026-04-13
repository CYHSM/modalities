# Run on a login node or interactive session
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"

mkdir -p "${MY_ROOT}/venvs"

# Create venv using container's python, with access to container site-packages
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    "$CONTAINER" \
    python -m venv --system-site-packages "$OLMES_VENV"

# Clone olmes somewhere stable
cd "$MY_ROOT"
git clone https://github.com/allenai/olmes.git
cd olmes

# Install into the venv, from inside the container
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    "$CONTAINER" bash -c "
        source ${OLMES_VENV}/bin/activate
        pip install --no-build-isolation -e '.[gpu]'
    "