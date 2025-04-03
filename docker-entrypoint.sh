#!/bin/bash
set -e

# Activate conda environment
# source /opt/conda/etc/profile.d/conda.sh
conda activate aia

# Run based on the argument
# if [ "$1" = 'app' ]; then
#     echo "Starting web application..."
#     cd /app
#     python app.py
# elif [ "$1" = 'notebook' ]; then
#     echo "Starting Jupyter notebook server..."
#     cd /app/src/notebooks
#     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# else
#     exec "$@"
# fi