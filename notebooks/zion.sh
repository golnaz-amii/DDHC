#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=a40:1
#SBATCH --output=./notebook.out
#SBATCH --time=10:00:00

source .venv/bin/activate
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
