#!/bin/bash

# Slurm Resource Parameters (Example)
#SBATCH --ntasks=1                     # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -w bender
#SBATCH --cpus-per-task=1
#SBATCH -t 1-00:00              # Runtime in D-HH:MM
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --gres=gpu:2            # Number of gpus
#SBATCH --mem=10000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o training-outputs/hostname_%j.out      # File to which STDOUT will be written
#SBATCH -e training-outputs/hostname_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tscherli@andrew.cmu.edu # Email to which notifications will be sent

# After requested resources are allocated, run your program (for example in a docker container)

. /home/tscherli/.bash_profile

echo "Starting Docker Image"

pwd
docker ps
nvidia-docker ps

set -x

nvidia-docker run -v /data/datasets:/data/datasets -v /home/tscherli:/home/tscherli tscherli/alphatraining python3 /data/datasets/tscherli/task2/train/resbase/resgates4.py &

wait

echo "Done!"
