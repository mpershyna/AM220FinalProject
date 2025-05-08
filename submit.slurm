#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpershyna@g.harvard.edu

# load modules
module load python/3.12.8-fasrc01
module load cuda/12.4.1-fasrc01

# set environment variables
export WANDB_DIR=/n/home01/mpershyna/AM220/CEConv/wandb
export DATA_DIR=/n/home01/mpershyna/AM220/CEConv/data
export OUT_DIR=/n/home01/mpershyna/AM220/CEConv/output

# run code
source venv/bin/activate
#python -m experiments.color_mnist.train_biased --std 0 --rotations 3 --planes 17 --separable
#python -m experiments.color_mnist.train_biased --std 12 --rotations 3 --planes 17 --separable
#python -m experiments.color_mnist.train_biased --std 36 --rotations 3 --planes 17 --separable
#python -m experiments.color_mnist.train_biased --std 48 --rotations 3 --planes 17 --separable
#python -m experiments.color_mnist.train_biased --std 60 --rotations 3 --planes 17 --separable
#python -m experiments.color_mnist.train_biased --std 1000000 --rotations 3 --planes 17 --separable

python -m experiments.classification.train --rotations 3 --separable
python -m experiments.classification.train --rotations 3 --separable --jitter 0.5
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --jitter 0.5
