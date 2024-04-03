#!/bin/bash

#SBATCH --mail-user=jovian.l.wang@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=log-uken.txt
#SBATCH --account=cs3892-oguz_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2

module --ignore_cache load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle
cd ~
python -m venv myvenv
. myvenv/bin/activate
cd u-ken
pip install -r requirements.txt
python train.py 