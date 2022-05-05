#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J UNet128
#SBATCH -p dgx
#SBATCH -t 72:00:00
#SBATCH -o output_dgx.out
#SBATCH --gres=gpu:4

# Load software

# Run python script
which conda
#/home/sci/kyle.anderson/anaconda3/bin/conda init bash
#source ~/.bashrc
eval "$(/home/sci/kyle.anderson/anaconda3/bin/conda shell.bash hook)"
conda activate base
which python
python /home/sci/kyle.anderson/lymph_nodes/Lymph-Node-Segmentation/KyleUNetTrainDDP.py -n 1 -nr 0 -g 4 -b 8 -e 300 -c KyleUNetWeightedBCE128.tar -l KyleUNetWeightedBCE128.log
