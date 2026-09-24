#!/bin/bash
#SBATCH --job-name=train_model_1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8 # Number of Cores per Task
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
##SBATCH --gres=gpu:1 # This is commented
#SBATCH --exclude=toltec-cpu[001-007]

#module load r-rocker-ml-verse/4.2.3+apptainer
module load apptainer/latest                  # load the apptainer/singularity runtime
shopt -s expand_aliases
CONTAINER=/work/pi_drsheldon_umass_edu/birdflow_modeling/BirdFlowContainer/BirdFlowContainer.sif

apptainer exec \
  --bind '' \
  $CONTAINER \
  Rscript --no-restore --quiet --no-save ./kfold.R