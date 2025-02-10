#!/bin/bash

#SBATCH --job-name=download_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --output=/scratch/ea3418/download_datasets.out

cd /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-51257214

singularity \
    exec \
    --overlay overlay-25GB-500K.ext3:rw \
    cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate atm-domain-adapt
cd /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation
python3 scripts/download_dataset.py --dataset_name urbansound8k 
python3 scripts/download_dataset.py --dataset_name tau2019uas 
"
