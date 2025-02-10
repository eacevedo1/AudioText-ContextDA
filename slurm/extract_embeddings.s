#!/bin/bash

#SBATCH --job-name=extract_embd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=18GB
#SBATCH --output=/scratch/ea3418/extract_embd_%j.out

# Define the folder of the UrbanSound8K datasets in a single variable
US8k_FOLDER_PATHS="urbansound8k-20240923151200 \
                   urbansound8k-20240923180514 \
                   urbansound8k-20240923161033 \
                   urbansound8k-20240923185657 \
                   urbansound8k-20240923171455 \
                   urbansound8k-20240923195207"

cd /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-51257214

singularity \
    exec \
    --nv \
    --overlay overlay-25GB-500K.ext3:rw \
    cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate atm-domain-adapt
cd /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation
python3 scripts/extract_embeddings.py --dataset urbansound8k 
for us8k_folder in $US8k_FOLDER_PATHS; do
  echo '-- Extracting embeddings from UrbanSound8K path: $us8k_folder --'
  python3 scripts/extract_embeddings.py --dataset urbansound8k --path \$us8k_folder
done
echo '-- Extracting embeddings from TAU Urban Acoustic Scenes 2019 --'
python3 scripts/extract_embeddings.py --dataset tau2019uas
echo 'Completed!'
"