#!/bin/bash

#SBATCH --job-name=soundscape_augmentations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --output=/scratch/ea3418/soundscape_augmentations_%j.out

# Define the background types in a single variable
BACKGROUNDS_TYPES='park public_square street_traffic shopping_mall metro_station airport'

cd /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-51257214

singularity \
    exec \
    --overlay overlay-25GB-500K.ext3:rw \
    cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate atm-domain-adapt
cd /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation
for bg_type in $BACKGROUNDS_TYPES; do
  echo '-- Park - SNR 6 db - 1 soundscape - 10 folds - -36 ref dB --'
  python3 scripts/soundscape_augmentations.py --folds 1,2,3,4,5,6,7,8,9,10\
                                              --parameters snr_dist=\(const,6\)\ n_soundscapes=1\ bg=\$bg_type\ ref_db=-36
done
echo 'Completed!'
"