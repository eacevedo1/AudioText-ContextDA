#!/bin/bash

#SBATCH --job-name=sound_classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=18GB
#SBATCH --output=/scratch/ea3418/sound_classification_%j.out

EMBEDDINGS_PATH="urbansound8k_1219.pt \
                 urbansound8k-20240923151200_1228.pt \
                 urbansound8k-20240923161033_1245.pt \
                 urbansound8k-20240923171455_1302.pt \
                 urbansound8k-20240923180514_1236.pt \
                 urbansound8k-20240923185657_1253.pt \
                 urbansound8k-20240923195207_1311.pt"

TRAIN_MODES="zs tgap sv"

MODALITY_MODES="audio text"

TEMPERATURES="0.2 0.5 0.8"

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
for embeddings_path in $EMBEDDINGS_PATH; do
    for train_mode in $TRAIN_MODES; do
        echo '-- NO Domain Adaptation Sound classification -- \$embeddings_path -- \$train_mode --'
        python3 scripts/sound_classification.py --embeddings_path \$embeddings_path \
                                                --dataset urbansound8k \
                                                --mode \$train_mode \
                                                --save_results True
    done
done
for embeddings_path in $EMBEDDINGS_PATH; do
    for train_mode in $TRAIN_MODES; do
        for modality_mode in $MODALITY_MODES; do
            for temperature in $TEMPERATURES; do
                echo '-- Domain Adaptation Sound classification -- \$embeddings_path -- \$train_mode -- \$modality_mode --'
                python3 scripts/sound_classification.py --embeddings_path \$embeddings_path \
                                                        --dataset urbansound8k \
                                                        --mode \$train_mode \
                                                        --modality \$modality_mode \
                                                        --temperature \$temperature \
                                                        --bg_embeddings_path tau2019uas_1327.pt \
                                                        --save_results True
            done
        done
    done
done
echo 'Completed!'
"