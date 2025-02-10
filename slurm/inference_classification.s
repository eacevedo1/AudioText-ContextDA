#!/bin/bash

#SBATCH --job-name=inference_classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=18GB
#SBATCH --output=/scratch/ea3418/inference_classification_%j.out

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
python3 scripts/inference_classification.py --class_labels classes_demo.txt \
                                            --audio_folder_path demo/inference_demo \
                                            --modality text \
                                            --temperature 0.5 \
                                            --bg_type park
python3 scripts/inference_classification.py --class_labels classes_demo.txt \
                                            --audio_folder_path demo/inference_demo \
                                            --modality audio \
                                            --temperature 0.5 \
                                            --bg_folder_path demo/inference_bg_demo                                      
echo 'Completed!'
"