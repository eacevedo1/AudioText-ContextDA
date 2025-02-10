#!/bin/bash

#SBATCH --job-name=create_singularity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=4GB
#SBATCH --output=/scratch/ea3418/create_singularity_status.out

# NOTE: Change ea3418 to your username

echo "Starting job $SLURM_JOB_ID"
echo "Creating Singularity environment"

random_number=$SLURM_JOB_ID

mkdir /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-$random_number
cd /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-$random_number

echo "Copying files to the node"

cp -rp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip overlay-25GB-500K.ext3.gz

cp /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif .

echo "Creating Singularity environment"

singularity \
    exec \
    --overlay overlay-25GB-500K.ext3:rw \
    cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
echo 'Step 1: Install Miniconda'
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
cp /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation/slurm/env.sh /ext3/env.sh
source /ext3/env.sh
conda update -n base conda -y
conda clean --all --yes
conda install pip -y
echo 'Step 3: Create conda environment'
conda create --name atm-domain-adapt python=3.9
conda activate atm-domain-adapt
cd /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation
pip install -r requirements.txt
echo 'Step 4: Download the model'
mkdir -p models/LAION-CLAP
wget -P models/LAION-CLAP https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
echo 'Completed'
"