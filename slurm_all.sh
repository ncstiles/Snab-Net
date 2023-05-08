#!/bin/bash
#SBATCH -J myjob_4GPUs
#SBATCH -o myjob_4GPUs_%j.out
#SBATCH -e myjob_4GPUs_%j.err
#SBATCH --mail-user=nstiles@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=cv_project_2
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

## Horovod execution
python -m visdom.server & 

echo " Run of baseline started at:- "
date
git checkout master_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv1 started at:- "
date
git checkout ablate_conv1_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv2 started at:- "
date
git checkout ablate_conv2_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv3 started at:- "
date
git checkout ablate_conv3_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv4 started at:- "
date
git checkout ablate_conv4_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv12 started at:- "
date
git checkout ablate_conv12_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv13 started at:- "
date
git checkout ablate_conv13_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv14 started at:- "
date
git checkout ablate_conv14_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv23 started at:- "
date
git checkout ablate_conv23_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv24 started at:- "
date
git checkout ablate_conv24_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date

echo " Run of ablating conv34 started at:- "
date
git checkout ablate_conv34_viz
python main.py --data ISIC2018 --val_folder folder1 --id Comp_Atten_Unet 
echo "Run completed at:- "
date


