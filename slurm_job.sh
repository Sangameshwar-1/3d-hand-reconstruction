#!/bin/bash
#SBATCH --job-name=3dfist_finetune
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Setting up virtual environment..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install --default-timeout=1000 -r requirements.txt

echo "Running fine-tuning script..."
python3 finetune.py --dataset dataset_aligned --epochs 50 --batch-size 8

echo "Job complete."
