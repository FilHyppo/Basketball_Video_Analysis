#!/bin/bash

#SBATCH --partition=all_usr_prod     # Specifica la partizione
#SBATCH --account=cvcs2024           # Specifica l'account di progetto
#SBATCH --gres=gpu:4                 # Richiede una GPU
#SBATCH --job-name=rim-detection        # Nome del job
#SBATCH --output=/homes/lcorradi/output_%j.txt       # Specifica il file di output
#SBATCH --error=/homes/lcorradi/error_%j.txt         # Specifica il file di errori
#SBATCH --time=24:00:00 


# Attiva il virtual environment
source /work/cvcs2024/Basketball_Video_Analysis/venv/bin/activate

# Esegui lo script Python
srun python /work/cvcs2024/Basketball_Video_Analysis/rim_detection/rim-detection.py
