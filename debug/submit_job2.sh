#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=cvcs2024
#SBATCH --partition=all_usr_prod
#SBATCH --output=logs/%jtest_predictor.out    
#SBATCH --error=logs/%jtest_predictor.err     
#SBATCH --gres=gpu:1                
#SBATCH --mem=32G

source /work/cvcs2024/Basketball_Video_Analysis/venv/bin/activate

cd /work/cvcs2024/Basketball_Video_Analysis/repo/Basketball_Video_Analysis/debug
python test_rim_ball_detections.py