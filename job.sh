#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=cvcs2024
#SBATCH --partition=all_usr_prod
#SBATCH --output=logss/score4.out    
#SBATCH --error=logss/score4.err     
#SBATCH --gres=gpu:1                
#SBATCH --mem=32G

source /work/cvcs2024/Basketball_Video_Analysis/venv/bin/activate

cd /work/cvcs2024/Basketball_Video_Analysis/repo/Basketball_Video_Analysis/
python utils/rim_ball_detection.py -o test -i input_videos/1_1.mp4