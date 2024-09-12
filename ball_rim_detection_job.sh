#!/bin/bash
#SBATCH --job-name=ball_rim_detection
#SBATCH --account=cvcs2024
#SBATCH --partition=all_usr_prod
#SBATCH --output=logs/%jball_rim.out    
#SBATCH --error=logs/%jball_rim.err     
#SBATCH --gres=gpu:1                
#SBATCH --mem=16G

if [ "$#" -ne 2 ]; then
    echo "Errore: Devi fornire esattamente 2 parametri."
    echo "Utilizzo: $0 input_video_path output_dir_path"
    exit 1
fi

source /work/cvcs2024/Basketball_Video_Analysis/venv/bin/activate

cd /work/cvcs2024/Basketball_Video_Analysis/repo/Basketball_Video_Analysis/utils
python rim_ball_detection.py -i $1 -o $2
