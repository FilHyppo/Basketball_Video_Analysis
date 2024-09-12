#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Errore: Devi fornire esattamente 2 parametri."
    echo "Utilizzo: $0 input_video_path output_dir_path"
    exit 1
fi

input_video_path = $1
output_dir_path = $2

cd /work/cvcs2024/Basketball_Video_Analysis/repo/Basketball_Video_Analysis

sbatch ball_rim_detection_job.sh $input_video_path $output_dir_path

# Job di fillo per players tracking

# Job di lollo per scoring detection

python utils/bva.py -i $input_video_path -o $output_dir_path