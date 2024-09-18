import os
import supervision as sv
import numpy as np
import argparse
import json

def load_rim_ball_labels_xyxy(file_path: str, verbose=False) -> list[sv.Detections]:
    """
    Funzione per caricare label salvate in formato: frame_number, class_id, x1, y1, x2, y2, conf
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tot_frames = int(lines[-1].split(',')[0])
    tot_frames += 1
    detections_per_frame= [sv.Detections.empty() for _ in range(tot_frames)]
    for i, line in enumerate(lines[1:]):
        frame_number, class_id, x1, y1, x2, y2, conf = map(float, line.strip().split(','))
        class_id = int(class_id)
        frame_number = int(frame_number)
        if verbose:
            print(f"Riga {i+1}: ", frame_number, class_id, x1, y1, x2, y2, conf)
        detection = sv.Detections(
            xyxy=np.array([x1,y1,x2,y2]).reshape(-1, 4),
            class_id=np.array([class_id]),
            confidence=np.array([conf]),
            tracker_id=np.array([0]), # track_id 0 per consistenza con il merge delle detections dei players
            data={"team": [2]}
        )
        detections_per_frame[frame_number] = sv.Detections.merge([detection, detections_per_frame[frame_number]])

    return detections_per_frame

def load_player_tracking(file_path: str):
    """
    Funzione per caricare label salvate in formato: 
    frame_number, id_giocatore, x, y, width, height, conf, team 

    Ritorna una lista di liste di tuple
    """

    with open(file_path, 'r') as f:
        # Leggi tutte le linee del file
        lines = f.readlines()
        
    tot_frames = int(lines[-1].split(',')[0])
    tot_frames += 1
    #detections = [[] for _ in range(tot_frames)]
    detections_per_frame:list[sv.Detections] = [sv.Detections.empty() for _ in range(tot_frames)]
    for line in lines:
        # Elimina eventuali spazi e separa i valori per ogni linea
        row = line.strip().split(',')

        # Estrai i valori dal file CSV
        frame_number = int(row[0])
        player_id = int(row[1])
        x = float(row[2])
        y = float(row[3])
        width = float(row[4])
        height = float(row[5])
        conf = float(row[6])
        team = int(float(row[7]))

        # Crea l'oggetto Detections
        detection = (
            player_id,
            x,
            y,
            width,
            height,
            conf,
            team
        )
        
        d = sv.Detections(
            xyxy=np.array([x, y, x + width, y + height]).reshape(-1, 4),
            class_id=np.array([2]),
            confidence=np.array([conf]),
            tracker_id=np.array([player_id]),
            data={"team": [team]},
        )

        # Aggiungi la rilevazione alla lista
        #detections[frame_number].append(d)
        detections_per_frame[frame_number] = sv.Detections.merge([detections_per_frame[frame_number], d])

    return detections_per_frame

def load_scores(file_path: str) -> list[tuple]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tot_frames = int(lines[-1].split(',')[0])
    tot_frames += 1
    scores= []
    for i, line in enumerate(lines):
        frame_number, score_left, score_right = map(int, line.strip().split(','))
        scores.append((score_left, score_right))
    return scores


def get_ball_rim_labels(input_video_path, output_dir):
    filename = os.path.basename(input_video_path).split('.')[0] + '.txt'
    labels_path = os.path.join(output_dir, "ball_rim_labels", filename)
    ball_rim_detections_per_frame = load_rim_ball_labels_xyxy(labels_path)
    return ball_rim_detections_per_frame

def get_player_tracking(input_video_path, output_dir):
    filename = os.path.basename(input_video_path).split('.')[0] + '.txt'
    labels_path = os.path.join(output_dir, "tracking_labels", filename)
    tracking_per_frame = load_player_tracking(labels_path)
    return tracking_per_frame

def get_scores(input_video_path, output_dir):
    filename = os.path.basename(input_video_path).split('.')[0] + '.txt'
    scores_path = os.path.join(output_dir, "scores", filename)
    scores = load_scores(scores_path)
    return scores

def get_court_corners(input_video_path, output_dir) -> dict:
    filename = os.path.basename(input_video_path).split('.')[0] + '.json'
    corners_path = os.path.join(output_dir, "corners", filename)
    with open(corners_path, 'r') as f:
        corners= json.load(f)

    return corners


def main(): # Main per testing
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-i", "--input_video", required=True)
    args = ap.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    filename = os.path.basename(args.input_video).split('.')[0] + '.txt'
    labels_path = os.path.join(output_dir, "labels", filename)

    detections_per_frame = load_rim_ball_labels_xyxy(labels_path)

    for frame_number, detections in enumerate(detections_per_frame):
        print(f"Per il frame {frame_number} sono state trovate le seguenti detections:\n", detections, "\n\n")

if __name__ == '__main__':
    main()
   