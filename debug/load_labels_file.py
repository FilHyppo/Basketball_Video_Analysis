import os
import supervision as sv
import numpy as np
import argparse

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
        )
        detections_per_frame[frame_number] = sv.Detections.merge([detection, detections_per_frame[frame_number]])

    return detections_per_frame

def load_rim_ball_labels_xywh_top_corner(file_path: str, verbose=False) -> list[sv.Detections]:
    """
    Funzione per caricare label salvate in formato: frame_number, class_id, x1, y1, w, h, conf:
        - x1 e y1 SONO LE COORDINATE DELL' ANGOLO IN ALTO A SINISTRA DELLA BBOX
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tot_frames = int(lines[-1].split(', ')[0])
    tot_frames += 1
    detections_per_frame= [sv.Detections.empty() for _ in range(tot_frames)]
    for i, line in enumerate(lines[1:]):
        frame_number, class_id, x1, y1, w, h, conf = map(float, line.strip().split(', '))
        if verbose:
            print(f"Riga {i+1}: ", frame_number, class_id, x1, y1, w, h, conf)
        x2 = x1 + w
        y2 = y1 + h
        class_id = int(class_id)
        frame_number = int(frame_number)
        detection = sv.Detections(
            xyxy=np.array([x1,y1,x2,y2]).reshape(-1, 4),
            class_id=np.array([class_id]),
            confidence=np.array([conf]),
        )
        detections_per_frame[frame_number] = sv.Detections.merge([detection, detections_per_frame[frame_number]])

    return detections_per_frame

def load_rim_ball_labels_xywh_center(file_path: str, verbose=False) -> list[sv.Detections]:
    """
    Funzione per caricare label salvate in formato: frame_number, class_id, x1, y1, w, h, conf:
    - x1 e y1 SONO LE COORDINATE DEL CENTRO DELLA BBOX
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tot_frames = int(lines[-1].split(', ')[0])
    tot_frames += 1
    detections_per_frame= [sv.Detections.empty() for _ in range(tot_frames)]
    for i, line in enumerate(lines[1:]):
        frame_number, class_id, x_center, y_center, w, h, conf = map(float, line.strip().split(', '))
        if verbose:
            print(f"Riga {i+1}: ", frame_number, class_id, x_center, y_center, w, h, conf)
        x1, x2 = x_center - w/2, x_center + w/2
        y1, y2 = y_center - h/2, y_center + h/2
        class_id = int(class_id)
        frame_number = int(frame_number)
        detection = sv.Detections(
            xyxy=np.array([x1,y1,x2,y2]).reshape(-1, 4),
            class_id=np.array([class_id]),
            confidence=np.array([conf]),
        )
        detections_per_frame[frame_number] = sv.Detections.merge([detection, detections_per_frame[frame_number]])

    return detections_per_frame


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