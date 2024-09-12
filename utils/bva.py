import os
import argparse
import cv2
from load_labels_file import load_rim_ball_labels_xyxy
from court_detection.geometry import *
import supervision as sv

def get_ball_rim_labels(input_video_path, output_dir):
    filename = os.path.basename(input_video_path).split('.')[0] + '.txt'
    labels_path = os.path.join(output_dir, "ball_rim_labels", filename)
    ball_rim_detections_per_frame = load_rim_ball_labels_xyxy(labels_path)

    return ball_rim_detections_per_frame

def get_court_corners() -> dict:
    pass

def draw_detections(frame, detections, color=(0, 255, 0), label="Detection"):
    # Controlla se detections è un'istanza di sv.Detections
    if isinstance(detections, sv.Detections):
        for i in range(len(detections.xyxy)):
            # Estrai le coordinate della bounding box
            x1, y1, x2, y2 = detections.xyxy[i]
            # Estrai il punteggio di confidenza
            score = detections.confidence[i]
            # Disegna il rettangolo (bounding box)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Aggiungi l'etichetta della detection (se necessaria)
            cv2.putText(frame, f'{label}: {score:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print("Formato delle rilevazioni non supportato.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-i", "--input_video", required=True)
    args = ap.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    input_video_path = args.input_video

    # Carica le etichette per palla e canestro
    ball_rim_detections_per_frame = get_ball_rim_labels(input_video_path, output_dir)

    # Carica i corner del campo
    corners = get_court_corners()

    # TODO: Inserire il caricamento delle altre detection (scoring e giocatori)
    scoring_detection_per_frame = ... # LOLLO
    players_detections_per_frame = ... # FILLO

    ########################## INIZIO LOGICA ANALISI ################################

    # Video input
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video output
    output_video_path = os.path.join(output_dir, 'output_with_detections.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ottieni le detection per il frame corrente
        ball_rim_detections = ball_rim_detections_per_frame[frame_count]
        #scoring_detection = scoring_detection_per_frame[frame_count]
        #players_detections = players_detections_per_frame[frame_count]

        # Disegna le bounding box per palla e canestro
        draw_detections(frame, ball_rim_detections, color=(0, 255, 0), label="Ball/Rim")
        #draw_detections(frame, scoring_detection, color=(255, 0, 0), label="Score")
        #draw_detections(frame, players_detections, color=(0, 0, 255), label="Player")

        # Aggiorna i corners se la telecamera si fosse mossa leggermente
        if frame_count == 0:
            prev_frame = frame
        if frame_count % 10000000 == 0 and frame_count != 0:
            cur_frame = frame
            corners = new_corners(prev_frame, cur_frame, corners)

        # Scrivi il frame con le bounding box sul video di output
        out.write(frame)

        frame_count += 1

    # Rilascia le risorse
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
