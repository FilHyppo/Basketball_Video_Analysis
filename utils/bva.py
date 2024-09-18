import os
import argparse
import cv2
import supervision as sv
from load_labels_file import load_rim_ball_labels_xyxy, load_scores, load_player_tracking, get_ball_rim_labels, get_court_corners, get_player_tracking, get_scores
from court_detection.geometry import *
from court_detection.motion_compensation import *
from rim_ball_detection import annotate_frame_ball_rim
from sanitize_labels import sanitize_players_detections


def draw_scores(frame, scores, color=(0, 255, 0)):
    cv2.putText(frame, f"Baskets: {scores[0]}", (50, 150) , cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Baskets: {scores[1]}", (frame.shape[1] -200, 150) , cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame


def link_rim_2_team(detections: sv.Detections, rim_bbox:np.ndarray) -> int:
    """
    Ritorna 0 se il canestro è difeso dal team 0, altrimenti 1 se difeso dal team 1
    """
    players_detections = detections[detections.class_id == 2]
    print("Players detections:", players_detections)
    mask_team0 = np.array(players_detections.data['team']) == 0
    mask_team1 = np.array(players_detections.data['team']) == 1

    players_bboxes_team0: np.ndarray = players_detections.xyxy[mask_team0]
    players_bboxes_team1: np.ndarray = players_detections.xyxy[mask_team1]

    print("BBoxes giocatori team 0:", players_bboxes_team0)
    print("BBoxes giocatori team 1:", players_bboxes_team1)

    sum_team0 = (np.abs(players_bboxes_team0 - rim_bbox)).sum(axis=0)[0]
    sum_team1 = (np.abs(players_bboxes_team1 - rim_bbox)).sum(axis=0)[0]

    if sum_team0 > sum_team1:
        return 1
    
    return 0

def get_shooter(detections_list: list[sv.Detections], side:str):
    
    last_detections = detections_list[-1]
    rim_bboxes = last_detections[last_detections.class_id == 1].xyxy.tolist()
    print("Rim_bboxes:", rim_bboxes)
    rim_bbox = None # [(x1,y1,x2,y2)]
    if side == 'right':
        rim_bbox:np.ndarray = max(rim_bboxes, key=lambda x:x[0])
    elif side == 'left':
        rim_bbox:np.ndarray = min(rim_bboxes, key=lambda x:x[0])

    team_defense = link_rim_2_team(last_detections, rim_bbox)
    print(team_defense)
    return team_defense
    ############################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-i", "--input_video", required=True)
    ap.add_argument("--court_type", required=True)
    args = ap.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    input_video_path = args.input_video
    
    court_type = args.court_type

    # Carica le etichette per palla e canestro dai file di testo
    players_detections_per_frame = get_player_tracking(input_video_path, output_dir)
    ball_rim_detections_per_frame: list[sv.Detections] = get_ball_rim_labels(input_video_path, output_dir)
    scoring_detection_per_frame = get_scores(input_video_path, output_dir)
    
    # Video input
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Carica i corner del campo
    corners = get_court_corners(input_video_path, output_dir)
    #Maschera dell'area interna alla linea da 3
    MASK_3_POINT_LINE = mask_3_point_line(corners, court_type=court_type, h=frame_height, w=frame_width)
    MASK_COURT = mask_court(corners, h=frame_height, w=frame_width)
    players_detections_per_frame = sanitize_players_detections(players_detections_per_frame, MASK_COURT)

    # Video output
    video_filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    ########################## INIZIO LOGICA ANALISI ################################

    NUM_SEC_PREV = 4

    frame_count = 0
    prev_frame = None
    previous_frames = []
    detections_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        previous_frames.append(frame)
        if len(previous_frames) == NUM_SEC_PREV * fps:
            previous_frames.pop(0)
        
        print(f"Frame {frame_count}/{tot_frames}")
        # Ottieni le detection per il frame corrente
        ball_rim_detections = ball_rim_detections_per_frame[frame_count]
        players_detections = players_detections_per_frame[frame_count]
        detections = sv.Detections.merge([ball_rim_detections, players_detections]) # class_id 0:ball, 1:rim, 2:player
        detections_list.append(detections)
        if len(detections_list) == NUM_SEC_PREV * fps:
            detections_list.pop(0)
        scoring_detection = scoring_detection_per_frame[frame_count]

        score_left, score_right = scoring_detection
        if score_left or score_right:
            side = 'left' if score_left else 'right'
            player_id = get_shooter(detections_list, side)
            frame = annotate_frame_ball_rim(frame, detections)
            cv2.imwrite("frame_annotated.jpg", frame)


        # Disegna le bounding box per palla e canestro
        #frame = annotate_frame_ball_rim(frame, detections)
        #frame = draw_scores(frame, scoring_detection, color=(0, 255, 0),)

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


############ DEBUG############
#python.exe .\utils\bva.py -o prova -i .\input_videos\1_1.mp4 --court_type 0