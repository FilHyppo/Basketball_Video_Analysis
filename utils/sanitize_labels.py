from load_labels_file import *
from court_detection.geometry import mask_3_point_line, mask_court, mask_free_throw_lines
import cv2

def sanitize_players_detections(players_detections_per_frame, MASK_COURT):
    """
    Per togliere detections al di fuori del campo
    """
    h, w = MASK_COURT.shape[:2]
    cnt_out_of_bounds = 0
    for frame_number in range(len(players_detections_per_frame)):
        players_detections = players_detections_per_frame[frame_number]

        valid_players_detections = np.ones(len(players_detections))
        for i, bbox in enumerate(players_detections.xyxy):
            bottom_center_point = [int(bbox[0] * 0.5 + bbox[2] * 0.5), int(bbox[3])]
            tol = 0.01 # tolleranza ai bordi
            check_1 = MASK_COURT[int(bottom_center_point[1] + tol*h), bottom_center_point[0]] == 255
            check_2 = MASK_COURT[int(bottom_center_point[1] - tol*h), bottom_center_point[0]] == 255
            if not check_1 and not check_2:
                valid_players_detections[i] = 0
                print(f"{cnt_out_of_bounds}-th point out of bounds:", bottom_center_point)
                img = MASK_COURT.copy()
                img = np.dstack([img]*3)
                img = cv2.circle(img, bottom_center_point, 10, (0, 0, 255), -1)
                cv2.imwrite(f"debug/masks/mask{cnt_out_of_bounds}.jpg", img)
                cnt_out_of_bounds += 1
        players_detections_per_frame[frame_number] = players_detections[valid_players_detections == 1]
    
    return players_detections_per_frame

def sanitize_ball_labels(ball_rim_detections_per_frame):
    """
    Per eliminare le detection multiple e sopratutto che abbiano un senso (max distance)
    """
    pass

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

    players_detections_per_frame = get_player_tracking(input_video_path, output_dir)
    ball_rim_detections_per_frame: list[sv.Detections] = get_ball_rim_labels(input_video_path, output_dir)
    scoring_detection_per_frame = get_scores(input_video_path, output_dir)
    

    corners = get_court_corners(input_video_path, output_dir)
    #Maschera dell'area interna alla linea da 3
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    MASK_3_POINT_LINE = mask_3_point_line(corners, court_type=court_type, h=frame_height, w=frame_width)
    MASK_COURT = mask_court(corners, h=frame_height, w=frame_width)


    players_detections_per_frame = sanitize_players_detections(players_detections_per_frame, MASK_COURT)
    
if __name__ == '__main__':
    main()
