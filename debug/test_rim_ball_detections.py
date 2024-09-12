import os
import sys
utils = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(utils)
from utils.rim_ball_detection import RimBallDetector
import cv2
import supervision as sv
import time
import numpy as np

def test_video(model: RimBallDetector, input_video_path: str):
    output_video_path = f"video/{os.path.basename(input_video_path).split('.')[0]}.mp4"
    label_file_path = f"labels/{os.path.basename(input_video_path).split('.')[0]}.txt"
    start_time = time.time()
    model.predict(input_video_path, output_video_path, save_txt=True, label_file_path=label_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo per inferenza: {elapsed_time:.6f} secondi")

def test_frame(model: RimBallDetector, input_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    for i in range(10):
        ret, frame = cap.read()
        
        detections = model.predict_frame(frame)
        
        frame_annotated = model.annotate_frame(frame, detections)
        cv2.imwrite(f"frame_annotated_{i}.jpg", frame_annotated)


def test_rim_bboxes(model: RimBallDetector, input_video_path: str):
    start_time = time.time()
    bboxes = model.get_hoop_region(input_video_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo di per inferenza: {elapsed_time:.6f} secondi")
    print("BBoxes rim: ", bboxes)
    cap = cv2.VideoCapture(input_video_path)
    _, frame = cap.read()

    detections = sv.Detections(
        xyxy=bboxes,
        class_id=np.array([1 for _ in range(len(bboxes))]),
        confidence=np.array([0.99 for _ in range(len(bboxes))])
    )

    frame = model.annotate_frame(frame, detections)

    cv2.imwrite("prova.jpg", frame)

if __name__ == '__main__':
    start_time = time.time()
    model_path = '../best_weights.pt'
    model = RimBallDetector(model_path=model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo di caricamento del modello YOLO: {elapsed_time:.6f} secondi")

    input_video_path = 'video_camera_fissa_20240904_150610.mp4'

    test_frame(model, input_video_path)
    #test_video(model, input_video_path)
    #test_rim_bboxes(model, input_video_path)