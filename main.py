from utils.scoring import BasketballScoreDetector
from utils.rim_detection import RimDetector
import cv2


# detector = BasketballHoopDetector(
#     video_path='input_videos/example.mp4',
#     output_path_color='outputs/example.avi',
#     hoop_regions=hoop_regions,
#     output_path_gray='outputs/example_gray.avi',  # Set to None if you don't want a gray video
#     verbose=True
# )
# detector.run()

video_capture = cv2.VideoCapture('input_videos/example.mp4')
output_video=cv2.VideoWriter('outputs/example.avi', cv2.VideoWriter_fourcc(*'XVID'), video_capture.get(cv2.CAP_PROP_FPS), 
                                            (int(video_capture.get(3)), int(video_capture.get(4))))

_, base_frame = video_capture.read()


rim_detector = RimDetector(model_path='models/rim-detection/yolo_v5(my-dataset).pt')
hoop_regions= rim_detector.detect_rims(base_frame, verbose=True)

for h in range(len(hoop_regions)):      #change  hoops regions for more robust scoring
    height = hoop_regions[h][1][1] - hoop_regions[h][0][1]
    width = hoop_regions[h][1][0] - hoop_regions[h][0][0]
    if h==0:
        hoop_regions[h][0]=(int(hoop_regions[h][0][0]-(0.2*width)), int(hoop_regions[h][0][1] +  (0.2*height))) 
        hoop_regions[h][1]= (hoop_regions[h][1][0], int(hoop_regions[h][1][1] +  (0.2*height)))
    else:
        hoop_regions[h][0]=(hoop_regions[h][0][0], int(hoop_regions[h][0][1] +  (0.2*height))) 
        hoop_regions[h][1]= (int(hoop_regions[h][1][0]+ (0.2*width)), int(hoop_regions[h][1][1] +  (0.2*height))) 


score_detector = BasketballScoreDetector(
    base_frame=base_frame,
    hoop_regions=hoop_regions,
    verbose=True
)

# detector = BasketballHoopDetector(
#     video_path='input_videos/example.mp4',
#     output_path_color='outputs/example.avi',
#     hoop_regions=hoop_regions,
#     output_path_gray='outputs/example_gray.avi',  # Set to None if you don't want a gray video
#     verbose=True
# )
# detector.run()

prec_score=None
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    score_detector.process_frame(frame)
    if prec_score != score_detector.states[0].score:
        print(score_detector.states[0].score)
    
    prec_score=score_detector.states[0].score


    frame=score_detector.write_score(frame)
    output_video.write(frame)


video_capture.release()
output_video.release()


    
