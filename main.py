from utils.scoring import BasketballScoreDetector
from utils.rim_detection import RimDetector
from utils.rim_ball_detection import RimBallDetector
import cv2
from utils.ball_color import get_ball_color_range


# Carica l'immagine
image_path = 'outputs/frame_1570.jpg'
frame = cv2.imread(image_path)

# Definisci la bounding box (esempio)
bounding_box = (213,674,23,23)  # (x, y, w, h)

# Calcola il range di colore della palla
lower_ball_color, upper_ball_color = get_ball_color_range(bounding_box, frame)





video_capture = cv2.VideoCapture('input_videos/out.mp4')
output_video=cv2.VideoWriter('outputs/out.avi', cv2.VideoWriter_fourcc(*'XVID'), video_capture.get(cv2.CAP_PROP_FPS), 
                                            (int(video_capture.get(3)), int(video_capture.get(4))))
_, base_frame = video_capture.read()

rim_detector = RimDetector(model_path='models/rim-detection/yolo_v5(my-dataset).pt')
hoop_regions= rim_detector.detect_rims(base_frame, verbose=True)

#rb_detector= RimBallDetector(model_path='models/rim_ball_detection/best_weights.pt')
#hoop_regions= rb_detector.get_hoop_regions('input_videos/clip3.mp4')
#print(hoop_regions)
#rb_detector.predict('input_videos/clip3.mp4', 'outputs/rim_ball.avi', save=True, save_txt=False)

# detector = BasketballScoreDetector(
#     video_path='input_videos/1_1.mp4',
#     output_path_color='outputs/example.avi',
#     hoop_regions=hoop_regions,
#     output_path_gray='outputs/example_gray.avi',  # Set to None if you don't want a gray video
#     verbose=True,
#     output_txt = 'outputs/labels.txt'
# )
# detector.run()





# # for h in range(len(hoop_regions)):      #change  hoops regions for more robust scoring
# #     height = hoop_regions[h][1][1] - hoop_regions[h][0][1]
# #     width = hoop_regions[h][1][0] - hoop_regions[h][0][0]
# #     if h==0:
# #         hoop_regions[h][0]=(int(hoop_regions[h][0][0]-(0.2*width)), int(hoop_regions[h][0][1] +  (0.2*height))) 
# #         hoop_regions[h][1]= (hoop_regions[h][1][0], int(hoop_regions[h][1][1] +  (0.2*height)))
# #     else:
# #         hoop_regions[h][0]=(hoop_regions[h][0][0], int(hoop_regions[h][0][1] +  (0.2*height))) 
# #         hoop_regions[h][1]= (int(hoop_regions[h][1][0]+ (0.2*width)), int(hoop_regions[h][1][1] +  (0.2*height))) 


score_detector = BasketballScoreDetector(
    base_frame=base_frame,
    hoop_regions=hoop_regions,
    verbose=True
)

#score_detector.upper_ball_color = upper_ball_color
#score_detector.lower_ball_color = lower_ball_color

# # detector = BasketballHoopDetector(
# #     video_path='input_videos/example.mp4',
# #     output_path_color='outputs/example.avi',
# #     hoop_regions=hoop_regions,
# #     output_path_gray='outputs/example_gray.avi',  # Set to None if you don't want a gray video
# #     verbose=True
# # )
# # detector.run()

prec_score=None
count=0
file=open('outputs/out.txt', 'w')
file.write(f"{0},{score_detector.states[0].score},{score_detector.states[1].score}\n")
last_score_sx=0
last_score_dx=0
while video_capture.isOpened():
    count+=1 
    ret, frame = video_capture.read()
    if not ret:
        break
    score_detector.process_frame(frame)
    if prec_score != score_detector.states[0].score:
        print(score_detector.states[0].score)
    
    prec_score=score_detector.states[0].score
    
    if last_score_sx != score_detector.states[0].score:
        last_score_sx=score_detector.states[0].score
        file.write(f"{count},{1},{0}\n")

    if last_score_dx != score_detector.states[1].score:
        last_score_dx=score_detector.states[1].score
        file.write(f"{count},{0},{1}\n")
    
    if last_score_dx==score_detector.states[1].score and last_score_sx==score_detector.states[0].score:
        file.write(f"{count},{0},{0}\n")
    frame=score_detector.write_score(frame)
    output_video.write(frame)
    cv2.imwrite("outputs/frame" + str(count) + ".jpg", score_detector.ball_mask)


video_capture.release()
output_video.release()


    
