from utils.scoring import BasketballHoopDetector
import cv2


hoop_regions = [
    ((205, 316), (227, 331)),
    ((1052,324),(1066,347))
]

detector = BasketballHoopDetector(
    video_path='input_videos/example.mp4',
    output_path_color='outputs/example.avi',
    hoop_regions=hoop_regions,
    output_path_gray='outputs/example_gray.avi',  # Set to None if you don't want a gray video
    verbose=True
)
detector.run()

# video_capture = cv2.VideoCapture('input_videos/example.mp4')
# output_video=cv2.VideoWriter('outputs/example.avi', cv2.VideoWriter_fourcc(*'XVID'), video_capture.get(cv2.CAP_PROP_FPS), 
#                                             (int(video_capture.get(3)), int(video_capture.get(4))))

# _, base_frame = video_capture.read()
# detector_frame = BasketballHoopDetector(
#     base_frame=base_frame,
#     hoop_regions=hoop_regions,
#     verbose=True
# )


# prec_score=None
# while video_capture.isOpened():
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#     detector_frame.process_frame(frame)
#     if prec_score != detector_frame.states[0].score:
#         print(detector_frame.states[0].score)
    
#     prec_score=detector_frame.states[0].score


#     frame=detector_frame.write_score(frame)
#     output_video.write(frame)


# video_capture.release()
# output_video.release()


    
