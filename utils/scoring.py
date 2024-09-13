import cv2
import numpy as np

class State:                         
    def __init__(self, state_1_patience: int = 10, state_2_patience: int = 10, state_1_consecutive_frames: int = 2):             
        """
        Initializes a new State object with the specified parameters.

        Args:
            state_1_patience (int): Number of frames to wait before considering state 1 concluded.
            state_2_patience (int): Number of frames to wait before considering state 2 concluded.
            state_1_consecutive_frames (int): Number of consecutive frames required to update the score.

        Attributes:
            patience (int): Current count of waiting frames.
            checking (bool): Indicates if we are checking for a new score.
            ball_over (bool): Indicates if the ball is above the rim but not yet inside (state 2).
            consecutive_frames (int): Count of consecutive frames.
            score (int): Score count.
        """   
        self.patience = 0            
        self.checking = False       
        self.ball_over = False
        self.consecutive_frames = 0
        self.score=0

        self.state_1_patience = state_1_patience
        self.state_2_patience = state_2_patience
        self.state_1_consecutive_frames = state_1_consecutive_frames


    def update_state(self, over=False, detected=False):
        if not self.checking and not self.ball_over:     #state 0
            if detected and over:
                self.checking = True
                self.consecutive_frames = 1
                self.ball_over = True
                self.patience = 0
            elif over:
                self.checking = False
                self.consecutive_frames = 0
                self.ball_over = True
                self.patience = 0
            else:
                self.checking = False
                self.consecutive_frames = 0
                self.ball_over = False
                self.patience = 0
            return

        if self.ball_over and self.checking:         #state 1
            if detected:
                self.checking = True
                self.consecutive_frames +=1
                self.ball_over = True
                self.patience = 0
                if self.consecutive_frames == self.state_1_consecutive_frames:          
                    self.score += 1
                    
            else:
                self.checking = True
                #self.consecutive_frames +=1
                self.ball_over = True
                self.patience += 1

            if self.patience >= self.state_1_patience:                          
                self.checking = False
                self.consecutive_frames = 0
                self.ball_over = False
                self.patience = 0
            return
        
        if self.ball_over and not self.checking:             #state 2
            if detected:
                self.checking = True
                self.consecutive_frames =1
                self.ball_over = True
                self.patience = 0
            elif over:
                self.checking = False
                self.consecutive_frames =0
                self.ball_over = True
                self.patience = 0
            else:
                self.checking = False
                self.consecutive_frames =0
                self.ball_over = True
                self.patience +=1

            if self.patience >= self.state_2_patience:                 
                self.checking = False
                self.consecutive_frames = 0
                self.ball_over = False
                self.patience = 0
            return
        
    def write_stats(self, frame, frame_count, right=False):
        offset = 50
        if right:
            offset = frame.shape[1] - 300
        cv2.putText(frame, f"count: {self.consecutive_frames}", (offset,frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"patience: {self.patience}", (offset,frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"over: {self.ball_over}", (offset,frame.shape[0] - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame



class BasketballScoreDetector:
    def __init__(self, *args, **kwargs):  
        if 'base_frame' in kwargs:
            self._init_single_frame_detector(kwargs['base_frame'], kwargs['hoop_regions'], kwargs.get('verbose', False), kwargs.get('threshold', 25))
        elif 'video_path' in kwargs:
            self._init_full_detector(kwargs['video_path'], kwargs['output_path_color'], kwargs.get('hoop_regions', []), kwargs.get('output_path_gray', None), kwargs.get('verbose', False),kwargs.get('output_txt', None),kwargs.get('threshold', 25))
        else:
            raise ValueError("Invalid arguments passed to BasketballHoopDetector")


    def _init_full_detector(self, video_path, output_path_color, hoop_regions, output_path_gray=None, verbose=False,  output_txt=None, threshold=25,):
        self.verbose = verbose
        self.threshold = threshold
        self.video_path = video_path
        self.output_path_color = output_path_color
        self.output_path_gray = output_path_gray
        self.hoop_regions = hoop_regions
        if self.verbose:
            print(f"Initializing detector for video '{video_path}'")
        self.video_capture = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter(output_path_color, self.fourcc, self.video_capture.get(cv2.CAP_PROP_FPS), 
                                            (int(self.video_capture.get(3)), int(self.video_capture.get(4))))
        
        if self.output_path_gray:
            self.output_video_gray = cv2.VideoWriter(output_path_gray, self.fourcc, self.video_capture.get(cv2.CAP_PROP_FPS), 
                                                     (int(self.video_capture.get(3)), int(self.video_capture.get(4))))
        else:
            self.output_video_gray = None
        
        self.states = [State() for _ in self.hoop_regions]
        self.frame_count = 0
        
        self.output_file = None
        if output_txt is not None:
            self.output_file = open(output_txt, 'w')  # Apre il file in modalità scrittura
        self._initialize_base_frame()
        

    def _init_single_frame_detector(self, base_frame, hoop_regions, verbose=False, threshold=25):
        self.verbose = verbose
        self.threshold = threshold
        self.output_video = None
        self.output_video_gray = None
        self.hoop_regions = hoop_regions
        self.frame_count = 0
        self.states = [State() for _ in self.hoop_regions]     
        self.base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        self.base_frame_gray = cv2.GaussianBlur(self.base_frame_gray, (5, 5), 0)
        self.lower_ball_color = None
        self.upper_ball_color = None
        self.ball_mask=None

    def _initialize_base_frame(self):
        ret, base_frame = self.video_capture.read()
        if not ret:
            print("Error: could not read the video file.")
            self._release_resources()
            exit()
        
        self.base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        self.base_frame_gray = cv2.GaussianBlur(self.base_frame_gray, (5, 5), 0)
        #cv2.imwrite('first_frameg.jpg', self.base_frame_gray)
        
        for hoop_top_left, hoop_bottom_right in self.hoop_regions:
            cv2.rectangle(base_frame, hoop_top_left, hoop_bottom_right, (0, 0, 255), 3)
        #cv2.imwrite('first_frame.jpg', base_frame)
        if self.verbose:
            print("First frame saved as 'first_frame.jpg'")

    def draw_colors_on_frame(self,frame, lower_color_hsv, upper_color_hsv):
        # Definisci le dimensioni dei rettangoli
        rect_size = (50, 50)

        # Converti i colori HSV in BGR
        lower_color_bgr = cv2.cvtColor(np.uint8([[lower_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        upper_color_bgr = cv2.cvtColor(np.uint8([[upper_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

        # Disegna il rettangolo per il colore inferiore
        cv2.rectangle(frame, (10, 10), (10 + rect_size[0], 10 + rect_size[1]), lower_color_bgr.tolist(), -1)

        # Disegna il rettangolo per il colore superiore
        cv2.rectangle(frame, (70, 10), (70 + rect_size[0], 10 + rect_size[1]), upper_color_bgr.tolist(), -1)



    def write_score(self, frame):
        for idx in range(len(self.hoop_regions)):
            if len(self.hoop_regions) == 1:
                text_position = (50, 50)  
            elif idx == 0:
                text_position = (50, 50)  
            else:
                text_position = (frame.shape[1] -200, 50)
            cv2.putText(frame, f"Baskets: {self.states[idx].score}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)        
        frame_diff = cv2.absdiff(frame_gray, self.base_frame_gray)
        _, binary_diff = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Aggiungi conversione da BGR a HSV per il controllo colore
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Crea la maschera per individuare i colori simili alla palla
        ball_mask = None
        if self.lower_ball_color is not None and self.upper_ball_color is not None:
            ball_mask = cv2.inRange(frame_hsv, self.lower_ball_color, self.upper_ball_color)
            self.draw_colors_on_frame(frame, self.lower_ball_color, self.upper_ball_color)
            self.ball_mask=ball_mask


        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_diff, connectivity=8)
        
        for hoop_top_left, hoop_bottom_right in self.hoop_regions:
            cv2.rectangle(frame, hoop_top_left, hoop_bottom_right, (0, 0, 255), 3)
        
        founds=[False for _ in self.hoop_regions]
        at_least_one_detected=[False for _ in self.hoop_regions]
        for i in range(1, num_labels):  # Skip the background label (0)
            over=False
            detected=False
            x, y, w, h, area = stats[i]
            if area > 5 and area < 200:  # Filter out small components
                for idx, (hoop_top_left, hoop_bottom_right) in enumerate(self.hoop_regions): 
                    over=False
                    detected=False
                    if founds[idx]:
                        continue
                    #cv2.rectangle(binary_diff_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    W=hoop_bottom_right[0]-hoop_top_left[0]
                    H=hoop_bottom_right[1]-hoop_top_left[1]
                    cv2.rectangle(frame, (round(hoop_top_left[0] - 0.5*W), hoop_top_left[1]-round(1*H)), (round(hoop_bottom_right[0] + 0.5*W), hoop_top_left[1]), (255, 0, 0), 2)
                    if(hoop_top_left[1]-round(1*H) < y < hoop_top_left[1] and hoop_top_left[0] - 0.5*W < x < hoop_bottom_right[0] + 0.5*W):     #need to to hit the top of the hoop first
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Controlla se l'area individuata ha il colore della palla
                        if ball_mask is not None:
                            object_region = ball_mask[y:y+h, x:x+w]
                            ball_pixels = cv2.countNonZero(object_region)
                            object_area = w * h
                    
                            # Se una parte significativa dell'oggetto corrisponde alla palla, lo consideriamo come la palla
                            if ball_pixels / object_area < 0.5:  # Almeno il 50% dell'area deve essere del colore della palla
                                cv2.putText(frame, "No Ball", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                continue
                        
                        over=True
                        founds[idx]=True
                    
                    if (hoop_top_left[0] < x < hoop_bottom_right[0] and 
                        hoop_top_left[1] < y < hoop_bottom_right[1]):
                        founds[idx]=True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        if self.verbose:
                            print(f"Frame {self.frame_count}: Detected moving object in the hoop area")
                        detected=True

                    if founds[idx]:
                        self.states[idx].update_state(over, detected)
                        at_least_one_detected[idx]=True

        for idx in range(len(self.hoop_regions)):
            if not at_least_one_detected[idx]:
                self.states[idx].update_state(False, False)

        if self.verbose:
            self.states[0].write_stats(frame, self.frame_count)
            self.states[1].write_stats(frame, self.frame_count, right=True)
        
        if self.output_video_gray:
            binary_diff_color = cv2.cvtColor(binary_diff, cv2.COLOR_GRAY2BGR)
            for hoop_top_left, hoop_bottom_right in self.hoop_regions:
                cv2.rectangle(binary_diff_color, hoop_top_left, hoop_bottom_right, (0, 0, 255), 3)
            binary_diff_color=self.write_score(binary_diff_color)
            self.output_video_gray.write(binary_diff_color)

        cv2.putText(frame, f"framecount: {self.frame_count}", (round(frame.shape[1]/2) - 200,frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if self.output_video:
            frame=self.write_score(frame)
            self.output_video.write(frame)
        #cv2.imwrite('./outputs/frame' + str(self.frame_count) + ".jpg", frame)      #save single frames
        #cv2.imwrite('./outputs/frame' + str(self.frame_count) + ".jpg", binary_diff_color)      #save single frames
        
        self.frame_count += 1

        if self.verbose:
            print(self.frame_count)

    def _release_resources(self):
        self.video_capture.release()
        self.output_video.release()
        if self.output_video_gray:
            self.output_video_gray.release()
        if self.output_file:
            self.output_file.close()
        cv2.destroyAllWindows()

    def run(self):
        frame_count = 0
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            self.process_frame(frame)
            if self.output_file:
                self.output_file.write(f"{frame_count},{self.states[0].score},{self.states[1].score}\n")

            frame_count+=1
        
        self._release_resources()


