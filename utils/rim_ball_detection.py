import cv2
import numpy as np

import supervision as sv
import math
import time
import os
import argparse

@staticmethod
def get_rims_wavg(hoops_boxes):
    boxes = []
    for i in range(len(hoops_boxes)):
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        conf = 0
        for j in range(len(hoops_boxes[i])):
            x1 += hoops_boxes[i][j][0] * hoops_boxes[i][j][4]*100
            y1 += hoops_boxes[i][j][1] * hoops_boxes[i][j][4]*100
            x2 += hoops_boxes[i][j][2] * hoops_boxes[i][j][4]*100
            y2 += hoops_boxes[i][j][3] * hoops_boxes[i][j][4]*100
            conf += hoops_boxes[i][j][4]*100
        x1 = int(x1//conf)
        y1 = int(y1//conf)
        x2 = int(x2//conf)
        y2 = int(y2//conf)
        boxes.append([(x1, y1), (x2, y2)])
    return boxes


class RimBallDetector:
    def __init__(self, model_path):
        from ultralytics import YOLO
        from torch.cuda import is_available
        self.device = "cuda" if is_available() else "cpu"
        print("Modello caricato su:", self.device)
        self.model = YOLO(model_path).to(self.device)
        self.SIZE = (640, 640)

    def center_crop_frame(self, frame: cv2.UMat) -> cv2.UMat:
        h, w = frame.shape[:2]
        y1, y2 = max(0, int(h // 2 - self.SIZE[0]/2)), int(h // 2 + self.SIZE[0]/2)
        x1, x2 = max(0, int(w // 2 - self.SIZE[1]/2)), int(w // 2 + self.SIZE[1]/2)
        return frame[y1:y2, x1: x2]

    def split_frame(self, frame: cv2.UMat) -> list:
        h, w = frame.shape[:2]

        num = math.ceil(w / 640)

        split_w = w // num
        orig_splits = []
        cropped_splits = []
        for i in range(num):
            split = frame[:, i*split_w:(i+1)*split_w]
            orig_splits.append(split)
            split = self.center_crop_frame(split.copy())
            cropped_splits.append(split)
        
        return orig_splits, cropped_splits

    def predict_frame(self, frame: cv2.UMat, debug=False) -> sv.Detections:
        orig_splits, cropped_splits = self.split_frame(frame)
        
        results = self.model.predict(cropped_splits, save=False) #INFERENCE

        detections_list = []
        for i, (orig_split, cropped_split, result) in enumerate(zip(orig_splits, cropped_splits, results)):
            xyxy = result.boxes.xyxy.cpu().numpy()
            orig_h, orig_w = orig_split.shape[:2]
            offset_h = max(0, (orig_h - self.SIZE[0]) / 2)
            offset_w = max(0, (orig_w - self.SIZE[1]) / 2)
            new_xyxy = []
            for box in xyxy:                    
                new_xyxy.append([
                    box[0] + offset_w + i * orig_w, #x1
                    box[1] + offset_h,              #y1
                    box[2] + offset_w + i * orig_w, #x2
                    box[3] + offset_h               #y2
                    ]
                )
            new_xyxy = np.array(new_xyxy).reshape(-1, 4)
            orig_detections = sv.Detections(
                xyxy=new_xyxy,
                confidence=result.boxes.conf.cpu().numpy(),
                class_id=result.boxes.cls.cpu().numpy().astype(int),
            )
            detections_list.append(orig_detections)
            detections = sv.Detections.merge(detections_list)

        return detections

    def predict(self, input_video_path, output_video_path=None, save=True, save_txt=False, label_file_path=None) -> None:
        """
        Funzione che prende in input un video e crea un video annotato con palle e canestri
        """ 
        print(f"[+] Predizione iniziata con parametri:") 
        print(f"\t- input_video_path:{input_video_path}") 
        print(f"\t- output_video_path:{output_video_path}, ")
        print(f"\t- save:{save},")
        print(f"\t- save_txt:{save_txt},")
        print(f"\t- label_file_path:{label_file_path}")
        
        cap = cv2.VideoCapture(input_video_path)

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if save:
            if output_video_path is None:
                raise Exception("Video path di output non specificato!")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if save_txt:
            if label_file_path is None:
                raise Exception("File di testo non specificato!")
            lines = ["frame_number, class_id, x1, y1, x2, y2, conf\n"]

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.predict_frame(frame)

            if save:
                annotated_frame = self.annotate_frame(frame, detections)
                video_out.write(annotated_frame)
            if save_txt:
                for bbox, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
                    lines.append(f"{frame_number}, {class_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {conf}\n")
            frame_number += 1
            
            print(f"Processato il frame {frame_number}/{tot_frames}")
        
        cap.release()
        
        if save_txt:
            with open(label_file_path, 'w') as f:
                f.writelines(lines)
            print("Fine inferenza, labels file salvato in", label_file_path)
        if save:
            video_out.release()
            print("Fine inferenza, video salvato in", output_video_path)

    def get_hoop_regions(self, input_video_path):
        cap = cv2.VideoCapture(input_video_path)

        _, frame = cap.read()
        cap.release()

        h, w = frame.shape[:2]
        rims = self.detect_rims(frame,verbose=False)
        box_array = np.array([[box[0][0], box[0][1], box[1][0], box[1][1]] for box in rims])
        rim_detections = sv.Detections(xyxy=box_array)
        return rim_detections
    

    def detect_rims(self, image, verbose=False):
        height, width = image.shape[:2]
        grid_rows = 9
        grid_cols = 8
        hoops_boxes =[]
        grid_image_height = height // grid_rows
        grid_image_width = width // grid_cols
        prec_col=-100
        
        for c in range(grid_cols-1):
            for r in range(grid_rows-2):
                im_y1 = r * grid_image_height
                im_y2 = (r + 3) * grid_image_height
                im_x1 = c * grid_image_width
                im_x2 = (c + 2) * grid_image_width
                window_width = grid_image_width * 2
                window_height = grid_image_height * 3

                crop_img = image[im_y1:im_y2, im_x1:im_x2].copy()
                result = self.model.predict(crop_img, save=False, show_labels=False, conf=0.3)

                if len(result[0].boxes.cls == 1) == 0:
                    pass
                else:
                    rim_boxes = result[0].boxes[result[0].boxes.cls == 1]
                    box = rim_boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    conf = box.conf[0]  
                    label = f'{conf:.2f}'
                    if conf > 0.4 :
                        if x1 < window_width/10 or  y1 < window_height/10 or x2 > window_width*9/10 or y2 > window_height*9/10:
                            cv2.putText(crop_img, "skipped", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            if c- prec_col >2:
                                hoops_boxes.append([])
                                hoops_boxes[-1].append([x1+grid_image_width*c, y1+r*grid_image_height, x2+grid_image_width*c, y2+r*grid_image_height, conf])
                                prec_col = c
                            else:
                                hoops_boxes[-1].append([x1+grid_image_width*c, y1+r*grid_image_height, x2+grid_image_width*c, y2+r*grid_image_height,conf])
                                prec_col = c

                   
                    cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(crop_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if verbose:
                    cv2.imwrite(f'./outputs/rims/rim_out{r}_{c}.jpg', crop_img)

        return get_rims_wavg(hoops_boxes)

    

    @staticmethod
    def reconstruct_frame(splits: list) -> cv2.UMat:
        frame = np.hstack(splits)
        return frame
    
    @staticmethod
    def annotate_frame(frame, detections: sv.Detections) -> cv2.UMat:
        class_names = ['ball', 'rim']
        palette = sv.ColorPalette([sv.Color.BLUE, sv.Color.RED])
        bbox_annotator = sv.BoxAnnotator(
            color=palette,
            )
        label_annotator = sv.LabelAnnotator(
            color=palette,
        )
        out = bbox_annotator.annotate(
            frame.copy(),
            detections,
        )
        out = label_annotator.annotate(
            out.copy(),
            detections,
            labels=[class_names[id] + f", {conf:.3f}" for id, conf in zip(detections.class_id, detections.confidence)]
        )

        return out

def annotate_frame_ball_rim(frame, detections: sv.Detections) -> cv2.UMat:
        class_names = ['ball', 'rim', 'player']
        palette = sv.ColorPalette([sv.Color.BLUE, sv.Color.RED, sv.Color.GREEN])
        bbox_annotator = sv.BoxAnnotator(
            color=palette,
            )
        label_annotator = sv.LabelAnnotator(
            color=palette,
            text_scale=0.2,
            color_lookup=sv.ColorLookup.CLASS, # si può fare per trakcking anche
        )
        out = bbox_annotator.annotate(
            frame.copy(),
            detections,
        )
        labels = [class_names[id] + f", {conf:.3f}" for id, conf in zip(detections.class_id, detections.confidence)]
        if detections.tracker_id is not None:
            labels = [label + f", ID:{id}" if id is not None else label for label, id in zip(labels, detections.tracker_id)]
            labels = [label + f", team:{team}" if team is not None else label for label, team in zip(labels, detections.data["team"])]

        out = label_annotator.annotate(
            out.copy(),
            detections,
            labels=labels
        )

        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("-i", "--input_video", required=True)
    args = ap.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    input_video_path: str = args.input_video
    input_video_path = os.path.abspath(input_video_path)

    model_path = '/work/cvcs2024/Basketball_Video_Analysis/repo/Basketball_Video_Analysis/best_weights.pt'
    model = RimBallDetector(model_path)
    
    output_video_path = os.path.join(output_dir, f"ball_rim_video/{os.path.basename(input_video_path).split('.')[0]}.mp4") # Solo per debug
    label_file_path = os.path.join(output_dir, f"ball_rim_labels/{os.path.basename(input_video_path).split('.')[0]}.txt")
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
    model.predict(input_video_path, output_video_path, save=True, save_txt=True, label_file_path=label_file_path)


if __name__ == '__main__':
    main()