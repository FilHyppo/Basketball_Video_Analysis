import cv2
from ultralytics import YOLO

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

class RimDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    

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
                y1 = r * grid_image_height
                y2 = (r + 3) * grid_image_height
                x1 = c * grid_image_width
                x2 = (c + 2) * grid_image_width

                crop_img = image[y1:y2, x1:x2].copy()
                result = self.model.predict(crop_img, save=False, show_labels=False, conf=0.3)

                if len(result[0].boxes) == 0:
                    continue
                else:
                    box = result[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    conf = box.conf[0]  
                    label = f'{conf:.2f}'
                    if conf > 0.4 :
                        if c- prec_col >2:
                            hoops_boxes.append([])
                            hoops_boxes[-1].append([x1+grid_image_width*c, y1+r*grid_image_height, x2+grid_image_width*c, y2+r*grid_image_height, conf])
                            prec_col = c
                        else:
                            hoops_boxes[-1].append([x1+grid_image_width*c, y1+r*grid_image_height, x2+grid_image_width*c, y2+r*grid_image_height,conf])
                            prec_col = c

                if verbose:
                    cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(crop_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imwrite(f'./outputs/rims/rim_out{r}_{c}.jpg', crop_img)

        return get_rims_wavg(hoops_boxes)




