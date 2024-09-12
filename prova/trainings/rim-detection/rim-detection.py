from ultralytics import YOLO
from roboflow import Roboflow
import shutil

# rf = Roboflow(api_key="spKKTXBZ2osRkiflqdVk")
# project = rf.workspace("ball101").project("rim-detection")
# version = project.version(1)
# dataset = version.download("yolov8")



# shutil.move('/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/Rim-Detection-1/train',
#             '/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/datasets/Rim-Detection-1/train'
#             )

# shutil.move('/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/Rim-Detection-1/test',
#              '/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/datasets/Rim-Detection-1/test'
#             )

# shutil.move('/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/Rim-Detection-1/valid',
#              '/work/cvcs2024/Basketball_Video_Analysis/rim-detecetion/datasets/Rim-Detection-1/valid'
#             )


model = YOLO('yolov8n.pt') 
model.train(data='Augmented_Ball_and_rim-1/data.yaml', epochs=300, imgsz=640, batch=16)  