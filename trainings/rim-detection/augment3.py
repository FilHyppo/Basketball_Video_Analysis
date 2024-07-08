import os
import cv2
import numpy as np
import albumentations as A

def read_labels(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Calcola le coordinate della bounding box
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            if x1 < x2 and y1 < y2:  # Verifica se le coordinate sono valide
                boxes.append([x1, y1, x2, y2, class_id])
    return boxes

def write_labels(label_path, boxes, img_width, img_height):
    with open(label_path, 'w') as f:
        for box in boxes:
            x1, y1 = box[:2]
            x2, y2 = box[2:4]
            class_id = box[4]

            # Calcola le coordinate normalizzate per YOLO
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            bbox_width = (x2 - x1) / img_width
            bbox_height = (y2 - y1) / img_height

            # Scrive nel formato YOLO
            f.write(f'{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n')

def create_augmented_dataset(image_folder, label_folder, output_image_folder, output_label_folder, num_augmentations=3):
    transform = A.Compose([
        A.RandomSizedBBoxSafeCrop(height=300, width=300, p=1.0),
        A.Resize(640, 640)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        if os.path.exists(label_path):
            boxes = read_labels(label_path, width, height)
        else:
            boxes = []

        # Salva l'immagine originale
        output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_orig.jpg")
        output_label_path = os.path.join(output_label_folder, f"{os.path.splitext(image_file)[0]}_orig.txt")
        cv2.imwrite(output_image_path, image)
        write_labels(output_label_path, boxes, width, height)

        for i in range(num_augmentations):
            attempts = 0
            while attempts < 10:  # Limita il numero di tentativi a 10
                transformed = transform(image=image, bboxes=boxes, class_labels=[box[4] for box in boxes])
                transformed_image = transformed['image']
                transformed_boxes = transformed['bboxes']
                
                # Filtra le bounding box non valide
                valid_boxes = []
                for box in transformed_boxes:
                    x_min, y_min, x_max, y_max = box[:4]
                    if x_max > x_min and y_max > y_min:
                        valid_boxes.append(box)
                
                # Verifica che ci sia almeno una bounding box valida nell'immagine croppata
                if len(valid_boxes) > 0:
                    break
                attempts += 1

            if len(valid_boxes) == 0:
                print(f"Could not create a valid augmentation for {image_file} after 10 attempts")
                continue

            output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
            output_label_path = os.path.join(output_label_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.txt")

            cv2.imwrite(output_image_path, transformed_image)
            write_labels(output_label_path, valid_boxes, 640, 640)



if __name__ == "__main__":
    image_folder = 'C:/Users/loren/Desktop/yolo/images'  # Cartella contenente le immagini originali
    label_folder = 'C:/Users/loren/Desktop/yolo/output_labels'  # Cartella contenente le etichette originali
    output_image_folder = 'C:/Users/loren/Desktop/a_im'  # Cartella di output per le immagini augmentate
    output_label_folder = 'C:/Users/loren/Desktop/a_lab'  # Cartella di output per le etichette augmentate

    create_augmented_dataset(image_folder, label_folder, output_image_folder, output_label_folder, num_augmentations=3)
