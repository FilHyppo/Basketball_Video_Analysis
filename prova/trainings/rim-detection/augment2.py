import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Definisci le trasformazioni di data augmentation focalizzate su rumore e risoluzione
transform = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Aggiunge rumore gaussiano
    A.Downscale(scale_min=0.1, scale_max=0.9, p=0.5),  # Riduce e poi riporta la risoluzione
    A.OneOf([
        A.MotionBlur(p=0.5),  # Aggiunge sfocatura di movimento
        A.MedianBlur(blur_limit=5, p=0.5),  # Aggiunge sfocatura mediana
    ], p=0.5),
    A.Resize(width=640, height=480),  # Mantiene le dimensioni costanti
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Path to the main directory
main_directory = './datasets/Ball_and_rim-1_copy'

def augment_dataset(directory, output_dir, num_augmentations=5):
    for subset in ['train', 'test', 'valid']:
        image_dir = os.path.join(directory, subset, 'images')
        label_dir = os.path.join(directory, subset, 'labels')
        aug_image_dir = os.path.join(output_dir, subset, 'images')
        aug_label_dir = os.path.join(output_dir, subset, 'labels')

        if not os.path.exists(aug_image_dir):
            os.makedirs(aug_image_dir)
        if not os.path.exists(aug_label_dir):
            os.makedirs(aug_label_dir)

        sad=0
        for image_file in os.listdir(image_dir):
            sad+=1
            print(sad)
            if image_file.endswith(".jpg"):
                image_path = os.path.join(image_dir, image_file)
                label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

                # Load image and annotations
                image = cv2.imread(image_path)
                h, w, _ = image.shape

                with open(label_path, 'r') as file:
                    labels = file.readlines()

                class_labels = []
                bboxes = []
                for label in labels:
                    parts = label.strip().split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(class_id)

                for i in range(num_augmentations):
                    # Apply augmentations
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']

                    # Save augmented image
                    aug_image_file = f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg"
                    aug_image_path = os.path.join(aug_image_dir, aug_image_file)
                    aug_image_np = aug_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    cv2.imwrite(aug_image_path, aug_image_np)

                    # Save augmented labels
                    aug_label_file = f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"
                    aug_label_path = os.path.join(aug_label_dir, aug_label_file)
                    with open(aug_label_path, 'w') as file:
                        for bbox, class_label in zip(aug_bboxes, aug_class_labels):
                            bbox_str = ' '.join(map(str, bbox))
                            file.write(f'{class_label} {bbox_str}\n')

augment_dataset(main_directory, '/path/to/Augmented_Ball_and_rim-1', num_augmentations=5)
