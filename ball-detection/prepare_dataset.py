import os
import shutil

# Path del dataset YOLO
dataset_path = "C:\\Users\\marco\\OneDrive\\Desktop\\my-dataset"

# Percorso di destinazione per tutte le immagini e le etichette
dest_images_path = os.path.join(dataset_path, "all_images")
dest_labels_path = os.path.join(dataset_path, "all_labels")

# Crea le cartelle di destinazione se non esistono
os.makedirs(dest_images_path, exist_ok=True)
os.makedirs(dest_labels_path, exist_ok=True)

# Lista delle cartelle da cui spostare (train, valid, test)
splits = ['train', 'valid', 'test']

for split in splits:
    split_images_path = os.path.join(dataset_path, split, 'images')
    split_labels_path = os.path.join(dataset_path, split, 'labels')
    
    # Sposta le immagini
    if os.path.exists(split_images_path):
        for image_file in os.listdir(split_images_path):
            src_image = os.path.join(split_images_path, image_file)
            dest_image = os.path.join(dest_images_path, image_file)
            shutil.move(src_image, dest_image)
    
    # Sposta le etichette
    if os.path.exists(split_labels_path):
        for label_file in os.listdir(split_labels_path):
            src_label = os.path.join(split_labels_path, label_file)
            dest_label = os.path.join(dest_labels_path, label_file)
            shutil.move(src_label, dest_label)

print("Unione completata!")
