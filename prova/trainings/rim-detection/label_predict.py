import os
import cv2
import shutil
from ultralytics import YOLO

def create_dataset(model_path, input_folder, label_output_folder, image_output_folder, conf=0.01):
    # Carica il modello
    model = YOLO(model_path)
    
    # Crea le cartelle di output se non esistono
    os.makedirs(label_output_folder, exist_ok=True)
    os.makedirs(image_output_folder, exist_ok=True)
    
    # Itera attraverso tutte le sottocartelle e immagini nell'input_folder
    for root, _, files in os.walk(input_folder):
        parent_folder_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                
                # Effettua la previsione sull'immagine
                result = model.predict(source=image_path, save=False, conf=0.01)

                # Genera il nome del file concatenando il nome della cartella padre e il nome dell'immagine
                new_file_name = f"{parent_folder_name}_{file}"
                new_file_name_base = os.path.splitext(new_file_name)[0]
                
                # Crea il percorso di output per l'immagine e la label
                label_file_path = os.path.join(label_output_folder, f"{new_file_name_base}.txt")
                output_image_path = os.path.join(image_output_folder, new_file_name)
                
                # Scrive le etichette nel file di testo
                with open(label_file_path, 'w') as f:
                    if not result or len(result[0].boxes) == 0:
                        print(f"No detections for image: {image_path}")
                    else:
                        for box in result[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinate della bounding box
                            conf = box.conf[0].item()  # Confidenza
                            class_id = box.cls[0].item()  # Classe dell'oggetto

                            # Calcola le coordinate normalizzate per YOLO
                            x_center = (x1 + x2) / 2.0 / width
                            y_center = (y1 + y2) / 2.0 / height
                            bbox_width = (x2 - x1) / width
                            bbox_height = (y2 - y1) / height

                            # Scrive nel formato YOLO
                            f.write(f'{int(class_id)} {x_center} {y_center} {bbox_width} {bbox_height}\n')
                
                # Copia l'immagine nella cartella di output con il nuovo nome
                shutil.copy(image_path, output_image_path)



if __name__ == "__main__":
    model_path = '/Users/loren/Desktop/yolo/models/v5.pt'  # Percorso del modello
    input_folder = '/Users/loren/Desktop/output_frames'  # Cartella di input contenente le immagini
    label_output_folder = '/Users/loren/Desktop/inference_labels'  # Cartella di output per le etichette
    image_output_folder = '/Users/loren/Desktop/infererence_images'  # Cartella di output per le immagini

    create_dataset(model_path, input_folder, label_output_folder, image_output_folder)
