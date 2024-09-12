import os
import cv2

# Variabili globali per memorizzare le coordinate del rettangolo
start_point = None
end_point = None
cropping = False
bounding_boxes = []

def click_and_crop(event, x, y, flags, param):
    global start_point, end_point, cropping, bounding_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False
        bounding_boxes.append((start_point, end_point))
        start_point = None
        end_point = None

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, box[0], box[1], (0, 255, 0), 2)
    return image

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

            boxes.append(((x1, y1), (x2, y2)))
    return boxes

def write_labels(label_path, boxes, img_width, img_height):
    with open(label_path, 'w') as f:
        for box in boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]

            # Calcola le coordinate normalizzate per YOLO
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            bbox_width = (x2 - x1) / img_width
            bbox_height = (y2 - y1) / img_height

            # Scrive nel formato YOLO
            f.write(f'0 {x_center} {y_center} {bbox_width} {bbox_height}\n')

def process_images_and_labels(image_folder, label_folder, filter_file):
    global start_point, end_point, cropping, bounding_boxes

    # Leggi le stringhe di filtro dal file
    with open(filter_file, 'r') as f:
        filters = [line.strip() for line in f.readlines()]

    # Ottieni la lista dei file immagine che iniziano con una delle stringhe di filtro
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and any(f.startswith(prefix) for prefix in filters)]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Leggi le etichette esistenti
        if os.path.exists(label_path):
            bounding_boxes = read_labels(label_path, width, height)
        else:
            bounding_boxes = []

        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        while True:
            display_image = clone.copy()
            draw_bounding_boxes(display_image, bounding_boxes)
            if start_point and end_point:
                cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(display_image, image_file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("image", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Reset the cropping region
                start_point, end_point = None, None
                bounding_boxes = []
                clone = image.copy()
            elif key == ord("c"):  # Confirm the cropping regions and move to the next image
                write_labels(label_path, bounding_boxes, width, height)
                break
            elif key == ord("d"):  # Delete the current image and label
                os.remove(image_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                break
            elif key == ord("q"):  # Quit the application
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder = '/Users/loren/Desktop/infererence_images'  # Cartella di input contenente le immagini
    label_folder = 'C:/Users/loren/Desktop/inference_labels'  # Cartella di input contenente le etichette
    filter_file = 'C:/Users/loren/Desktop/filter_file.txt'  # File di testo contenente le stringhe di filtro

    process_images_and_labels(image_folder, label_folder, filter_file)
