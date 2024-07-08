import cv2
import os
import ctypes

# Variabili globali per memorizzare le coordinate del rettangolo
start_point = None
end_point = None
cropping = False
bounding_boxes = []
scale_factor = 1.0

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

def resize_image(image, screen_width, screen_height):
    height, width = image.shape[:2]
    scale_factor = min(screen_width / width, screen_height / height)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image, scale_factor

def process_images_in_folder(image_folder, output_folder, output_folder2):
    global start_point, end_point, cropping, bounding_boxes, scale_factor

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Ottenere la dimensione dello schermo
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    folder_name = os.path.basename(image_folder)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        resized_image, scale_factor = resize_image(image, screen_width, screen_height)
        clone = resized_image.copy()
        
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("image", click_and_crop)

        bounding_boxes = []

        while True:
            display_image = clone.copy()
            for box in bounding_boxes:
                cv2.rectangle(display_image, box[0], box[1], (0, 255, 0), 2)
            if start_point and end_point:
                cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("image", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Reset the cropping region
                start_point, end_point = None, None
                bounding_boxes = []
                clone = resized_image.copy()
            elif key == ord("c"):  # Confirm the cropping regions and move to the next image
                new_image_name = f"{folder_name}_{image_file}"
                cv2.imwrite(os.path.join(output_folder2, new_image_name), image)
                with open(os.path.join(output_folder, f"{os.path.splitext(new_image_name)[0]}.txt"), "w") as f:
                    for box in bounding_boxes:
                        x_min, y_min = box[0]
                        x_max, y_max = box[1]

                        # Convert coordinates back to original image size
                        x_min = int(x_min / scale_factor)
                        y_min = int(y_min / scale_factor)
                        x_max = int(x_max / scale_factor)
                        y_max = int(y_max / scale_factor)

                        # Calcolare le coordinate del centro e la dimensione della bounding box
                        x_center = (x_min + x_max) / 2.0 / width
                        y_center = (y_min + y_max) / 2.0 / height
                        box_width = (x_max - x_min) / width
                        box_height = (y_max - y_min) / height
                        
                        # Scrivere la label nel formato richiesto
                        f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
                break
            elif key == ord("d"):  # Delete the current image and move to the next one
                # Elimina il file immagine corrente
                os.remove(image_path)
                break
            elif key == ord("q"):  # Quit the application
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()

def main(txt_file, output_folder, output_folder2):
    with open(txt_file, 'r') as f:
        folders = f.read().splitlines()

    for folder in folders:
        process_images_in_folder(folder, output_folder, output_folder2)

if __name__ == "__main__":
    txt_file = "C:/Users/loren/Desktop/input_folders.txt"  # Sostituisci con il percorso corretto
    output_folder = "C:/Users/loren/Desktop/yolo/output_labels"  # Sostituisci con il percorso corretto
    output_folder2 = "C:/Users/loren/Desktop/yolo/images"  # Sostituisci con il percorso corretto
    main(txt_file, output_folder, output_folder2)
