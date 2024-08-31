import os
import cv2
import numpy as np

# Variabili globali
start_point = None
end_point = None
drawing = False
ball_added = False
current_image = None

def load_labels(label_path) -> list:
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return labels

def save_labels(label_path, labels) -> None:
    with open(label_path, 'w') as file:
        file.writelines(labels)

def draw_bounding_boxes(image, labels, class_names) -> cv2.UMat:
    h, w = image.shape[:2]
    for label in labels:
        label = label.strip().split()
        class_id = int(label[0])
        cx, cy, bw, bh = map(float, label[1:])
        # Convert YOLO format to bounding box format
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        # Draw the bounding box
        colors = [(0, 255, 0), (0, 255, 255)]
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 1)
        # Draw the class name
        #cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

def display_instructions(window_name, instructions):
    instruction_image = np.zeros((300, 400, 3), dtype=np.uint8)
    y0, dy = 30, 30
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i*dy
        cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(window_name, instruction_image)

def click_event(event, x, y, flags, param):
    global start_point, end_point, drawing, img, ball_added, current_image
    windowName = param['filename']
    color_bounding_box = (0, 255, 255)
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, start_point, end_point, color_bounding_box, 2)
            cv2.imshow(windowName, img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        if start_point == end_point:
            return
        x_min = min(start_point[0], end_point[0])
        y_min = min(start_point[1], end_point[1])
        x_max = max(start_point[0], end_point[0])
        y_max = max(start_point[1], end_point[1])
        start_point = (x_min, y_min)
        end_point = (x_max, y_max)

        drawing = False
        
        # Draw the final bounding box on the image 
        cv2.rectangle(current_image, start_point, end_point, color_bounding_box, 2)
        cv2.imshow(windowName, current_image)

def process_directory(dataset_dir, class_names, ball_class_id):
    global img, labels, label_path, ball_added, current_image, start_point, end_point

    last_image_path = None
    last_image_file = "C:\\Users\\marco\\OneDrive\\Documents\\GitHub\\Basketball_Video_Analysis\\ball-detection\\last_image.txt"
    if os.path.exists(last_image_file):
        with open(last_image_file, "r") as file:
            last_image_path = file.read().strip()

    image_dir = os.path.join(dataset_dir, 'all_images')
    label_dir = os.path.join(dataset_dir, 'all_labels')
    
    # Salto quelle già etichettate
    file_names = [f for f in os.listdir(image_dir) if not f.startswith('video')]

    for filename in file_names:
         
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(image_dir, filename)
            if last_image_path is not None:
                if img_path != last_image_path:
                    continue
                else:
                    last_image_path = None
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            if os.path.exists(label_path):
                labels = load_labels(label_path)
                img_with_bb = draw_bounding_boxes(img.copy(), labels, class_names)
            else:
                labels = []
                img_with_bb = img.copy()
            ball_added = False
            current_image = img_with_bb.copy()
            original_image = current_image.copy()
            start_point, end_point = None, None

            instructions = (
                "Instructions:\n"
                "'n': No New Ball\n"
                "'q': Quit\n"
                "'r': Reset\n"
                "'c': Confirm\n"
                "'a': Remove Rim BB"
            )

            # Display instructions in a separate window
            instruction_window_name = "Instructions"
            display_instructions(instruction_window_name, instructions)
            
            windowName = filename
            cv2.imshow(windowName, current_image)
            cv2.setMouseCallback(windowName, click_event, param={'filename': filename})

            while True:
                key = cv2.waitKey(0)
                
                if key == ord('r'):
                    ball_added = False
                    current_image = original_image.copy()
                    cv2.imshow(windowName, current_image)
                    start_point, end_point = None, None
                    print("Bounding box reset. Draw a new one.")

                elif key == ord('n'):
                    if ball_added:
                        print("Ball selected on image! Reset the bounding box to continue.")
                        continue
                    if not labels and not ball_added:
                        os.remove(img_path)
                        try:
                            os.remove(label_path)
                        except FileNotFoundError:
                            pass
                        print(f"Deleted {filename}, no object found.")
                    else:
                        print("No ball found on image.")
                    break

                elif key == ord('q'):
                    with open(last_image_file, "w") as file:
                        file.write(img_path)
                        print(f"Last image processed: {img_path}")
                    cv2.destroyAllWindows()
                    return  # Exit the program

                elif key == ord('c'):
                    if start_point and end_point:
                        h, w = img.shape[:2]
                        x1, y1 = start_point
                        x2, y2 = end_point
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = abs(x2 - x1) / w
                        bh = abs(y2 - y1) / h
                        label = f"{ball_class_id} {cx} {cy} {bw} {bh}\n"
                        labels.append(label)
                        save_labels(label_path, labels)
                        ball_added = True
                        print("Bounding box confirmed.")
                    break
                elif key == ord('a'): #Se la bounding box del canestro è sbagliata
                    labels = []
                    save_labels(label_path, labels)
                    current_image = draw_bounding_boxes(img.copy(), labels, class_names)
                    cv2.imshow(windowName, current_image)
                    print("Bounding box del canestro eliminata.")
            
            cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_dir = "C:\\Users\\marco\\OneDrive\\Desktop\\my-dataset"  # Directory principale del dataset
    class_names = ['rim', 'ball']  # Lista delle classi. 'basket' è la classe per il canestro, 'ball' per la palla
    ball_class_id = 1  # ID della classe per la palla (l'ordine in class_names)

    process_directory(dataset_dir, class_names, ball_class_id)
