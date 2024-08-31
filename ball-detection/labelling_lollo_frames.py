import os
import cv2

# Variabili globali
start_point = None
end_point = None
drawing = False
ball_added = False
current_image = None

def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return labels

def save_labels(label_path, labels):
    with open(label_path, 'w') as file:
        file.writelines(labels)

def draw_bounding_boxes(image, labels, class_names):
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
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Draw the class name
        #cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

def click_event(event, x, y, flags, param):
    global start_point, end_point, drawing, img, ball_added, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, start_point, end_point, (255, 0, 0), 2)
            cv2.imshow("Image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        
        # Draw the final bounding box on the image
        cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
        cv2.imshow("Image", img)

def process_directory(dataset_dir, class_names, ball_class_id):
    global img, labels, label_path, ball_added, current_image, start_point, end_point
    

    subdirs = ['train', 'valid', 'test']
    for subdir in subdirs:
        image_dir = os.path.join(dataset_dir, subdir, 'images')
        label_dir = os.path.join(dataset_dir, subdir, 'labels')

        for filename in os.listdir(image_dir):
            #if filename != 'video_0_frame_11_left.jpg':
            #    continue
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

                img = cv2.imread(img_path)
                labels = load_labels(label_path)
                img_with_bb = draw_bounding_boxes(img.copy(), labels, class_names)
                
                ball_added = False
                current_image = img_with_bb.copy()
                start_point, end_point = None, None

                cv2.imshow("Image", img_with_bb)
                cv2.setMouseCallback("Image", click_event)

                print(f"Processing {filename}. Press 'n' if there's no ball, 'q' to quit, or 'r' to reset the bounding box.")
                while True:
                    key = cv2.waitKey(0)
                    
                    if key == ord('r'):
                        # Reset the bounding box
                        current_image = img_with_bb.copy()
                        cv2.imshow("Image", current_image)
                        start_point, end_point = None, None
                        print("Bounding box reset. Draw a new one.")

                    elif key == ord('n'):
                        if not ball_added:
                            print("No ball labeled, moving to next image.")
                        break

                    elif key == ord('q'):
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
                
                cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_dir = 'C:\\Users\\marco\\OneDrive\\Desktop\\my-dataset'  # Directory principale del dataset
    class_names = ['rim', 'ball']  # Lista delle classi. 'basket' è la classe per il canestro, 'ball' per la palla
    ball_class_id = 1  # ID della classe per la palla (l'ordine in class_names)

    process_directory(dataset_dir, class_names, ball_class_id)
