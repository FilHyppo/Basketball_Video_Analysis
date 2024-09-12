import cv2

def draw_bounding_boxes(image_path, txt_path, output_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Leggi il file di testo
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Estrai i valori dal file di testo
        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
        
        # Converti le coordinate normalizzate in coordinate pixel
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Calcola i punti in alto a sinistra e in basso a destra della bounding box
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        # Disegna la bounding box sull'immagine
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Salva l'immagine con le bounding box disegnate
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    image_path = "C:/Users/loren/Desktop/a_im/video_10_frame_8_left_aug_1.jpg"  # Sostituisci con il percorso corretto dell'immagine
    txt_path = "C:/Users/loren/Desktop/a_lab/video_10_frame_8_left_aug_1.txt"  # Sostituisci con il percorso corretto del file di testo
    output_path = "C:/Users/loren/Desktop/im.jpg"  # Sostituisci con il percorso corretto dell'output

    draw_bounding_boxes(image_path, txt_path, output_path)
