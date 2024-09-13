import cv2

def save_frame_from_video(video_path, frame_number=1570, output_image_path='frame_1570.jpg'):
    """
    Estrae un frame specifico da un video e lo salva come immagine.

    :param video_path: Percorso del file video.
    :param frame_number: Numero del frame da estrarre (default 1570).
    :param output_image_path: Percorso del file immagine da salvare (default 'frame_1570.jpg').
    """
    # Apri il video
    video = cv2.VideoCapture(video_path)

    # Controlla se il video è stato aperto correttamente
    if not video.isOpened():
        print("Errore nell'aprire il video.")
        return
    
    # Imposta il video sul frame desiderato
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Leggi il frame
    success, frame = video.read()

    # Controlla se il frame è stato letto correttamente
    if success:
        # Salva il frame come immagine
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} salvato come {output_image_path}")
    else:
        print(f"Impossibile leggere il frame {frame_number}")

    # Rilascia il video
    video.release()

def save_image_with_bbox(image_path, bbox, output_image_path='image_with_bbox.jpg'):
    """
    Disegna una bounding box su un'immagine e salva l'immagine risultante.

    :param image_path: Percorso dell'immagine di input.
    :param bbox: Tuple (x, y, w, h) che definisce la bounding box.
    :param output_image_path: Percorso dell'immagine di output (default 'image_with_bbox.jpg').
    """
    # Carica l'immagine
    image = cv2.imread(image_path)
    
    # Controlla se l'immagine è stata caricata correttamente
    if image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Estrai le coordinate della bounding box
    x, y, w, h = bbox
    
    # Disegna la bounding box sull'immagine (rettangolo)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rettangolo verde con spessore 2

    # Salva l'immagine risultante
    cv2.imwrite(output_image_path, image)
    print(f"Immagine con bounding box salvata come {output_image_path}")



def main():
    # Definisci il percorso del video
    video_path = "../input_videos/partita_2.mp4"

    # Estrai e salva il frame 1570 come immagine
    save_frame_from_video(video_path, frame_number=1570, output_image_path='../outputs/frame_1570.jpg')
    image_path = '../outputs/frame_1570.jpg'

    save_image_with_bbox(image_path, (213,674,23,23))

if __name__ == '__main__':
    main()
