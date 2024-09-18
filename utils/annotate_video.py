import cv2
import sys
import logging
import csv
from paddleocr import PaddleOCR
from contextlib import contextmanager
from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

# Imposta il livello di logging per PaddleOCR e Paddle
logging.getLogger('paddleocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)

def extract_scoreboard_area(frame, scoreboard_coords):
    """Ritaglia la parte del tabellone dal frame e restituisce l'immagine dell'area."""
    x, y, w, h = scoreboard_coords
    scoreboard = frame[y:y+h, x:x+w]
    return scoreboard

def get_score_from_image(image, ocr):
    """Utilizza PaddleOCR per estrarre il testo (punteggi) dall'immagine del tabellone."""
    
    result = ocr.ocr(image, cls=True)
    
    text = ''
    for line in result:
        for word_info in line:
            if len(word_info) > 1 and word_info[1][0].isdigit():
                text += word_info[1][0]
    
    # Converti il testo in numero intero, se possibile
    return int(text) if text.isdigit() else 0

def format_seconds(seconds):
    """Formatta i secondi in ore:minuti:secondi."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def resize_image(image, width, height):
    """Ridimensiona l'immagine alle dimensioni specificate."""
    return cv2.resize(image, (width, height))

def main(video_path, scoreboard_coords_team1, scoreboard_coords_team2, csv_filename):
    # Inizializza PaddleOCR
    ocr = PaddleOCR()

    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    prev_score_team1 = None
    prev_score_team2 = None
    
    frames_to_skip = int(frame_rate)  # Numero di frame da saltare per controllare ogni secondo
    
    # Dimensioni per il ridimensionamento delle finestre
    window_width, window_height = 200, 100

    # Apri il file CSV per la scrittura
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Seconds', 'Team 1', 'Team 2'])  # Intestazione del CSV

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            
            # Controlla solo ogni "frames_to_skip" frame (circa 1 secondo)
            if frame_count % frames_to_skip == 0:
                # Estrai l'area del tabellone per entrambe le squadre
                scoreboard_image_team1 = extract_scoreboard_area(frame, scoreboard_coords_team1)
                scoreboard_image_team2 = extract_scoreboard_area(frame, scoreboard_coords_team2)

                # Estrai il punteggio per entrambe le squadre
                current_score_team1 = get_score_from_image(scoreboard_image_team1, ocr)
                current_score_team2 = get_score_from_image(scoreboard_image_team2, ocr)

                # Confronta il punteggio corrente con quello precedente per entrambe le squadre
                if current_score_team1 != prev_score_team1 and current_score_team1 != 0:
                    timestamp = frame_count // frame_rate  # Tempo in secondi come intero
                    print(f"  Team 1 = {current_score_team1}")
                    prev_score_team1 = current_score_team1

                if current_score_team2 != prev_score_team2 and current_score_team2 != 0:
                    timestamp = frame_count // frame_rate  # Tempo in secondi come intero
                    print(f"  Team 2 = {current_score_team2}")
                    prev_score_team2 = current_score_team2

                # Ridimensiona le immagini dei tabelloni
                resized_image_team1 = resize_image(scoreboard_image_team1, window_width, window_height)
                resized_image_team2 = resize_image(scoreboard_image_team2, window_width, window_height)

                # Mostra le immagini dei tabelloni
                cv2.imshow('Team 1 Scoreboard', resized_image_team1)
                cv2.imshow('Team 2 Scoreboard', resized_image_team2)
                cv2.waitKey(1)

                # Scrivi i dati nel file CSV
                timestamp = frame_count // frame_rate  # Tempo in secondi come intero
                csv_writer.writerow([int(timestamp), current_score_team1, current_score_team2])

                # Stampa il tempo processato in una sola riga
                seconds_processed = frame_count / frame_rate
                formatted_time = format_seconds(seconds_processed)
                sys.stdout.write(f"\rProcessing time: {formatted_time}")
                sys.stdout.flush()

    video_capture.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete.")

if __name__ == "__main__":
    video_path = '../input_videos/partita_2.mp4'
    
    # Coordinate del tabellone per entrambe le squadre nel formato (x, y, width, height)
    scoreboard_coords_team1 = (1460, 194, 57, 47)  # Esempio: Punteggio squadra 1
    scoreboard_coords_team2 = (1725, 194, 57, 47)  # Esempio: Punteggio squadra 2

    # Nome del file CSV
    csv_filename = 'scores.csv'

    main(video_path, scoreboard_coords_team1, scoreboard_coords_team2, csv_filename)
