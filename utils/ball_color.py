import cv2
import numpy as np

def get_ball_color_range(bounding_box, frame, region_size=5):
    """
    Calcola la media del colore della palla in HSV, prendendo una piccola regione attorno al centro della bounding box.

    :param bounding_box: Tuple (x, y, w, h) che definisce la bounding box della palla.
    :param frame: Frame da cui estrarre il colore.
    :param region_size: Dimensione del lato della regione quadrata attorno al centro, default 5x5.
    
    :return: Tuple (lower_ball_color, upper_ball_color), i range HSV della palla calcolati come media +/- tolleranza.
    """
    x, y, w, h = bounding_box
    
    # Calcola le coordinate del centro della bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Definisci la metà della regione attorno al centro (per creare un quadrato)
    half_size = region_size // 2

    # Estrai la regione intorno al centro della bounding box
    region_top_left_x = max(center_x - half_size, 0)
    region_top_left_y = max(center_y - half_size, 0)
    region_bottom_right_x = min(center_x + half_size, frame.shape[1])
    region_bottom_right_y = min(center_y + half_size, frame.shape[0])
    
    ball_region = frame[region_top_left_y:region_bottom_right_y, region_top_left_x:region_bottom_right_x]
    
    # Convertire la regione in spazio colore HSV
    ball_region_hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

    # Calcolare la media dei valori per H, S e V nella regione
    h_mean, s_mean, v_mean = np.mean(ball_region_hsv, axis=(0, 1))

    # Definire una tolleranza per i colori attorno alla media (ad esempio ±20 per H, ±40 per S e V)
    h_tol, s_tol, v_tol = 20, 40, 40
    lower_ball_color = (max(h_mean - h_tol, 0), max(s_mean - s_tol, 0), max(v_mean - v_tol, 0))
    upper_ball_color = (min(h_mean + h_tol, 179), min(s_mean + s_tol, 255), min(v_mean + v_tol, 255))
    
    return lower_ball_color, upper_ball_color
