import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

# Carica il video
video_path = "C:\\Users\\marco\\OneDrive\\Documents\\GitHub\\Basketball_Video_Analysis\\input_videos\\tosti_zola.mp4"
cap = cv2.VideoCapture(video_path)

# Leggi il primo frame
ret, previous_frame = cap.read()
previous_frame = imutils.resize(previous_frame, width=600)
result = previous_frame
first = previous_frame

for _ in range(200):
    # Leggi il frame successivo
    ret, current_frame = cap.read()
    if not ret:
        break

    # Ridimensiona il frame
    current_frame = imutils.resize(current_frame, width=600)

    # Trova punti chiave e descrittori
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(previous_frame, None)
    kp2, des2 = orb.detectAndCompute(current_frame, None)

    # Corrispondenza tra descrittori
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Estrai le coordinate dai punti corrispondenti
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Trova l'omografia
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Applica la trasformazione omografica
    height, width, _ = current_frame.shape
    panorama = cv2.warpPerspective(current_frame, H, (width + 100, height))
    
    # Combina le immagini
    panorama[0:result.shape[0], 0:result.shape[1]] = result

    # Aggiorna il risultato
    result = panorama

    # Aggiorna il frame precedente
    previous_frame = current_frame

# Chiudi il video
cap.release()

cv2.imshow("First frame", imutils.resize(first, width=600))

# Visualizza il panorama
cv2.imshow("Result", result)
cv2.waitKey(0)
