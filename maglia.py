import cv2
import numpy as np
import pandas as pd
import supervision as sv
import sys
from sklearn.cluster import KMeans

def load_player_tracking(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    tot_frames = int(lines[-1].split(',')[0])
    tot_frames += 1
    detections_per_frame: list[sv.Detections] = [sv.Detections.empty() for _ in range(tot_frames)]
    for line in lines:
        row = line.strip().split(',')
        frame_number = int(row[0])
        player_id = int(row[1])
        x = float(row[2])
        y = float(row[3])
        width = float(row[4])
        height = float(row[5])
        conf = float(row[6])
        team = int(float(row[7]))

        d = sv.Detections(
            xyxy=np.array([x, y, x + width, y + height]).reshape(-1, 4),
            class_id=np.array([2]),
            confidence=np.array([conf]),
            tracker_id=np.array([player_id]),
            data={"team": [team]},
        )

        detections_per_frame[frame_number] = sv.Detections.merge([detections_per_frame[frame_number], d])

    return detections_per_frame

def apply_background_subtraction(frame, backSub):
    fgMask = backSub.apply(frame)
    return fgMask

def apply_bbox_mask(fgMask, bboxes):
    mask = np.zeros_like(fgMask)
    for bbox in bboxes:
        x, y, x2, y2 = bbox[0].round().astype(int)
        mask[y:y2, x:x2] = fgMask[y:y2, x:x2]
    return mask

def filter_skin_colors(sampled_colors):
    # Converte i colori campionati da RGB a HSV
    # Verifica la forma dell'array sampled_colors
    
   

    sampled_colors_hsv = cv2.cvtColor(sampled_colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
    
    # Intervallo di colori della pelle in HSV
    lower_skin_hsv = np.array([40, 120, 200])
    upper_skin_hsv = np.array([40, 120, 200])
    
    # Crea una maschera per escludere i colori della pelle
    mask_non_skin = cv2.inRange(sampled_colors_hsv, lower_skin_hsv, upper_skin_hsv)
    # print(sampled_colors_hsv.shape)   (1, 1710, 3)
    # print(mask_non_skin.flatten().shape)          (1, 1710)
    # print(sampled_colors.shape)     (1710, 3)
    mask_non_skin_flat = mask_non_skin.flatten()
    # Inverti la maschera per mantenere solo i colori non simili alla pelle
    non_skin_colors = sampled_colors[mask_non_skin_flat== 0]
    
    return non_skin_colors

import numpy as np

def remove_outliers_3d(data, threshold=3.0):
    """
    Rimuove outlier da dati tridimensionali usando la distanza euclidea dalla media.
    
    Args:
        data (np.ndarray): Array di colori 3D (forma N x 3).
        threshold (float): Soglia di distanza per considerare un punto come outlier.
    
    Returns:
        np.ndarray: Array senza outlier.
    """
    # Calcola la media dei dati lungo l'asse 0 (media di R, G e B separatamente)
    mean = np.mean(data, axis=0)
    
    # Calcola la distanza euclidea dalla media
    distances = np.linalg.norm(data - mean, axis=1)  # Norm lungo asse 1 per ogni punto

    # Calcola la deviazione standard delle distanze
    std_dev = np.std(distances)

    # Mantieni solo i punti che hanno una distanza dalla media inferiore alla soglia fissata
    filtered_data = data[distances < threshold * std_dev]
    
    return data


def sample_points_in_bbox(mask, frame, bbox, sample_fraction=0.40):
    x, y, x2, y2 = bbox[0].round().astype(int)

    print("x: ", x, "y: ", y, "x2: ", x2, "y2: ", y2)
    height = y2 - y
    width = x2 - x  
    sample_y_start = y + int(height * (1 - 0.1))
    sample_y_end = y + int(height * sample_fraction)
    sample_x_start = x + int(width * (1 - 0.10))
    sample_x_end = x + int(width * 0.10)
    
    # Estrai la regione di interesse (ROI) sia nella maschera che nel frame
    roi_mask = mask[sample_y_end:sample_y_start, sample_x_end:sample_x_start]
    roi_frame = frame[sample_y_end:sample_y_start, sample_x_end:sample_x_start]
    
    # Trova i punti validi nella maschera (dove fgMask == 255)
    valid_points = np.where(roi_mask == 255)
    
    # Estrai i colori dei pixel validi nel frame originale (frame a colori, quindi 3 canali)
    if len(valid_points[0]) > 0:
        sampled_colors = roi_frame[valid_points[0], valid_points[1], :]
        
        # Disegna i punti campionati sul frame originale
        for point in zip(valid_points[1], valid_points[0]):
            cv2.circle(frame, (x + point[0], y + point[1]), 3, (0, 255, 0), -1)  # Colore verde per i punti

    else:
        sampled_colors = np.array([])

    # Se ci sono colori campionati, filtra i colori della pelle
    if len(sampled_colors) > 0:
        # Filtra i colori simili alla pelle
        non_skin_colors = filter_skin_colors(sampled_colors)
        
        # Rimuovi eventuali outlier dai colori non simili alla pelle
        clean_colors = remove_outliers_3d(non_skin_colors)
    else:
        clean_colors = np.array([])  # Nessun colore valido campionato

    return clean_colors


    return clean_colors
def compute_weighted_mean_color(sampled_colors):
    # Converti i colori campionati da RGB a HSV
    sampled_colors_hsv = cv2.cvtColor(sampled_colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    
    # Estrai il canale della saturazione
    saturation = sampled_colors_hsv[:, 1]
    
    # Aggiungi una piccola costante per evitare pesi nulli
    #print("sat: ", saturation)
    saturation_weight = saturation / (saturation/60 + 1e-5) 
    if saturation_weight.sum() == 0:
        saturation_weight = np.ones_like(saturation)
    
    # Calcola il colore medio pesato usando i pesi derivati dalla saturazione
    weighted_mean_color = np.average(sampled_colors, axis=0, weights=saturation_weight)
    
    return weighted_mean_color.astype(int)



def apply_kmeans(colors, k=2):
    # Assicurati che colors non sia vuoto
    if len(colors) == 0:
        return [], []

    # Converti colors in un array NumPy
    colors_array = np.array(colors)

    # Applicare K-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(colors_array)
    
    # Ottenere i centroidi dei cluster e le etichette
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    return centroids, labels


def process_video_with_bbox(input_video_path, csv_file_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    # Leggi le bounding box dal file CSV
    bboxes_list = load_player_tracking(csv_file_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    detections_color = {}
    colors = {}
    old_centroids = None

    while True:
        ret, frame = cap.read()
        colors.clear()
        if not ret:
            break
        
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        fgMask = apply_background_subtraction(frame, backSub)
        
        if bboxes_list[current_frame_number] is not None:
            bboxes = bboxes_list[current_frame_number]
            masked_fgMask = apply_bbox_mask(fgMask, bboxes)
        else:
            masked_fgMask = np.zeros_like(fgMask)
        
        # Creare una maschera binaria
        binary_mask = np.zeros_like(masked_fgMask)
        binary_mask[masked_fgMask > 150] = 255

        # Processa ogni bounding box
        for i, bbox in enumerate(bboxes):
            sampled_colors = sample_points_in_bbox(binary_mask, frame, bbox)
            
            if len(sampled_colors) > 0:
                mean_color = compute_weighted_mean_color(sampled_colors)
                
                if bboxes.tracker_id[i] not in detections_color:
                    detections_color[bboxes.tracker_id[i]] = []
                detections_color[bboxes.tracker_id[i]].append(mean_color)
                
                colors[i] = mean_color
                
                x, y, x2, y2 = bbox[0].round().astype(int)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x - 20, y - 20), (x, y), mean_color.tolist(), -1)


        # Quando hai finito di elaborare i frame, applica K-means
        centroids, labels = apply_kmeans(list(colors.values()), k=2)
        

        if old_centroids is None:  
            old_centroids = centroids   

        # Definisci i colori per i cluster
        cluster_colors = [
            (0, 0, 255),  # Rosso per il cluster 1
            (255, 0, 0)   # Blu per il cluster 2
        ]

        my_distance1 = ((old_centroids - centroids)*(old_centroids - centroids)).sum(axis=1) 
        my_distance2 = ((old_centroids - centroids[::-1,:])*(old_centroids - centroids[::-1,:])).sum(axis=1) 
        distance_with_old = np.linalg.norm(old_centroids - centroids, axis=1)
        #print("Distance with old: ", distance_with_old)
        #print("My distance: ", my_distance1)
        if my_distance1.sum() < my_distance2.sum():
            pass
        else:
            copy = centroids.copy()
            centroids[0] = copy[1]
            centroids[1] = copy[0]

        old_centroids = centroids
        #print(centroids)
        # Assegna ogni bounding box al cluster più vicino
        for i, bbox in enumerate(bboxes):
            mean_color=None
            if i in colors:
                mean_color= colors[i]
            else:
                continue
    
            distances = np.linalg.norm(centroids - mean_color, axis=1)
            closest_cluster = np.argmin(distances) 
 
            
            
            # Disegna la bounding box con il colore del cluster
            cluster_color = cluster_colors[closest_cluster]
            x, y, x2, y2 = bbox[0].round().astype(int)
            cv2.rectangle(frame, (x, y), (x2, y2), cluster_color, 2)
            #cv2.rectangle(frame, (x - 20, y - 20), (x, y), cluster_color, -1)

        # Scrivi il frame modificato nel video di output
        out.write(frame)
        
        # Mostra il video processato in tempo reale
        cv2.imshow('Masked Background Subtraction with BBox and Mean Color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Percorsi del video di input, file CSV e video di output
input_video_path = 'input_videos/partita_2.mp4'
csv_file_path = 'input_videos/bboxes.csv'
output_video_path = 'background_subtraction_with_bbox.avi'

# Esegui il processamento del video
process_video_with_bbox(input_video_path, csv_file_path, output_video_path)