import cv2
import numpy as np
import os

NUM_CORNERS = 15 #Numero di angoli presi in considerazione per le omografie del campo

def corner_2_tuple(corner):
    return (corner['x'], corner['y'])

def feet(n, foot, inches=0):
    return int(n * foot + inches * foot / 12)

def corner_pos(id: str, w, h):
    """
    Funzione che ritorna la poszione dei corner del court plane
    """
    foot1 = h / 50
    foot2 = w / 92.6 ################################# 94
    foot = (foot1 + foot2) / 2

    if id == 'P0':
        return 0, 0
    if id == 'P1':
        return 0, int(h * 0.5) - feet(8, foot)
    if id == 'P2':
        return 0, int(h * 0.5) + feet(8, foot)
    if id == 'P3':
        return 0, h
    if id == 'P4':
        return 0 + feet(19, foot), int(h * 0.5) - feet(8, foot)
    if id == 'P5':
        return 0 + feet(19, foot), int(h * 0.5) +  feet(8,foot)
    if id == 'P6':
        return w // 2, 0
    if id == 'P7':
        return w // 2, h
    if id == 'P8':
        return w, 0
    if id == 'P9':
        return w, int(h * 0.5) - feet(8, foot)
    if id == 'P10':
        return w, int(h * 0.5) + feet(8, foot)
    if id == 'P11':
        return w, h
    if id == 'P12':
        return w - feet(19, foot), int(h * 0.5) - feet(8, foot)
    if id == 'P13':
        return w - feet(19, foot), int(h * 0.5) + feet(8, foot)
    if id == 'P14':
        return w // 2, h // 2
    if id == 'rim_left':
        return feet(4, foot), int(h * 0.5)
    if id == 'rim_right':
        return (w - feet(4, foot), int(h * 0.5))
    return 0, 0

def top_view(frame, corners, inverse=False, new_h=None, new_w=None):
    """
    Funzione che ritorna la vista dall'alto del campo    
    """
    h, w = frame.shape[:2]
    M, status = find_homography_matrix(frame, corners, inverse)
    if new_h is None:
        new_h = h
    if new_w is None:
        new_w = w
    frame_top_view = cv2.warpPerspective(frame, M, (new_w, new_h))

    return frame_top_view


def mask_court(frame, corners):
    """
    Funzione che ritorna maschera binaria del campo
    """
    court = np.array([[corners['P0']['x'], corners['P0']['y']], [corners['P3']['x'], corners['P3']['y']], [corners['P11']['x'], corners['P11']['y']], [corners['P8']['x'], corners['P8']['y']], [corners['P0']['x'], corners['P0']['y']]], dtype=np.int32)
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [court], 255)
    return mask

def mask_3_point_line(frame, corners):
    """
    Funzione per calcolar una maschera binaria che rappresenta tutti i punti dentro la linea da 3 punti
    """
    h, w = frame.shape[:2]

    foot1 = h / 50
    foot2 = w / 94
    foot = (foot1 + foot2) / 2


    mask = np.zeros((h, w), dtype=np.uint8)
    
    rim_left = [(feet(4, foot), int(h * 0.5))]
    rim_right = [(w - feet(4, foot), int(h * 0.5))]

    three_point_line_left = [(0, int(h - feet(3, foot))),
                             (feet(13, foot), int(h - feet(3, foot))),
                             (0, feet(3, foot)),
                             (feet(13, foot), feet(3, foot))
    ]
    #cv2.line(frame, three_point_line_left[0], three_point_line_left[1], (0, 255, 0), 2)
    #cv2.line(frame, three_point_line_left[2], three_point_line_left[3], (0, 255, 0), 2)
    cv2.rectangle(mask, three_point_line_left[0], three_point_line_left[3], 255, -1)
    cv2.ellipse(mask, rim_left[0], (feet(23, foot, 9), feet(23, foot, 9)), 0, -72, 70, 255, -1)


    three_point_line_right = [(w, int(h - feet(3, foot))),
                        (w - feet(13, foot), int(h - feet(3, foot))),
                        (w, feet(3, foot)),
                        (w - feet(13, foot), feet(3, foot))
    ]
    #cv2.line(frame, three_point_line_right[0], three_point_line_right[1], (0, 255, 0), 2)
    #cv2.line(frame, three_point_line_right[2], three_point_line_right[3], (0, 255, 0), 2)
    cv2.rectangle(mask, three_point_line_right[0], three_point_line_right[3], 255, -1)
    cv2.ellipse(mask, rim_right[0], (feet(23, foot, 9), feet(23, foot, 9)), 0, 110, 250, 255, -1)

    mask = top_view(mask, corners, inverse=True)

    return mask # == 255

def find_homography_matrix(frame, corners, inverse=False):
    """
    Funzione per calcolare la matrice di omografia dal camera plane al court plane
    """
    h, w = frame.shape[:2]
    
    src = []
    dst = []

    for id, corner in corners.items():
        if inverse:
            src.append([corner_pos(id, w, h)])
            dst.append([corner['x'], corner['y']])
        else:
            src.append([corner['x'], corner['y']])
            dst.append([corner_pos(id, w, h)])

    src = np.array(src, dtype=np.float32)
    src = src.reshape(-1, 1, 2)

    dst = np.array(dst, dtype=np.float32)
    dst = dst.reshape(-1, 1, 2)

    M, status = cv2.findHomography(src, dst)

    return M, status


def find_missing_corners(frame, corners):
    """
    Funzione per calcolare gli angoli non inseriti dall'utente (Possono avere cooridnate negative o maggiori di altezza/larghezza)
    """
    M, status = find_homography_matrix(frame, corners, inverse=True)

    ids = [f'P{i}' for i in range(NUM_CORNERS)]
    new_corners = corners

    for id in ids:
        if id not in corners:
            corner = corner_pos(id, frame.shape[1], frame.shape[0])
            corner = cv2.perspectiveTransform(np.array([[corner]], dtype=np.float32), M)[0][0]
            corner = np.round(corner)
            new_corners[id] = {'x': int(corner[0]), 'y': int(corner[1])}
        

    return new_corners


def enhance_corners(frame, corners):

    """
    Funzione per cercare di correggere leggermente l'input dell'utente sugli angoli
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray,15,30,apertureSize=3)


    # Apply HoughLinesP method to 
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=150, # Min allowed length of line
                maxLineGap=20 # Max allowed gap between line for joining them
                )
    candidates = {}
    ids = list(corners.keys())
    for id in ids:
        candidates[id] = [[corners[id]['x'], corners[id]['y']]]
    img = frame.copy()
    # Iterate over points
    if lines is None:
        return corners
    for points in lines:
          # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        #keep only horizontal or vertical lines
        dx = abs(x2 - x1)
        dy = abs(y2 -y1)
        if dx != 0:
            m = dy / dx
            if 0.1 < m < 10:
                continue
        # Draw line over the frame
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        for id, corner in corners.items():
            if id == 'P14':
                continue
            x, y = corner['x'], corner['y']
            # Check if point is close to the corner
            d = (x - x1)**2 + (y - y1)**2
            if np.sqrt(d) < 20:
                candidates[id].append([int(x1), int(y1)])
    
    new_corners = corners.copy()

    for id, cand in candidates.items():
        if len(cand) < 2:
            continue
        print(id, ": ", cand)
        sum_x, sum_y = 0, 0
        cnt = 0
        for x, y in cand:
            sum_x += x
            sum_y += y
            cnt += 1
        new_corners[id]['x'] = sum_x // cnt
        new_corners[id]['y'] = sum_y // cnt
       

    return new_corners

################################## DISTORTION ############################
def undistort_frame(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistort_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    min_x, min_y, max_x, max_y = roi
    return undistort_frame

def camera_matrix(distortion_parameters: dict):
    return np.array([
        [float(distortion_parameters['fx']), 0, float(distortion_parameters['cx'])],
        [0, float(distortion_parameters['fy']), float(distortion_parameters['cy'])],
        [0, 0, 1]
    ])
def dist_coeffs(distortion_parameters: dict):
    return np.array([
        float(distortion_parameters['k1']),
        float(distortion_parameters['k2']),
        float(distortion_parameters['k3']),
        float(distortion_parameters['k4'])
    ])



############################################## COURT DRAWINGS #################################################

def draw_points(frame, corners: dict):
    for id, corner in corners.items():
        cv2.circle(frame, (corner['x'], corner['y']), 3, (0, 255, 0), -1)
        cv2.putText(frame, id, (corner['x'], corner['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def feet(n, foot, inches=0):
    return int(n * foot + inches * foot / 12)

def draw_court_lines(frame):
    h, w = frame.shape[:2]

    foot1 = h / 50
    foot2 = w / 94
    foot = (foot1 + foot2) / 2

    inside_area_left = [(0, int(h * 0.5) + feet(8, foot)), 
                        (0 + feet(19, foot), int(h * 0.5) + feet(8, foot)), 
                        (0, int(h * 0.5) - feet(8, foot)),
                        (0 + feet(19, foot), int(h * 0.5) - feet(8,foot)),
    ]
    cv2.rectangle(frame, inside_area_left[0], inside_area_left[3], (0, 255, 0), 2)
    inside_area_right = [(w - feet(19, foot), int(h * 0.5) + feet(8, foot)), 
                        (w, int(h * 0.5) + feet(8, foot)), 
                        (w - feet(19, foot), int(h * 0.5) - feet(8,foot)),
                        (w, int(h * 0.5) - feet(8, foot)),
    ]
    cv2.rectangle(frame, inside_area_right[0], inside_area_right[3], (0, 255, 0), 2)

    middle_line = [(w // 2, 0), (w // 2, h)]
    center = [(w // 2, h // 2)]
    cv2.line(frame, middle_line[0], middle_line[1], (0, 255, 0), 2)
    cv2.circle(frame, center[0], feet(10, foot), (0, 255, 0), 2)

    rim_left = [(feet(4, foot), int(h * 0.5))]
    cv2.circle(frame, rim_left[0], 10, (255, 255, 0), -1)

    three_point_line_left = [(0, int(h - feet(3, foot))),
                             (feet(13, foot), int(h - feet(3, foot))),
                             (0, feet(3, foot)),
                             (feet(13, foot), feet(3, foot))
    ]
    cv2.line(frame, three_point_line_left[0], three_point_line_left[1], (0, 255, 0), 2)
    cv2.line(frame, three_point_line_left[2], three_point_line_left[3], (0, 255, 0), 2)
    cv2.ellipse(frame, rim_left[0], (feet(23, foot, 9), feet(23, foot, 9)), 0, -72, 70, (0, 255, 0), 2)

    rim_right = [(w - feet(4, foot), int(h * 0.5))]
    cv2.circle(frame, rim_right[0], 10, (255, 255, 0), -1)

    three_point_line_right = [(w, int(h - feet(3, foot))),
                            (w - feet(13, foot), int(h - feet(3, foot))),
                            (w, feet(3, foot)),
                            (w - feet(13, foot), feet(3, foot))
        ]
    cv2.line(frame, three_point_line_right[0], three_point_line_right[1], (0, 255, 0), 2)
    cv2.line(frame, three_point_line_right[2], three_point_line_right[3], (0, 255, 0), 2)
    cv2.ellipse(frame, rim_right[0], (feet(23, foot, 9), feet(23, foot, 9)), 0, 110, 250, (0, 255, 0), 2)

def calcola_intersezioni(p1, p2, larghezza, altezza):
    x1, y1 = p1
    x2, y2 = p2
    
    # Calcolare i coefficienti della retta
    dx = x2 - x1
    dy = y2 - y1
    
    punti = []
    
    # Intersezione con il lato superiore (y = 0)
    if dy != 0:
        t = -y1 / dy
        x = int(x1 + t * dx)
        if 0 <= x <= larghezza:
            punti.append((x, 0))
    
    # Intersezione con il lato inferiore (y = altezza)
    if dy != 0:
        t = (altezza - y1) / dy
        x = int(x1 + t * dx)
        if 0 <= x <= larghezza:
            punti.append((x, altezza))
    
    # Intersezione con il lato sinistro (x = 0)
    if dx != 0:
        t = -x1 / dx
        y = int(y1 + t * dy)
        if 0 <= y <= altezza:
            punti.append((0, y))
    
    # Intersezione con il lato destro (x = larghezza)
    if dx != 0:
        t = (larghezza - x1) / dx
        y = int(y1 + t * dy)
        if 0 <= y <= altezza:
            punti.append((larghezza, y))
    

    return punti[0], punti[1]

def draw_lines(frame, corners: dict, color=(0, 255, 255), thickness=4):
    
    #TOP SIDELINE
    if 'P0' in corners.keys() and 'P8' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P0']['x'], corners['P0']['y']), (corners['P8']['x'], corners['P8']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P0' in corners.keys() and 'P6' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P0']['x'], corners['P0']['y']), (corners['P6']['x'], corners['P6']['y']), frame.shape[1], frame.shape[0])
    elif 'P8' in corners.keys() and 'P6' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P8']['x'], corners['P8']['y']), (corners['P6']['x'], corners['P6']['y']), frame.shape[1], frame.shape[0])
    #LEFT BASELINE
    if 'P0' in corners.keys() and 'P3' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P0']['x'], corners['P0']['y']), (corners['P3']['x'], corners['P3']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P0' in corners.keys() and 'P2' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P0']['x'], corners['P0']['y']), (corners['P2']['x'], corners['P2']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P0' in corners.keys() and 'P1' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P0']['x'], corners['P0']['y']), (corners['P1']['x'], corners['P1']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P1' in corners.keys() and 'P2' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P1']['x'], corners['P1']['y']), (corners['P2']['x'], corners['P2']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P2' in corners.keys() and 'P3' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P2']['x'], corners['P2']['y']), (corners['P3']['x'], corners['P3']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P1' in corners.keys() and 'P3' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P1']['x'], corners['P1']['y']), (corners['P3']['x'], corners['P3']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    #BOTTOM SIDELINE
    if 'P11' in corners.keys() and 'P3' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P11']['x'], corners['P11']['y']), (corners['P3']['x'], corners['P3']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P11' in corners.keys() and 'P7' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P11']['x'], corners['P11']['y']), (corners['P7']['x'], corners['P7']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P3' in corners.keys() and 'P7' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P3']['x'], corners['P3']['y']), (corners['P7']['x'], corners['P7']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    #RIGHT BASELINE
    if 'P11' in corners.keys() and 'P8' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P11']['x'], corners['P11']['y']), (corners['P8']['x'], corners['P8']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P11' in corners.keys() and 'P10' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P11']['x'], corners['P11']['y']), (corners['P10']['x'], corners['P10']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P11' in corners.keys() and 'P9' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P11']['x'], corners['P11']['y']), (corners['P9']['x'], corners['P9']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P9' in corners.keys() and 'P10' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P9']['x'], corners['P9']['y']), (corners['P10']['x'], corners['P10']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P10' in corners.keys() and 'P8' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P10']['x'], corners['P10']['y']), (corners['P8']['x'], corners['P8']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    elif 'P9' in corners.keys() and 'P8' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P9']['x'], corners['P9']['y']), (corners['P8']['x'], corners['P8']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    #FREE-THROW LINES
    if 'P1' in corners.keys() and 'P4' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P1']['x'], corners['P1']['y']), (corners['P4']['x'], corners['P4']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    if 'P2' in corners.keys() and 'P5' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P2']['x'], corners['P2']['y']), (corners['P5']['x'], corners['P5']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    if 'P4' in corners.keys() and 'P5' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P4']['x'], corners['P4']['y']), (corners['P5']['x'], corners['P5']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    if 'P12' in corners.keys() and 'P13' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P12']['x'], corners['P12']['y']), (corners['P13']['x'], corners['P13']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
    #HALF COURT LINE
    if 'P6' in corners.keys() and 'P7' in corners.keys():
        p1, p2 = calcola_intersezioni((corners['P6']['x'], corners['P6']['y']), (corners['P7']['x'], corners['P7']['y']), frame.shape[1], frame.shape[0])
        cv2.line(frame, p1, p2, color, thickness)
