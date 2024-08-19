import cv2
import numpy as np

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
