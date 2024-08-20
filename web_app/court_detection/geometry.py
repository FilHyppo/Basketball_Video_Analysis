import cv2
import numpy as np
from . import court_drawings, utils
from django.conf import settings
import os

def enhance_corners(frame, corners):


    # Convert image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray,15,30,apertureSize=3)

    cv2.imwrite(os.path.join(settings.MEDIA_TEMP, 'edges.jpg'), edges)

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
       

    
    cv2.imwrite(os.path.join(settings.MEDIA_TEMP, 'lines_found.jpg'), img)
    
    return new_corners