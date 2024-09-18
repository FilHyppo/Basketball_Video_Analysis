import cv2
import numpy as np
import os
from .geometry import *

def line_intersection(line1, line2) -> tuple:
    # Estrai i punti
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calcola i coefficienti a1, b1, c1 per la prima retta
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    
    # Calcola i coefficienti a2, b2, c2 per la seconda retta
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    
    # Costruisci le matrici A e B del sistema lineare
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    
    # Risolvi il sistema A * [x, y] = B
    if np.linalg.det(A) != 0:  # Verifica che le rette non siano parallele
        x, y = np.linalg.solve(A, B)
        return int(x), int(y)
    else:
        return None  # Le rette sono parallele e non si intersecano

def points_on_circle(frame, corners):
    top = top_view(frame, corners)

    M, status = find_homography_matrix(frame, corners, True)

    h, w = top.shape[:2]

    foot1 = h / 50
    foot2 = w / 94
    foot = (foot1 + foot2) / 2


    points = []

    # Calcola il centro del campo
    center = (w // 2, h // 2)
    r = feet(10, foot)
    for theta in np.linspace(0, 2 * np.pi, 100):
        x = int(center[0] +  r * np.cos(theta))
        y = int(center[1] + r * np.sin(theta))
        points.append((x, y))

    P12 = corner_pos('P12', w, h)
    P13 = corner_pos('P13', w, h)
    middle_free_throw_line_right = P12[0] + (P13[0] - P12[0]) // 2, P12[1] + (P13[1] - P12[1]) // 2
    r = feet(5, foot, inches=10)
    for theta in np.linspace(np.pi / 2, 3 * np.pi / 2, 50):
        x = int(middle_free_throw_line_right[0] + r * np.cos(theta))
        y = int(middle_free_throw_line_right[1] + r * np.sin(theta))
        points.append((x, y))

    P4 = corner_pos('P4', w, h)
    P5 = corner_pos('P5', w, h)
    middle_free_throw_line_left = P4[0] + (P5[0] - P4[0]) // 2, P4[1] + (P5[1] - P4[1]) // 2
    r = feet(6, foot)
    for theta in np.linspace(- np.pi / 2, np.pi / 2, 50):
        x = int(middle_free_throw_line_left[0] + r * np.cos(theta))
        y = int(middle_free_throw_line_left[1] + r * np.sin(theta))
        points.append((x, y))

    left = center[0] - feet(9, foot, inches=10), center[1]
    right = center[0] + feet(9, foot, inches=10), center[1]

    for point in np.linspace(0, h, 20):
        points.append((left[0], int(point)))
        points.append((right[0], int(point)))

    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(points, M)
    points = points.reshape(-1, 2)
    return points.tolist()

def points_on_line(p1, p2, h, w, num_points=100):
    x_values = np.linspace(p1[0], p2[0], num_points)
    y_values = np.linspace(p1[1], p2[1], num_points)
    return list(zip(x_values, y_values))


def get_points_to_track(frame, corners):
    h, w = frame.shape[:2]
    num_points = 15
    points_to_track = points_on_line(corner_2_tuple(corners['P0']), corner_2_tuple(corners['P3']), h, w, num_points * 5)
    points_to_track += points_on_line(corner_2_tuple(corners['P3']), corner_2_tuple(corners['P11']), h, w, num_points * 10)
    points_to_track += points_on_line(corner_2_tuple(corners['P11']), corner_2_tuple(corners['P8']), h, w, num_points * 5)
    points_to_track += points_on_line(corner_2_tuple(corners['P8']), corner_2_tuple(corners['P0']), h, w, num_points * 10)

    points_to_track += points_on_line(corner_2_tuple(corners['P1']), corner_2_tuple(corners['P4']), h, w, num_points)
    points_to_track += points_on_line(corner_2_tuple(corners['P2']), corner_2_tuple(corners['P5']), h, w, num_points)
    #points_to_track += points_on_line(corner_2_tuple(corners['P5']), corner_2_tuple(corners['P4']), h, w, num_points)
    #points_to_track += points_on_line(corner_2_tuple(corners['P1']), corner_2_tuple(corners['P2']), h, w, num_points)

    points_to_track += points_on_line(corner_2_tuple(corners['P12']), corner_2_tuple(corners['P9']), h, w, num_points)
    #points_to_track += points_on_line(corner_2_tuple(corners['P10']), corner_2_tuple(corners['P9']), h, w, num_points)
    points_to_track += points_on_line(corner_2_tuple(corners['P10']), corner_2_tuple(corners['P13']), h, w, num_points)
    #points_to_track += points_on_line(corner_2_tuple(corners['P13']), corner_2_tuple(corners['P12']), h, w, num_points)

    points_to_track += points_on_line(corner_2_tuple(corners['P6']), corner_2_tuple(corners['P7']), h, w, num_points)

    # Aggiungi i punti lungo la linea del tiro libero destro
    free_throw_right = (corner_2_tuple(corners['P12']) + corner_2_tuple(corners['P13']))
    top_side_line = (corner_2_tuple(corners['P6']) + corner_2_tuple(corners['P8']))
    bottom_side_line = (corner_2_tuple(corners['P7']) + corner_2_tuple(corners['P11']))

    top_intersect = line_intersection(free_throw_right, top_side_line)
    bottom_intersect = line_intersection(free_throw_right, bottom_side_line)

    points_to_track += points_on_line(top_intersect, bottom_intersect, h, w, num_points * 5)

    # Aggiungi i punti lungo la linea del tiro libero sinistro
    free_throw_left = (corner_2_tuple(corners['P4']) + corner_2_tuple(corners['P5']))
    top_side_line = (corner_2_tuple(corners['P6']) + corner_2_tuple(corners['P0']))
    bottom_side_line = (corner_2_tuple(corners['P7']) + corner_2_tuple(corners['P3']))

    top_intersect = line_intersection(free_throw_left, top_side_line)
    bottom_intersect = line_intersection(free_throw_left, bottom_side_line)

    points_to_track += points_on_line(top_intersect, bottom_intersect, h, w, num_points)

    # Aggiungi i punti lungo le linee curve
    points_to_track += points_on_circle(frame, corners)

    # Aggiungo più punti attorno ai corners
    for id, corner in corners.items():
        point = corner['x'], corner['y']
        points_to_track.append(point)
        #points_to_track += generate_points_around(point, n=10, num_points=10)

    #points_to_track = [generate_points_around(point, 1, 1) for point in points_to_track if (point[0] < 0 or point[0] >= w or point[1] < 0 or point[1] >= h) == False]
    points_to_track = [point for point in points_to_track if (point[0] < 0 or point[0] >= w or point[1] < 0 or point[1] >= h) == False]

    points_to_track = np.array(points_to_track).reshape(-1, 2)

    return points_to_track.tolist()


def find_homography_points_with_optical_flow(img1, img2, points_to_track):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Convert points_to_track to the required format (numpy array of shape (N,1,2))
    p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 7),
                     maxLevel=7,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    #good_new = p1[(st == 1) & (err < np.mean(err))]
    #good_old = p0[(st == 1) & (err < np.mean(err))]
    good_new = p1[(st == 1) & (err < 15)]
    good_old = p0[(st == 1) & (err < 15)]


    src_pts = good_old.tolist()
    dst_pts = good_new.tolist()

    return src_pts, dst_pts


def new_corners(frame1, frame2, corners):
    """
    Per trasformare i corners di frame1 in quelli di frame2
    """
    frame1 = cv2.resize(frame1, (frame1.shape[1] // 2, frame1.shape[0] // 2))
    frame2 = cv2.resize(frame2, (frame2.shape[1] // 2, frame2.shape[0] // 2))
    for id, corner in corners.items():
        corners[id] = {'x': np.round(corner['x'] / 2), 'y': np.round(corner['y'] / 2)}

    img1 = frame1.copy()
    img2 = frame2.copy()


    points_to_track = get_points_to_track(frame1, corners)
    print("Number of points to track: ", len(points_to_track))

    for point in points_to_track:
        cv2.circle(img1, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)

    src, dst = find_homography_points_with_optical_flow(frame1, frame2, points_to_track)
    print("Points found by OF: ", len(dst), len(src))

    src = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst, dtype=np.float32).reshape(-1, 1, 2)

    for point in src:
        cv2.circle(img1, (int(point[0][0]), int(point[0][1])), 5, (255, 255, 255), -1)
    for point in dst:
        cv2.circle(img2, (int(point[0][0]), int(point[0][1])), 5, (255, 255, 255), -1)

    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    #print("Points used by findHomography: ", len(dst[mask.ravel() == 1]), len(src[mask.ravel() == 1]), "M: \n", M)

    for point in src[mask.ravel() == 1]:
        cv2.circle(img1, (int(point[0][0]), int(point[0][1])), 5, (255, 0, 0), -1)
    for point in dst[mask.ravel() == 1]:
        cv2.circle(img2, (int(point[0][0]), int(point[0][1])), 5, (0, 0, 255), -1)


    new_corners = {}
    for id, corner in corners.items():
        new_corner = cv2.perspectiveTransform(np.array([[[corner['x'], corner['y']]]], dtype=np.float32), M)[0][0]
        new_corner = np.round(new_corner)
        new_corners[id] = {'x': int(new_corner[0]), 'y': int(new_corner[1])}

    frame1_warped = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))   
    top_view2 = top_view(frame2, new_corners)

    for id, corner in new_corners.items():
        new_corners[id] = {'x': corner['x'] * 2, 'y': corner['y'] * 2}

    #new_corners = enhance_corners(top_view2, new_corners)

    return new_corners
   