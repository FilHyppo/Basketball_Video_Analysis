import cv2
import numpy as np
from django.conf import settings
import os
from . import court_drawings, geometry
import math
import random

def corner_2_tuple(corner):
    return (corner['x'], corner['y'])

def undistort_video(input_path, output_path, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        # Correggi la distorsione
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

        out.write(undistorted_frame)

    cap.release()
    out.release()

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



def feet(n, foot, inches=0):
    return int(n * foot + inches * foot / 12)


def corner_pos(id: str, w, h):

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
    h, w = frame.shape[:2]
    M, status = find_homography_matrix(frame, corners, inverse)
    if new_h is None:
        new_h = h
    if new_w is None:
        new_w = w
    frame_top_view = cv2.warpPerspective(frame, M, (new_w, new_h))

    return frame_top_view


def find_homography_points_with_SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = [ kp1[m.queryIdx].pt for m in good ]
    dst_pts = [ kp2[m.trainIdx].pt for m in good ]
    #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    return src_pts, dst_pts

def points_on_line(p1, p2, h, w, num_points=100):
    x_values = np.linspace(p1[0], p2[0], num_points)
    y_values = np.linspace(p1[1], p2[1], num_points)
    return list(zip(x_values, y_values))


def mask_court_lines(frame, corners):
    h, w = frame.shape[:2]
    thickness = 10
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(mask, corner_2_tuple(corners['P0']), corner_2_tuple(corners['P3']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P3']), corner_2_tuple(corners['P11']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P11']), corner_2_tuple(corners['P8']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P8']), corner_2_tuple(corners['P0']), 255, thickness)


    cv2.line(mask, corner_2_tuple(corners['P1']), corner_2_tuple(corners['P4']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P2']), corner_2_tuple(corners['P5']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P5']), corner_2_tuple(corners['P4']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P1']), corner_2_tuple(corners['P2']), 255, thickness)
    
    cv2.line(mask, corner_2_tuple(corners['P12']), corner_2_tuple(corners['P9']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P10']), corner_2_tuple(corners['P9']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P10']), corner_2_tuple(corners['P13']), 255, thickness)
    cv2.line(mask, corner_2_tuple(corners['P13']), corner_2_tuple(corners['P12']), 255, thickness)

    cv2.line(mask, corner_2_tuple(corners['P6']), corner_2_tuple(corners['P7']), 255, thickness)


    return mask

def mask_court(frame, corners):
    court = np.array([[corners['P0']['x'], corners['P0']['y']], [corners['P3']['x'], corners['P3']['y']], [corners['P11']['x'], corners['P11']['y']], [corners['P8']['x'], corners['P8']['y']], [corners['P0']['x'], corners['P0']['y']]], dtype=np.int32)
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [court], 255)
    return mask

def generate_points_around(p, n, num_points=5):
    x, y = p
    points = []
    for _ in range(num_points):
        # Genera un angolo casuale tra 0 e 2π
        angle = random.uniform(0, 2 * math.pi)
        # Genera una distanza casuale tra 0 e n
        radius = random.uniform(0, n)
        # Calcola le nuove coordinate
        new_x = int(x + radius * math.cos(angle))
        new_y = int(y + radius * math.sin(angle))
        points.append((new_x, new_y))
    return points


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



def landscape(frame1, frame2, corners):
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

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

    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 10)


    copy1 = frame1.copy()
    copy2 = frame2.copy()
    for point in src[mask.ravel() == 1]:
        cv2.circle(copy2, (int(point[0][0]), int(point[0][1])), 3, (0, 0, 255), -1)
    for point in dst[mask.ravel() == 1]:
        cv2.circle(copy1, (int(point[0][0]), int(point[0][1])), 3, (255, 0, 0), -1)

    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame1.jpg'), copy1)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame2.jpg'), copy2)

    print("Points used by findHomography: ", len(dst[mask.ravel() == 1]), len(src[mask.ravel() == 1]))

    frame2_warped = cv2.warpPerspective(frame2, M, (w1, h1))

    corners_frame2 = np.array([
        [0, 0],
        [w2 - 1, 0],
        [w2 - 1, h2 - 1],
        [0, h2 - 1]
    ], dtype=np.float32)
    
    transformed_corners_frame2 = cv2.perspectiveTransform(np.array([corners_frame2]), M)[0]
    
    # Trova i nuovi limiti dell'immagine trasformata per frame2
    min_x_frame2 = min(transformed_corners_frame2[:, 0])
    max_x_frame2 = max(transformed_corners_frame2[:, 0])
    min_y_frame2 = min(transformed_corners_frame2[:, 1])
    max_y_frame2 = max(transformed_corners_frame2[:, 1])

    # Trova i limiti complessivi della nuova tela
    min_x = min(0, min_x_frame2)
    max_x = max(w1, max_x_frame2)
    min_y = min(0, min_y_frame2)
    max_y = max(h1, max_y_frame2)

    # Calcola la nuova dimensione della tela di output
    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)
 
    # Calcola la traslazione necessaria per evitare coordinate negative
    translation_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    
    # Applica la traslazione alla matrice di omografia
    M_translated = translation_matrix @ M   

    # Crea una nuova tela vuota
    output_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Posiziona frame1 sulla tela
    output_image[-int(min_y):-int(min_y)+h1, -int(min_x):-int(min_x)+w1] = frame1

    # Trasforma e posiziona frame2 sulla tela
    frame2_transformed = cv2.warpPerspective(frame2, M_translated, (new_width, new_height))
    mask = (frame2_transformed.sum(axis=-1) > 0).astype(np.uint8) * 255
    output_image = cv2.bitwise_and(output_image, output_image, mask=cv2.bitwise_not(mask))
    output_image = cv2.add(output_image, frame2_transformed)

    new_corners = dict()
    for id, corner in corners.items():
        new_corner = cv2.perspectiveTransform(np.array([[[corner['x'], corner['y']]]], dtype=np.float32), translation_matrix)[0][0]
        #new_corner = cv2.perspectiveTransform(np.array([[[corner['x'], corner['y']]]], dtype=np.float32), M_translated)[0][0]
        new_corner = np.round(new_corner).astype(np.int32)
        new_corners[id] = {'x': new_corner[0], 'y': new_corner[1]}

    return output_image, new_corners


def find_homography_matrix(frame, corners, inverse=False):
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

    M, status = find_homography_matrix(frame, corners, inverse=True)

    ids = [f'P{i}' for i in range(settings.NUM_CORNERS)]
    new_corners = corners

    for id in ids:
        if id not in corners:
            corner = corner_pos(id, frame.shape[1], frame.shape[0])
            corner = cv2.perspectiveTransform(np.array([[corner]], dtype=np.float32), M)[0][0]
            corner = np.round(corner)
            new_corners[id] = {'x': int(corner[0]), 'y': int(corner[1])}
        

    return new_corners

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

    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame1.jpg'), img1)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame2.jpg'), img2)

    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    #print("Points used by findHomography: ", len(dst[mask.ravel() == 1]), len(src[mask.ravel() == 1]), "M: \n", M)

    for point in src[mask.ravel() == 1]:
        cv2.circle(img1, (int(point[0][0]), int(point[0][1])), 5, (255, 0, 0), -1)
    for point in dst[mask.ravel() == 1]:
        cv2.circle(img2, (int(point[0][0]), int(point[0][1])), 5, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame1.jpg'), img1)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame2.jpg'), img2)

    new_corners = {}
    for id, corner in corners.items():
        new_corner = cv2.perspectiveTransform(np.array([[[corner['x'], corner['y']]]], dtype=np.float32), M)[0][0]
        new_corner = np.round(new_corner)
        new_corners[id] = {'x': int(new_corner[0]), 'y': int(new_corner[1])}

    frame1_warped = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))
    court_drawings.draw_points(frame1_warped, new_corners)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'frame1_warped.jpg'), frame1_warped)
    
    top_view2 = top_view(frame2, new_corners)

    for id, corner in new_corners.items():
        new_corners[id] = {'x': corner['x'] * 2, 'y': corner['y'] * 2}

    new_corners = geometry.enhance_corners(top_view2, new_corners)


    return new_corners
    