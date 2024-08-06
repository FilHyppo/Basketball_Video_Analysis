import cv2
import numpy as np

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

def undistort_points(points, camera_matrix, dist_coeffs):
    print("Points:", points)
    return cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)

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

def get_corners(corners: list): 
    
    top_left = corners[0]
    top_right = corners[0]
    bottom_right = corners[0]
    bottom_left = corners[0]
    for value in corners:
        if value['x'] ** 2 + value['y'] ** 2 > bottom_right['x'] ** 2 + bottom_right['y'] ** 2:
            bottom_right = value

    max_distance = 0
    for value in corners:
        distance = abs(bottom_right['x'] - value['x']) + abs(bottom_right['y'] - value['y']) ** 2
        if distance > max_distance:
            max_distance = distance
            top_left = value

    max_distance = 0
    for value in corners:
        distance = (bottom_right['x'] - value['x']) ** 2 + abs(top_left['y'] - value['y'])
        if distance > max_distance:
            max_distance = distance
            bottom_left = value

    max_distance = 0
    for value in corners:
        distance = abs(bottom_left['x'] - value['x']) ** 2 + abs(bottom_left['y'] - value['y']) ** 2
        if distance > max_distance and (value['x'] != bottom_left['x'] and value['y'] != bottom_left['y']) and (value['x'] != top_left['x'] and value['y'] != top_left['y']) and (value['x'] != bottom_right['x'] and value['y'] != bottom_right['y']):
            max_distance = distance
            top_right = value

    print("Top left:", top_left)
    print("Bottom right:", bottom_right)
    print("Bottom left:", bottom_left)
    print("Top right:", top_right)

    return {
        'P0': top_left,
        'P11': bottom_right,
        'P3': bottom_left,
        'P8': top_right,
    }