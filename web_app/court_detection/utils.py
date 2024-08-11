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

    bottom_right = max(corners, key=lambda x: x['x'])
    corners.remove(bottom_right)

    distance_x = 999999
    for value in corners:
        if abs(value['x'] - bottom_right['x']) < distance_x:
            top_right = value
            distance_x = abs(value['x'] - bottom_right['x'])
    corners.remove(top_right)

    min_y = 999999
    for value in corners:
        if value['y'] < min_y:
            top_left = value
            min_y = value['y']
    corners.remove(top_left)

    diff_x = 999999
    for value in corners:
        d = (value['x'] - top_left['x']) #SENZA abs perchè voglio sia indietro
        if d < diff_x:
            bottom_left = value
            diff_x = d
    
    corners.remove(bottom_left)

    middle_top = min(corners, key=lambda x: x['y'])
    corners.remove(middle_top)
    middle_bottom = corners[0]

    print("Top left:", top_left)
    print("Bottom right:", bottom_right)
    print("Bottom left:", bottom_left)
    print("Top right:", top_right)

    return {
        'P0': top_left,
        'P11': bottom_right,
        'P3': bottom_left,
        'P8': top_right,
        'P6': middle_top,
        'P7': middle_bottom,
    }

def draw_points(frame, corners: dict):
    
    for id, corner in corners.items():
        cv2.circle(frame, (corner['x'], corner['y']), 10, (0, 255, 0), -1)
        cv2.putText(frame, id, (corner['x'], corner['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



def feet(n, foot, inches=0):
    return int(n * foot + inches * foot / 12)

def draw_lines(frame):
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


def corner_pos(id: str, w, h):

    foot1 = h / 50
    foot2 = w / 94
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
    return 0, 0