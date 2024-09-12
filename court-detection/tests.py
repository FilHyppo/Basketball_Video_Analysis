from utils import *
from motion_compensation import *

def main():
    img = cv2.imread('image.jpg')
    dist_param = {"fx": "1150", 
                "fy": "1150",
                "cx": "640",
                "cy": "360", 
                "k1": "-0.55", 
                "k2": "0.17", 
                "k3": "0", 
                "k4": "0"
            }
    M = camera_matrix(distortion_parameters=dist_param)
    dist_coeff = dist_coeffs(distortion_parameters=dist_param)

    img = undistort_frame(img, M, dist_coeff)
    cv2.imwrite("out/undistorted_img.jpg", img)
    corners = {
        "P0": {"x": 410, "y": 368}, 
        "P1": {"x": 356, "y": 383}, 
        "P2": {"x": 284, "y": 404}, 
        "P3": {"x": 153, "y": 442}, 
        "P4": {"x": 457, "y": 386}, 
        "P5": {"x": 405, "y": 409}, 
        "P6": {"x": 626, "y": 371}, 
        "P7": {"x": 570, "y": 465}, 
        "P8": {"x": 887, "y": 375}, 
        "P9": {"x": 943, "y": 397}, 
        "P10": {"x": 1026, "y": 430}, 
        "P11": {"x": 1207, "y": 501}, 
        "P12": {"x": 797, "y": 394}, 
        "P13": {"x": 832, "y": 423}, 
        "P14": {"x": 608, "y": 401},
    }

    mask = mask_3_point_line(img, corners)
    #mask = cv2.bitwise_not(mask)

    three_point_area = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("out/3_point.jpg", three_point_area)
    cv2.imwrite("out/mask.jpg", mask)
    print("Maschera salvata in mask.jpg")

if __name__ == '__main__':
    main()