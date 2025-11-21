import yaml
import numpy as np
import cv2

camera = '20936'
cam_name = f'camera_{camera}'

calib_path = 'calibration/calib.yaml'
with open(calib_path, "r") as f:
        calib_data = yaml.safe_load(f)[cam_name]
print(calib_data)

projections_path = f'calibration/output_{camera}.txt'
with open(projections_path, 'r') as file:
            proj_data = yaml.safe_load(file)
print(proj_data)

fisheye = calib_data.get("fisheye", True)
K_values = calib_data["K"]
K = np.eye(3)
K[0,0] = K_values[0]
K[1,1] = K_values[1]
K[0,2] = K_values[2]
K[1,2] = K_values[3]
d_values = calib_data["D"]
D = np.array(d_values)
H = np.array(proj_data[cam_name]["H"], dtype=np.float32).reshape(3, 3)

x,y = 100,100
undistorted_points = np.array([(x, y)], dtype=np.float32, ndmin=3)
if fisheye:
    undistorted_points = cv2.fisheye.undistortPoints(undistorted_points, K, D[:4])
else:
    undistorted_points = cv2.undistortPoints(undistorted_points, K, D)
projected_points = cv2.perspectiveTransform(undistorted_points, H)
rounded_points = np.round(projected_points).astype(int)

print(rounded_points)
