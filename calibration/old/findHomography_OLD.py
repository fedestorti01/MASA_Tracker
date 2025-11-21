import numpy as np
import cv2
import sys, getopt
import yaml
import os

cam_points = list()
map_points = list()

def mouse_callback_cam(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global cam_points
        cam_points.append([x, y])

def mouse_callback_map(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global map_points
        map_points.append([x, y])

def mouse_callback_dynamic(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Undistort and project point
        undistorted_points = np.array([(x, y)], dtype=np.float32, ndmin=3)
        if fisheye:
            undistorted_points = cv2.fisheye.undistortPoints(undistorted_points, K, D[:4])
        else:
            undistorted_points = cv2.undistortPoints(undistorted_points, K, D)
        projected_points = cv2.perspectiveTransform(undistorted_points, H)
        rounded_points = np.round(projected_points).astype(int)

        # Show selected and projected point in cam_img and in map_img
        cam_img_copy = cam_img.copy()
        map_img_copy = map_img.copy()
        cv2.circle(cam_img_copy, (x, y), 7, (0, 0, 255), 2)
        px, py = rounded_points[0][0]
        lon = round(JGW['A'] * px + JGW['B'] * py + JGW['C'], 6)
        lat = round(JGW['D'] * px + JGW['E'] * py + JGW['F'], 6)
        #print('iniziatl point:', px, py)
        #print(JGW)
        #print(f'geo point: {lat},{lon}')


        cv2.circle(map_img_copy, (px, py), 5, (0, 0, 255), 2)
        # add a label with the coordinates
        cv2.putText(cam_img_copy, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        cv2.putText(map_img_copy, f"({lat}, {lon})", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 1)

        cv2.imshow("Selected cam points", cam_img_copy)
        cv2.imshow("Selected map points", map_img_copy)



if __name__ == "__main__":

    IMG_W, IMG_H = 1920, 1080
    #IMG_W, IMG_H = 960, 540




    camera = '20936'
    cam_path = f'calibration/camera_{camera}.jpg'
    map_path = f'calibration/map_{camera}.jpeg'
    jgw_path = f'calibration/map_{camera}.jgw'
    n_points = 8
    projections_path = f'calibration/output_{camera}.txt'
    calib_path = 'calibration/calib.yaml'
    cam_name = f'camera_{camera}'
    read_points = None

    JGW = {}
    # read jgw file if exists
    with open(jgw_path, 'r') as f:
        lines = f.readlines()
    for letter, line in zip('ADBECF', lines):
        JGW[letter] = float(line.strip())
    print(f"JGW data: {JGW}")

    # Read camera and map images
    cam_img = cv2.imread(cam_path, 1)
    map_img = cv2.imread(map_path, 1)
    
    # Read calibration
    fisheye = True
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)[cam_name]
        K_values = data["K"]
        K = np.eye(3)
        K[0,0] = K_values[0]
        K[1,1] = K_values[1]
        K[0,2] = K_values[2]
        K[1,2] = K_values[3]
        d_values = data["D"]
        D = np.array(d_values)
        fisheye = data["fisheye"]
    print(f"K: {K}")
    print(f"D: {D}")
    print(f"fisheye: {fisheye}")
    
    # Read projections
    if os.path.exists(projections_path):
        with open(projections_path, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    else:
        existing_data = {}

    if read_points and cam_name in existing_data and 'camPoints' in existing_data[cam_name] and 'mapPoints' in existing_data[cam_name]:
        # Read points from the file
        cam_points = existing_data[cam_name]['camPoints']
        map_points = existing_data[cam_name]['mapPoints']

        # Draw selected point in cam_img and in map_img
        for value in cam_points:
            cv2.circle(cam_img, tuple(value), 3, (0, 255, 0), 2)
        for value in map_points:
            cv2.circle(map_img, tuple(value), 3, (0, 255, 0), 2)

        print("Points read from file:")
        print(f"camPoints: {cam_points}")
        print(f"mapPoints: {map_points}")

    elif cam_path != "" and map_path != "":
        # Get n_points points
        for i in range(n_points + 1):
            # Show camera image
            cv2.namedWindow("Selected cam points", cv2.WINDOW_NORMAL)
            cv2.imshow("Selected cam points", cam_img)
            cv2.resizeWindow("Selected cam points", IMG_W, IMG_H)
            cv2.setMouseCallback("Selected cam points", mouse_callback_cam)
            while len(cam_points) < i:
                cv2.waitKey(100)
            cv2.destroyAllWindows()

            # Show map image
            cv2.namedWindow("Selected map points", cv2.WINDOW_NORMAL)
            cv2.imshow("Selected map points", map_img)
            cv2.resizeWindow("Selected map points", IMG_W, IMG_H)
            cv2.setMouseCallback("Selected map points", mouse_callback_map)
            while len(map_points) < i:
                cv2.waitKey(100)
            cv2.destroyAllWindows()

            # Draw selected point in cam_img and in map_img
            if cam_points and map_points:
                cv2.circle(cam_img, tuple(cam_points[-1]), 3, (0, 255, 0), 2)
                cv2.circle(map_img, tuple(map_points[-1]), 3, (0, 255, 0), 2)

        print("Points captured:")
        print(f"camPoints: {cam_points}")
        print(f"mapPoints: {map_points}")

    else:
        print("You must specify images path or points in projections file")
        exit(1)

    # Undistort points
    undistorted_points = np.array(cam_points, dtype=np.float32)
    if fisheye:
        print("k", type(K), " D", type(D))
        undistorted_points = cv2.fisheye.undistortPoints(undistorted_points.reshape(len(cam_points), 1, 2), K, D[:4])
    else:
        undistorted_points = cv2.undistortPoints(undistorted_points, K, D)

    # Calculate projection matrix
    H, mask = cv2.findHomography(undistorted_points, np.array(map_points))
    H_LMEDS, mask = cv2.findHomography(undistorted_points, np.array(map_points), cv2.LMEDS)
    H_RANSAC, mask = cv2.findHomography(undistorted_points, np.array(map_points), cv2.RANSAC)

    # Project cam_points in map_img
    projected_points = cv2.perspectiveTransform(undistorted_points, H)
    projected_points_LMEDS = cv2.perspectiveTransform(undistorted_points, H_LMEDS)
    projected_points_RANSAC = cv2.perspectiveTransform(undistorted_points, H_RANSAC)

    # Extraction of distances between points
    distances = np.linalg.norm(projected_points.reshape(len(cam_points), 2) - np.array(map_points), axis=1)
    distances_LMEDS = np.linalg.norm(projected_points_LMEDS.reshape(len(cam_points), 2) - np.array(map_points), axis=1)
    distances_RANSAC = np.linalg.norm(projected_points_RANSAC.reshape(len(cam_points), 2) - np.array(map_points), axis=1)

    algorithm = "using all point pairs"
    if np.mean(distances_LMEDS) < np.mean(distances):
        algorithm = "least-Median robust method"
        H = H_LMEDS
        projected_points = projected_points_LMEDS
        distances = distances_LMEDS
    if np.mean(distances_RANSAC) < np.mean(distances):
        algorithm = "RANSAC-based robust method"
        H = H_RANSAC
        projected_points = projected_points_RANSAC
        distances = distances_RANSAC
        
    # Create the dictionary with generated data
    new_data = {
        cam_name: {
            'H': H.reshape(9).tolist(),
            'camPoints': cam_points,
            'mapPoints': map_points,
            'algorithm': algorithm,
            'maxDistance': float(np.max(distances)),
            'minDistance': float(np.min(distances)),
            'meanDistance': float(np.mean(distances)),
            'varianceDistance': float(np.var(distances))
        }
    }
    print("\nData saved:")
    for key, value in new_data[cam_name].items():
        print(f"{key}: {value}")
    
    # Update existing data with the new data
    existing_data.update(new_data)
    class InlineListDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(InlineListDumper, self).increase_indent(flow=True, indentless=False)
    with open(projections_path, 'w') as file:
        for key, value in existing_data.items():
            file.write(f"{key}:\n")
            yaml.dump(value, file, Dumper=InlineListDumper, default_flow_style=None)
            file.write("\n")

    # Show selected and projected point in cam_img and in map_img
    rounded_points = np.round(projected_points).astype(int)
    for point in rounded_points:
        cv2.circle(map_img, tuple(point[0]), 5, (255, 0, 0), 2)
    cv2.namedWindow("Selected cam points", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Selected map points", cv2.WINDOW_NORMAL)
    cv2.imshow("Selected cam points", cam_img)
    cv2.imshow("Selected map points", map_img)
    cv2.resizeWindow("Selected cam points", IMG_W, IMG_H)
    cv2.resizeWindow("Selected map points", IMG_W, IMG_H)

    # Set mouse event callback function to dinamically show a point
    cv2.setMouseCallback("Selected cam points", mouse_callback_dynamic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()