# findHomography

## Tool for projection matrix (homography) calculation

This tool makes you manually select n_points corresponding points on cam image and map image and then calculates and saves the projection matrix used to project pixel from camera image to map image. It is mandatory to select ground points and the better the chosen points will be distributed, the better will be the projection. The minimum of n_points to select is four as long as the points are three by three not aligned, the suggested n_points is 6 or 8 depending on the availability of recognizable features on the images.

After the generation of the projection matrix you will see both the camera and map image (move windows without pressing keyboard keys) with the selected point marked in green, the selected point in cam image projected on the map image using the calculated projection matrix marked in blue and you can move the cursor on the camera image highlighting a red point and see the corresponding projected point marke with red on the map image.

### How to run it

```bash
python3 findHomography.py
```
* Load camera and map images
* Select or load previous points-pairs
* Compute and export homography