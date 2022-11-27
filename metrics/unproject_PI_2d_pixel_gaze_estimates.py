# code addapted from a code Papr (from Pupil Labs) shared me on Discord


from pprint import pprint
import cv2
import numpy as np
import requests

# # ------------ Change accordingly ----------------------
# # To download the camera intrinsics, we need two things:
#
# # API key - Ensures fair use of Pupil Cloud resources.
# #           Go to your Pupil Cloud settings -> Developer
# #           and click `Generate New Token`. Copy and
# #           replace the xxx...xxx string below.
# API_KEY = "xxx...xxx"
#
# # Serial number - Tells Pupil Cloud which intrinsics to
# #                 to look for. You can find the value on
# #                 the side of your scene camera.
# SCENE_CAMERA_SERIAL_NUMBER = "HS6VE"
# # ------------------------------------------------------


def pixelPoints_to_gazeAngles(elevation_pixel, azimuth_pixel, SCENE_CAMERA_SERIAL_NUMBER, API_KEY):
    # Here we define some example pixel locations. Required shape: Nx2
    # points_2d = [
    #     [0, 0],  # top left
    #     [1088, 0],  # top right
    #     [1088, 1080],  # bottom right
    #     [0, 1080],  # bottom left
    #     [1088 // 2, 1080 // 2],  # center
    #     [0, 1080 // 2],  # left middle
    #     [1088 // 2, 0],  # top center
    # ]

    points_2d = [[] for i in range(len(elevation_pixel))]
    for i in range(len(elevation_pixel)):
        points_2d[i] = [azimuth_pixel[i], elevation_pixel[i]]

    def download_intrinsics():
        """Download your camera's intrinsics from Pupil Cloud"""
        # print("Downloading intrinsics...")
        serial = SCENE_CAMERA_SERIAL_NUMBER.lower()
        url = f"https://api.cloud.pupil-labs.com/hardware/{serial}/calibration.v1?json"
        resp = requests.get(url, params={"api-key": API_KEY})
        resp.raise_for_status()
        return resp.json()["result"]

    def unproject_points(points_2d, camera_matrix, distortion_coefs, normalize=False):
        """
        Undistorts points according to the camera model.
        :param pts_2d, shape: Nx2
        :return: Array of unprojected 3d points, shape: Nx3
        """
        # print("Unprojecting points...")
        # Convert type to numpy arrays (OpenCV requirements)
        camera_matrix = np.array(camera_matrix)
        distortion_coefs = np.array(distortion_coefs)
        points_2d = np.asarray(points_2d, dtype=np.float32)

        # Add third dimension the way cv2 wants it
        points_2d = points_2d.reshape((-1, 1, 2))

        # Undistort 2d pixel coordinates
        points_2d_undist = cv2.undistortPoints(points_2d, camera_matrix, distortion_coefs)
        # Unproject 2d points into 3d directions; all points. have z=1
        points_3d = cv2.convertPointsToHomogeneous(points_2d_undist)
        points_3d.shape = -1, 3

        if normalize:
            # normalize vector length to 1
            points_3d /= np.linalg.norm(points_3d, axis=1)[:, np.newaxis]

        return points_3d

    def cart_to_spherical(points_3d, apply_rad2deg=True):
        # convert cartesian to spherical coordinates
        # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
        # print("Converting cartesian to spherical coordinates...")
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        radius = np.sqrt(x**2 + y**2 + z**2)
        # elevation: vertical direction
        #   positive numbers point up
        #   negative numbers point bottom
        elevation = np.arccos(y / radius) - np.pi / 2
        # azimuth: horizontal direction
        #   positive numbers point right
        #   negative numbers point left
        azimuth = np.pi / 2 - np.arctan2(z, x)

        if apply_rad2deg:
            elevation = np.rad2deg(elevation)
            azimuth = np.rad2deg(azimuth)

        return radius, elevation, azimuth

    # print("pixel location input:")
    # pprint(points_2d)

    # Secondly, we download the camera intrinsics from Pupil Cloud
    intrinsics = download_intrinsics()
    camera_matrix = intrinsics["camera_matrix"]
    distortion_coeff = intrinsics["dist_coefs"]

    # Unproject pixel locations without normalizing. Resulting 3d points lie on a plane
    # with z=1 in reference to the camera origin (0, 0, 0).
    # points_3d = unproject_points(
    #     points_2d, camera_matrix, distortion_coeff, normalize=False
    # )
    # print("3d directional output (normalize=False):")
    # pprint(points_3d)

    # Unproject pixel locations with normalizing. Resulting 3d points lie on a sphere
    # with radius=1 around the camera origin (0, 0, 0).
    points_3d = unproject_points(points_2d, camera_matrix, distortion_coeff, normalize=True)
    # print("3d directional output (normalize=True):")
    # pprint(points_3d)

    radius, azimuth, elevation = cart_to_spherical(points_3d, apply_rad2deg=True)
    elevation = elevation * np.pi / 180
    azimuth = azimuth * np.pi / 180
    # print("radius, elevation, azimuth (in degrees):")
    # elevation: vertical direction
    #   positive numbers point up
    #   negative numbers point bottom
    # azimuth: horizontal direction
    #   positive numbers point right
    #   negative numbers point left
    # convert to numpy array for display purposes:
    # pprint(np.array([radius, elevation, azimuth]).T)

    return elevation, azimuth
