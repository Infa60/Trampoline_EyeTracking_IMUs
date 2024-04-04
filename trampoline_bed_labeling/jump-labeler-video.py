
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import json
import tkinter as tk
from tkinter import filedialog


def load_video_frames(video_file, num_frames=None):
    video = cv2.VideoCapture(video_file)
    frames = []

    if num_frames is None:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames_update = 0
    for i in tqdm(range(num_frames), desc='Loading video'):
        ret, frame = video.read()
        if type(frame) == np.ndarray:
            frames.append(frame)
            num_frames_update+=1

    video.release()

    return frames, num_frames_update

def put_text(small_image):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
    if current_state_label["Not an acrobatics"][frame_counter] == 1:
        color = (255, 0, 0)
        org = (50, 50)
        cv2.putText(small_image, "Not an acrobatics", org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif current_state_label["Acrobatics"][frame_counter] == 1:
        color = (0, 255, 0)
        org = (400, 50)
        cv2.putText(small_image, "Acrobatics", org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif current_state_label["Jump"][frame_counter] == 1:
        color = (0, 0, 255)
        org = (800, 50)
        cv2.putText(small_image, "Jump", org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow(Image_name, small_image)
    return

def labeled_not_an_acrobatics(*args):

    current_state_label["Not an acrobatics"][frame_counter:] = 1
    current_state_label["Acrobatics"][frame_counter:] = 0
    current_state_label["Jump"][frame_counter:] = 0

    image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter][:]
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
    put_text(small_image)

    put_text(small_image)
    return

def labeled_jump(*args):
    current_state_label["Jump"][frame_counter:] = 1
    current_state_label["Acrobatics"][frame_counter:] = 0
    current_state_label["Not an acrobatics"][frame_counter:] = 0

    image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter][:]
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))

    put_text(small_image)
    return

def labeled_acrobatics(*args):
    current_state_label["Acrobatics"][frame_counter:] = 1
    current_state_label["Jump"][frame_counter:] = 0
    current_state_label["Not an acrobatics"][frame_counter:] = 0

    image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter][:]
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))

    put_text(small_image)
    return


############################### code beginning #######################################################################
global image_clone, small_image, number_of_points_to_label, width_small, height_small, frame_counter, label_keys, points_labels, frames_clone
global ratio_image, Image_name, borders_points, current_state_label, csv_eye_tracking, point_label_names

Image_name = "Video"
Trackbar_name = "Frames"
ratio_image = 1 # 1.5

# $$$$$$$$$$$$$$$$$$

root = tk.Tk()
root.withdraw()
# file_path = filedialog.askopenfilename(initialdir = "/home/user/disk/Eye-tracking/PupilData/CloudExport/")
file_path = '/home/mickaelbegon/disk/Eye-tracking/PupilData/CloudExport/GuSe/2021-08-18_13-51-56-0c3cd486/8126b14b_0.0-342.656.mp4'

last_slash = file_path.rfind('/')
movie_path = file_path[:last_slash+1]
movie_name = file_path[last_slash+1 : -4].replace('.', '_')
# movie_name = file_path[last_slash+1 : -4]

# movie_file = movie_path + movie_name + "_undistorted_images.pkl"
movie_file = movie_path + movie_name + '.mp4'

# second_last_slash = file_path[:last_slash].rfind('/')
# eye_tracking_data_path = '/home/user/disk/Eye-tracking/PupilData/CloudExport/' + file_path[second_last_slash+1:last_slash+1]
# filename = eye_tracking_data_path  + 'gaze.csv'
# filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
# filename_info = eye_tracking_data_path + 'info.json'
# with open(filename_info, 'r') as f:
#   json_info = json.load(f)


frames, num_frames = load_video_frames(file_path)
num_frames = len(frames)
frames_clone = frames.copy()

# csv_read = np.char.split(pd.read_csv(filename, sep='\t').values.astype('str'), sep=',')
# timestamps_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')

# time_stamps_eye_tracking = np.zeros((len(timestamps_read), ))
# for i in range(len(timestamps_read)):
#     time_stamps_eye_tracking[i] = float(timestamps_read[i][0][2])
# csv_eye_tracking = np.zeros((len(csv_read), 7))
# for i in range(len(csv_read)):
#     csv_eye_tracking[i, 0] = float(csv_read[i][0][2]) # timestemp
#     csv_eye_tracking[i, 1] = int(round(float(csv_read[i][0][3]))) # pos_x
#     csv_eye_tracking[i, 2] = int(round(float(csv_read[i][0][4]))) # pos_y
#     csv_eye_tracking[i, 3] = float(csv_read[i][0][5]) # confidence
#     csv_eye_tracking[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

# 2 -> 0: gaze_timestamp
# 3 -> 1: norm_pos_x
# 4 -> 2: norm_pos_y
# 5 -> 3: confidence
# 4: closest image time_stamp
# 5: pos_x_bedFrame -> computed from labeling and distortion
# 6: pos_y_bedFrame -> computed from labeling and distortion

current_state_label = {"Not an acrobatics": np.ones((len(frames), )),
                       "Jump": np.ones((len(frames), )),
                       "Acrobatics": np.ones((len(frames), ))}

current_click = 0

def nothing(x):
    return


labeled_file = "/home/mickaelbegon/disk/Eye-tracking/PupilData/points_labeled/" + movie_name + "_labeling_points.pkl" # [:-4]
if os.path.exists(labeled_file):
    file = open(labeled_file, "rb")
    current_state_label = pickle.load(file)

cv2.namedWindow(Image_name)
cv2.createTrackbar(Trackbar_name, Image_name, 0, num_frames, nothing)
frame_counter = 0

playVideo = True
image_clone = frames[frame_counter].copy()
width, height, rgb = np.shape(image_clone)
small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
width_small, height_small, rgb_small = np.shape(small_image)
cv2.imshow(Image_name, small_image)

cv2.createButton("Not an acrobatics", labeled_not_an_acrobatics, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Jump", labeled_jump, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Acrobatics", labeled_acrobatics, 0, cv2.QT_PUSH_BUTTON, 0)


while playVideo == True:

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        labeled_not_an_acrobatics(small_image)
    elif key == ord('j'):
        labeled_jump(small_image)
    elif key == ord('a'):
        labeled_acrobatics(small_image)

    # if frame_counter % 15: # s'il ya un probleme, au moins on n'a pas tout perdu
    #     if not os.path.exists(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}'):
    #         os.makedirs(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}')
    #     with open(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}/{movie_name}_tempo_labeling_points.pkl', 'wb') as handle:
    #         pickle.dump([points_labels, active_points, current_state_label, csv_eye_tracking], handle)

    frame_counter = cv2.getTrackbarPos(Trackbar_name, Image_name)
    frames_clone = frames.copy()
    image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter][:]
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))

    if key == ord(','):  # if `<` then go back
        if frame_counter != 0:
            frame_counter -= 1
        cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
        image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
        image_clone[:] = frames_clone[frame_counter][:]
        small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
        put_text(small_image)

    elif key == ord('.'):  # if `>` then advance
        if frame_counter < num_frames-1:
            frame_counter += 1
        cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
        image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
        image_clone[:] = frames_clone[frame_counter][:]
        small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
        put_text(small_image)

    elif key == ord('x'):  # if `x` then quit
        playVideo = False

cv2.destroyAllWindows()

with open("/home/mickaelbegon/disk/Eye-tracking/PupilData/points_labeled/" + movie_name + "_labeling_jumps.pkl", 'wb') as handle:
    pickle.dump([current_state_label], handle)

