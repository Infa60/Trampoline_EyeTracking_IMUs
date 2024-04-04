
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle
import matplotlib.pyplot as plt
import os.path
import json
import tkinter as tk
from tkinter import filedialog


def load_video_frames(video_file, max_frames=None):

    video = cv2.VideoCapture(video_file)
    frames = []
    num_frames = 0
    if max_frames == None:
        max_frames = 100000000
    while num_frames < max_frames: # True
        ret, frame = video.read()
        if type(frame) == np.ndarray:
            frames.append(frame)
            num_frames+=1
        else:
            break
    video.release()

    return frames, num_frames

def put_text(image_clone):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1
    if current_state_label["Eyes closed"][frame_counter_left_eye] == 1:
        color = (255, 0, 0)
        org = (50, 50)
        cv2.putText(image_clone, "Eyes closed", org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif current_state_label["Eyes opened"][frame_counter_left_eye] == 1:
        color = (0, 255, 0)
        org = (400, 50)
        cv2.putText(image_clone, "Eyes opened", org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Video", image_clone)
    return


def labeled_closed(*args):
    current_state_label["Eyes closed"][frame_counter_left_eye:] = 1
    current_state_label["Eyes opened"][frame_counter_left_eye:] = 0

    image_clone = np.zeros(np.shape(frames_clone[frame_counter_world]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter_world][:]

    put_text(image_clone)
    return

def labeled_opened(*args):
    current_state_label["Eyes opened"][frame_counter_left_eye:] = 1
    current_state_label["Eyes closed"][frame_counter_left_eye:] = 0

    image_clone = np.zeros(np.shape(frames_clone[frame_counter_world]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter_world][:]

    put_text(image_clone)
    return

def nothing(x):
    return

############################### code beginning #######################################################################
global frame_counter_left_eye, frame_counter_right_eye, frame_counter_world

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"
elif os.path.exists("/home/lim"):
    home_path = "/home/lim"
else:
    raise ValueError("Home path not found, please provide an appropriate path")

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=home_path + "/disk/Eye-tracking/PupilData/CloudExport/")

last_slash = file_path.rfind('/')
movie_path = file_path[:last_slash+1]
movie_name = file_path[last_slash+1 : -4].replace('.', '_')

movie_file = movie_path + "PI world v1 ps1" + '.mp4'
if os.path.exists(movie_path + "PI right v1 ps1" + '.mp4'):
    movie_file_right_eye = movie_path + "PI right v1 ps1" + '.mp4'
    movie_file_left_eye = movie_path + "PI left v1 ps1" + '.mp4'
else:
    movie_file_right_eye = movie_path + "PI right v1 ps1" + '.mjpeg'
    movie_file_left_eye = movie_path + "PI left v1 ps1" + '.mjpeg'

world_time_stamps_file = movie_path + "PI world v1 ps1" + '.time'
right_eye_time_stamps_file = movie_path + "PI right v1 ps1" + '.time'
left_eye_time_stamps_file = movie_path + "PI left v1 ps1" + '.time'

# GuSe : 2000, 11000
frames, num_frames = load_video_frames(movie_file)
frames_right_eye, num_frames_right_eye = load_video_frames(movie_file_right_eye)
frames_left_eye, num_frames_left_eye = load_video_frames(movie_file_left_eye)

world_time_stamps_data = np.fromfile(world_time_stamps_file, dtype="int64")
right_eye_time_stamps_data = np.fromfile(right_eye_time_stamps_file, dtype="int64")
left_eye_time_stamps_data = np.fromfile(left_eye_time_stamps_file, dtype="int64")

# plt.figure()
# plt.plot(world_time_stamps_data, 0*np.ones((len(world_time_stamps_data),)), 'b.', label='world')
# plt.plot(right_eye_time_stamps_data, 1*np.ones((len(right_eye_time_stamps_data),)), 'r.', label='right eye')
# plt.plot(left_eye_time_stamps_data, 2*np.ones((len(left_eye_time_stamps_data),)), 'g.', label='left eye')
# plt.legend()
# plt.show()

current_state_label = {"Eyes closed": np.ones((len(frames_left_eye), )),
                       "Eyes opened": np.ones((len(frames_left_eye), ))}

labeled_file = home_path + "/disk/Eye-tracking/PupilData/blinks_labeled/" + movie_name + "_labeling_blinks.pkl" # [:-4]
if os.path.exists(labeled_file):
    file = open(labeled_file, "rb")
    current_state_label, left_eye_time_stamps_data = pickle.load(file)

cv2.namedWindow("Video")
cv2.namedWindow("Right eye")
cv2.namedWindow("Left eye")
cv2.createTrackbar("Frames", "Left eye", 0, num_frames_left_eye, nothing)
frame_counter_left_eye = 0

playVideo = True
image_clone = frames.copy()
image_clone_right_eye = frames_right_eye.copy()
image_clone_left_eye = frames_left_eye.copy()

cv2.imshow("Video", image_clone[0])
cv2.imshow("Right eye", image_clone_right_eye[0])
cv2.imshow("Left eye", image_clone_left_eye[0])

cv2.createButton("Eyes closed", labeled_closed, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Eyes opened", labeled_opened, 0, cv2.QT_PUSH_BUTTON, 0)


while playVideo == True:

    key = cv2.waitKey(0) & 0xFF

    frame_counter_left_eye = cv2.getTrackbarPos("Frames", "Left eye")

    left_eye_current_time = left_eye_time_stamps_data[frame_counter_left_eye]
    frame_counter_right_eye = np.argmin(np.abs(right_eye_time_stamps_data - left_eye_current_time))
    frame_counter_world = np.argmin(np.abs(world_time_stamps_data - left_eye_current_time))

    frames_clone = frames.copy()
    image_clone = np.zeros(np.shape(frames_clone[frame_counter_world]), dtype=np.uint8)
    image_clone[:] = frames_clone[frame_counter_world][:]

    frames_clone_right_eye = frames_right_eye.copy()
    image_clone_right_eye = np.zeros(np.shape(frames_clone_right_eye[frame_counter_right_eye]), dtype=np.uint8)
    image_clone_right_eye[:] = frames_clone_right_eye[frame_counter_right_eye][:]

    frames_clone_left_eye = frames_left_eye.copy()
    image_clone_left_eye = np.zeros(np.shape(frames_clone_left_eye[frame_counter_left_eye]), dtype=np.uint8)
    image_clone_left_eye[:] = frames_clone_left_eye[frame_counter_left_eye][:]


    if key == ord('c'):
        labeled_closed(image_clone)
    elif key == ord('o'):
        labeled_opened(image_clone)
        
        
    if key == ord(','):  # if `<` then go back
        if frame_counter_left_eye != 0:
            frame_counter_left_eye -= 1

        cv2.setTrackbarPos("Frames", "Left eye", frame_counter_left_eye)

        left_eye_current_time = left_eye_time_stamps_data[frame_counter_left_eye]
        frame_counter_right_eye = np.argmin(np.abs(right_eye_time_stamps_data - left_eye_current_time))
        frame_counter_world = np.argmin(np.abs(world_time_stamps_data - left_eye_current_time))

        image_clone = np.zeros(np.shape(frames_clone[frame_counter_world]), dtype=np.uint8)
        image_clone[:] = frames_clone[frame_counter_world][:]
        put_text(image_clone)

        frames_clone_right_eye = frames_right_eye.copy()
        image_clone_right_eye = np.zeros(np.shape(frames_clone_right_eye[frame_counter_right_eye]), dtype=np.uint8)
        image_clone_right_eye[:] = frames_clone_right_eye[frame_counter_right_eye][:]
        frames_clone_left_eye = frames_left_eye.copy()
        image_clone_left_eye = np.zeros(np.shape(frames_clone_left_eye[frame_counter_left_eye]), dtype=np.uint8)
        image_clone_left_eye[:] = frames_clone_left_eye[frame_counter_left_eye][:]
        cv2.imshow("Right eye", image_clone_right_eye)
        cv2.imshow("Left eye", image_clone_left_eye)


    elif key == ord('.'):  # if `>` then advance
        if frame_counter_left_eye < num_frames_left_eye-1:
            frame_counter_left_eye += 1

        cv2.setTrackbarPos("Frames", "Left eye", frame_counter_left_eye)

        left_eye_current_time = left_eye_time_stamps_data[frame_counter_left_eye]
        frame_counter_right_eye = np.argmin(np.abs(right_eye_time_stamps_data - left_eye_current_time))
        frame_counter_world = np.argmin(np.abs(world_time_stamps_data - left_eye_current_time))

        image_clone = np.zeros(np.shape(frames_clone[frame_counter_world]), dtype=np.uint8)
        image_clone[:] = frames_clone[frame_counter_world][:]
        put_text(image_clone)

        frames_clone_right_eye = frames_right_eye.copy()
        image_clone_right_eye = np.zeros(np.shape(frames_clone_right_eye[frame_counter_right_eye]), dtype=np.uint8)
        image_clone_right_eye[:] = frames_clone_right_eye[frame_counter_right_eye][:]
        frames_clone_left_eye = frames_left_eye.copy()
        image_clone_left_eye = np.zeros(np.shape(frames_clone_left_eye[frame_counter_left_eye]), dtype=np.uint8)
        image_clone_left_eye[:] = frames_clone_left_eye[frame_counter_left_eye][:]
        cv2.imshow("Right eye", image_clone_right_eye)
        cv2.imshow("Left eye", image_clone_left_eye)

    elif key == ord('x'):  # if `x` then quit
        playVideo = False

cv2.destroyAllWindows()

with open(home_path + "/disk/Eye-tracking/PupilData/blinks_labeled/" + movie_name + "_labeling_blinks.pkl", 'wb') as handle:
    pickle.dump([current_state_label, left_eye_time_stamps_data], handle)

