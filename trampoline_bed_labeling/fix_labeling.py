
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

### Load the problematic file

labeling_pkl_name = "ab92cfe0_0_0-289_74_labeling_points.pkl"

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"
else:
    raise ValueError("Home path not found, please provide an appropriate path")

labeling_pkl_path = home_path + "/disk/Eye-tracking/PupilData/points_labeled/"

with open(labeling_pkl_path + labeling_pkl_name, 'rb') as file:
    data = pickle.load(file)

# AOI labels
# points_labels = data[0]
# active_points = data[1]
# curent_AOI_label = data[2]
# csv_eye_tracking = data[3]

# Jump labels only
curent_AOI_label = data[0]

plt.figure()
plt.plot(curent_AOI_label['Jump'], '-r')
plt.plot(curent_AOI_label['Acrobatics'], '-b')
plt.show()

### Do modifications to the file
curent_AOI_label['Jump'][1200:] = 0
curent_AOI_label['Acrobatics'][1200:] = 0
curent_AOI_label['Not an acrobatics'][1200:] = 1


### Overwrite the file with corrections
with open(labeling_pkl_path + labeling_pkl_name + 'modif', 'wb') as handle:
    pickle.dump([points_labels, active_points, curent_AOI_label, csv_eye_tracking], handle)

### Overwrite the file with corrections
with open(labeling_pkl_path + labeling_pkl_name + 'modif', 'wb') as handle:
    pickle.dump([curent_AOI_label], handle)

