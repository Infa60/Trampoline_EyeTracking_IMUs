import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from animate_JCS import find_neighbouring_candidates

def remove_data_during_blinks(csv_eye_tracking, csv_blinks, blink_duration_threshold):

    # Use the blink provided by Pupil which turned out to be quite the same as the ones I estimated

    # Do not consider the visual data if during a blink
    blink_starts = (csv_blinks[:, 0] - csv_eye_tracking[0, 0]) / 1e9
    blink_ends = (csv_blinks[:, 1] - csv_eye_tracking[0, 0]) / 1e9
    time_vector = (csv_eye_tracking[:, 0] - csv_eye_tracking[0, 0]) / 1e9

    if np.shape(csv_blinks) != 0:
        number_of_blinks = csv_blinks.shape[0]
        blinks_candidates = np.zeros((len(time_vector)))
        for i in range(len(time_vector)):
            for j in range(number_of_blinks):
                if time_vector[i] > blink_starts[j] and time_vector[i] < blink_ends[j]:
                    blinks_candidates[i] = 1
        blinks_index = find_neighbouring_candidates(time_vector, blinks_candidates, blink_duration_threshold)

    csv_eye_tracking[np.where(blinks_index), 1] = np.nan
    csv_eye_tracking[np.where(blinks_index), 2] = np.nan
    csv_eye_tracking[np.where(blinks_index), 3] = 0

    return csv_eye_tracking
