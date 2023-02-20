import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from animate_JCS import find_neighbouring_candidates


def animate_confidence_threshold(pos_xy, idx_reject_points, timestamps_relative, frames, max_frame, movie_name, out_path):
    def update(
        i,
        pos_xy,
        idx_reject_points,
        gaze_point,
        wrong_gaze_estimate_point,
        gaze_trajectory,
        timestamps_relative,
        frames,
    ):

        if i % 100 == 0:
            print(str(i) + "th frame (confidence threshold)")

        # print("i+30 : ", i+30)
        # print("int(timestamps_relative[i+30]) : ", int(timestamps_relative[i+30]))
        # print("frames[0][int(timestamps_relative[i+30])] : ", frames[0][int(timestamps_relative[i+30])])
        if len(timestamps_relative) - 1 < i + 30:
            video_image.set_data(frames[0][int(timestamps_relative[-1])])
        else:
            video_image.set_data(frames[0][int(timestamps_relative[i + 30])])
        for i_point in range(30):
            if len(pos_xy[:, 0]) - 1 > i + 30:
                if idx_reject_points[i + i_point] == 0:
                    gaze_point[i_point][0].set_data(
                        np.array([pos_xy[i + i_point, 0]]), np.array([pos_xy[i + i_point, 1]])
                    )
                    wrong_gaze_estimate_point[i_point][0].set_data(np.array([np.nan]), np.array([np.nan]))
                else:
                    wrong_gaze_estimate_point[i_point][0].set_data(
                        np.array([pos_xy[i + i_point, 0]]), np.array([pos_xy[i + i_point, 1]])
                    )
                    gaze_point[i_point][0].set_data(np.array([np.nan]), np.array([np.nan]))

                if i_point < 29:
                    gaze_trajectory[i_point][0].set_data(
                        np.array([pos_xy[i + i_point, 0], pos_xy[i + i_point + 1, 0]]),
                        np.array([pos_xy[i + i_point, 1], pos_xy[i + i_point + 1, 1]]),
                    )
        return

    fig = plt.figure()

    gaze_point = [plt.plot(0, 0, "og") for _ in range(30)]
    wrong_gaze_estimate_point = [plt.plot(0, 0, "or") for _ in range(30)]
    gaze_trajectory = [plt.plot(np.array([0, 0]), np.array([0, 0]), "-k") for _ in range(30)]
    video_image = plt.imshow(frames[0][0])

    if max_frame == 0:
        frame_range = range(len(pos_xy))
    else:
        frame_range = range(max_frame)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_range,
        fargs=(
            pos_xy,
            idx_reject_points,
            gaze_point,
            wrong_gaze_estimate_point,
            gaze_trajectory,
            timestamps_relative,
            frames,
        ),
        blit=False,
    )

    # anim.save(out_path + '/' + movie_name + "_wrong_gaze_data_threshold.mp4", fps=60, extra_args=["-vcodec", "libx264"])
    # print("Generated : " + movie_name + "_wrong_gaze_data_threshold.mp4")
    plt.show()

    return

def home_made_blink_confidence_threshold(csv_eye_tracking, csv_blinks, blink_duration_threshold, frames=None, GENERATE_VIDEO_CONFIDENCE_THRESHOLD=None, movie_name=None, out_path=None):

    timestamps = csv_eye_tracking[:, 0]
    pos_xy = csv_eye_tracking[:, 1:3]
    timestamps_relative = csv_eye_tracking[:, 4]

    diff_dt = np.zeros((len(timestamps),))
    diff_dt[0] = timestamps[1] - timestamps[0]
    diff_dt[-1] = timestamps[-1] - timestamps[-2]
    diff_dt[1:-1] = timestamps[2:] - timestamps[:-2]

    diff_xy = np.array([(pos_xy[1:, 0] - pos_xy[:-1, 0]), pos_xy[1:, 1] - pos_xy[:-1, 1]]).T
    diff_xy_norm = np.sqrt(diff_xy[:, 0] ** 2 + diff_xy[:, 1] ** 2)

    angle_xy = np.zeros((len(timestamps),))
    for i in range(len(timestamps) - 3):
        if diff_xy_norm[i + 1] == 0 or diff_xy_norm[i] == 0:
            angle_xy[i + 1] = 0
        else:
            prod_div_norm = np.dot(diff_xy[i, :], diff_xy[i + 1, :]) / (diff_xy_norm[i + 1] * diff_xy_norm[i])
            if prod_div_norm > 1.0:
                angle_xy[i + 1] = 0
            elif prod_div_norm < -1.0:
                angle_xy[i + 1] = np.pi
            else:
                angle_xy[i + 1] = np.arccos(prod_div_norm)

    nb_points = 10
    orientation_angle_threshold = 15 * np.pi / 180  # degrees
    distance_threshold = 5  # pixels
    idx_reject_points = np.zeros((len(timestamps),))
    for i in range(len(timestamps) - nb_points):
        if (
            np.sum(diff_xy_norm[i : i + nb_points] > distance_threshold) > nb_points / 2
            and np.sum(angle_xy[i : i + nb_points] > orientation_angle_threshold) > nb_points / 2
        ):
            # if np.all(diff_xy_norm[i:i+nb_points] > distance_threshold) and np.sum(angle_xy[i:i+nb_points] > orientation_angle_threshold) > nb_points/2:
            idx_reject_points[i + 1 : i + nb_points - 1] = 1
    idx_reject_points = idx_reject_points.astype(np.int)

    if GENERATE_VIDEO_CONFIDENCE_THRESHOLD:
        animate_confidence_threshold(pos_xy, idx_reject_points, timestamps_relative, frames, 100, movie_name, out_path)

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
        blinks_index_pupil = find_neighbouring_candidates(time_vector, blinks_candidates, blink_duration_threshold)

    blinks_index_home_made = find_neighbouring_candidates(time_vector, idx_reject_points, blink_duration_threshold)

    plt.figure()
    plt.plot(csv_eye_tracking[:, 4], blinks_index_home_made, '-r', linewidth=4, label="points rejected by my code")
    # plt.plot(csv_eye_tracking[:, 4], blinks_index, '-g', label="points in a blink > 150ms")
    plt.plot(csv_eye_tracking[:, 4], blinks_index_pupil, '-b', label="points in a blink detected by pupil")
    plt.legend()
    plt.show()

    csv_eye_tracking[np.where(blinks_index_home_made), 1] = np.nan
    csv_eye_tracking[np.where(blinks_index_home_made), 2] = np.nan
    csv_eye_tracking[np.where(blinks_index_home_made), 3] = 0

    return csv_eye_tracking

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
