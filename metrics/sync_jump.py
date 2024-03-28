import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.io as sio
from scipy import signal
# from IPython import embed
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from os.path import exists
from casadi import *
import itertools

def moving_average(data_to_be_averaged, moving_average_window_size):

    if len(np.shape(data_to_be_averaged)) == 1:
        data_averaged = np.zeros((np.shape(data_to_be_averaged)[0]))
        for i in range(np.shape(data_to_be_averaged)[0]):
            if i < moving_average_window_size:
                data_averaged[i] = np.mean(data_to_be_averaged[:2*i+1])
            elif i > (np.shape(data_to_be_averaged)[0] - moving_average_window_size - 1):
                data_averaged[i] = np.mean(
                    data_to_be_averaged[-2*(np.shape(data_to_be_averaged)[0]-i)+1:]
                )
            else:
                data_averaged[i] = np.mean(
                    data_to_be_averaged[i - moving_average_window_size : i + moving_average_window_size + 1]
                )
    elif len(np.shape(data_to_be_averaged)) == 2:
        data_averaged = np.zeros(np.shape(data_to_be_averaged))
        for j in range(np.shape(data_to_be_averaged)[1]):
            for i in range(np.shape(data_to_be_averaged)[0]):
                if i < moving_average_window_size:
                    data_averaged[i, j] = np.mean(data_to_be_averaged[:2*i+1, j])
                elif i > (np.shape(data_to_be_averaged)[0] - moving_average_window_size - 1):
                    data_averaged[i, j] = np.mean(
                        data_to_be_averaged[-2*(np.shape(data_to_be_averaged)[0]-i)+1:, j]
                    )
                else:
                    data_averaged[i, j] = np.mean(data_to_be_averaged[i - moving_average_window_size : i + moving_average_window_size + 1, j])
    else:
        print('ioci')

    return data_averaged

def plot_xsens_threshold_selection(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm, moving_average_window_size, idx_jump_candidates_xsens, candidate_start_xsens, candidate_end_xsens):
    plt.plot(
        time_vector_xsens,
        Xsens_sensorFreeAcceleration_averaged_norm,
        "-k",
        alpha=0.5,
        label=f"averaged ({moving_average_window_size} points)",
    )

    plt.plot(
        time_vector_xsens[np.where(idx_jump_candidates_xsens)[0]],
        Xsens_sensorFreeAcceleration_averaged_norm[np.where(idx_jump_candidates_xsens)[0]],
        ".k",
        label="potential jump",
    )

    plt.plot(
        time_vector_xsens[candidate_start_xsens],
        Xsens_sensorFreeAcceleration_averaged_norm[candidate_start_xsens],
        "xg",
        label="start of air time",
    )
    plt.plot(
        time_vector_xsens[candidate_end_xsens],
        Xsens_sensorFreeAcceleration_averaged_norm[candidate_end_xsens],
        "xr",
        label="end of air time",
    )
    return

def plot_synchro(
        moving_average_window_size,
        idx_jump_candidates_xsens,
        candidate_start_xsens,
        candidate_end_xsens,
        start_of_jump_pupil_index,
        end_of_jump_pupil_index,
        Xsens_sensorFreeAcceleration_averaged_norm,
        pupil_start_index_optim,
        xsens_start_of_jump_index,
        xsens_end_of_jump_index,
        candidate_start_xsens_index_optim,
        candidate_end_xsens_index_optim,
        start_of_move_index,
        xsens_start_of_move_index,
        xsens_end_of_move_index,
        time_vector_pupil,
        time_vector_xsens,
        time_offset,
        output_file_name,
):
    """
    Plot the synchronization between the Xsens and the Pupil data
    """

    plt.figure()
    plot_xsens_threshold_selection(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm,
                                   moving_average_window_size, idx_jump_candidates_xsens, candidate_start_xsens,
                                   candidate_end_xsens)

    for i in range(len(start_of_jump_pupil_index)):
        if i in pupil_start_index_optim:
            linewidth = 3
        else:
            linewidth = 1
        if i == 0:
            plt.plot(
                np.ones((2,)) * time_vector_pupil[int(start_of_jump_pupil_index[i])] - time_offset,
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--c",
                linewidth=linewidth,
                label="Start of jump Pupil",
            )
            plt.plot(
                np.ones((2,)) * time_vector_pupil[int(end_of_jump_pupil_index[i])] - time_offset,
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-c",
                linewidth=linewidth,
                label="End of jump Pupil",
            )
        else:
            plt.plot(
                np.ones((2,)) * time_vector_pupil[int(start_of_jump_pupil_index[i])] - time_offset,
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--c",
            )
            plt.plot(
                np.ones((2,)) * time_vector_pupil[int(end_of_jump_pupil_index[i])] - time_offset,
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-c",
            )

    for i in range(len(xsens_start_of_jump_index)):
        if i == 0:
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_start_of_jump_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--b",
                label="Start of jump Xsens",
            )
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_end_of_jump_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-b",
                label="End of jump Xsens",
            )
            plt.plot(time_vector_xsens[candidate_start_xsens_index_optim.astype(int)],
                     np.ones((len(candidate_start_xsens_index_optim, ))), 'xg', markersize=10,
                     label="start of jump considered")
            plt.plot(time_vector_xsens[candidate_end_xsens_index_optim.astype(int)],
                     np.ones((len(candidate_end_xsens_index_optim, ))), 'xr', markersize=10,
                     label="end of jump considered")
        else:
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_start_of_jump_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--b",
            )
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_end_of_jump_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-b",
            )
            plt.plot(time_vector_xsens[candidate_start_xsens_index_optim.astype(int)],
                     np.ones((len(candidate_start_xsens_index_optim, ))), 'xg', markersize=10)
            plt.plot(time_vector_xsens[candidate_end_xsens_index_optim.astype(int)],
                     np.ones((len(candidate_end_xsens_index_optim, ))), 'xr', markersize=10)

    # for i in range(len(start_of_move_index)):
    for i in range(len(xsens_start_of_move_index)):
        if i == 0:
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_start_of_move_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--r",
                label="Start of move Xsens",
            )
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_end_of_move_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-r",
                label="End of move Xsens",
            )
        else:
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_start_of_move_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "--r",
            )
            plt.plot(
                np.ones((2,)) * time_vector_xsens[int(xsens_end_of_move_index[i])],
                np.array(
                    [
                        np.min(Xsens_sensorFreeAcceleration_averaged_norm),
                        np.max(Xsens_sensorFreeAcceleration_averaged_norm),
                    ]
                ),
                "-r",
            )

    plt.legend()
    plt.title("Xsens")
    plt.savefig(output_file_name)
    plt.show()

def chose_closest_index_xsens(time_vector_pupil,
                              time_offset,
                              start_of_jump_pupil_index,
                              end_of_jump_pupil_index,
                              time_vector_xsens,
                              start_of_move_index,
                              end_of_move_index):
    """
    This function finds the closest index in the Xsens time vector to the pupil time vector
    """

    time_vector_pupil_offset = time_vector_pupil - time_offset
    xsens_start_of_jump_index = np.zeros((len(start_of_jump_pupil_index)))
    xsens_end_of_jump_index = np.zeros((len(start_of_jump_pupil_index)))
    for i in range(len(start_of_jump_pupil_index)):
        xsens_start_of_jump_index[i] = np.argmin(
            np.abs(time_vector_pupil_offset[int(start_of_jump_pupil_index[i])] - time_vector_xsens)
        )
        xsens_end_of_jump_index[i] = np.argmin(
            np.abs(time_vector_pupil_offset[int(end_of_jump_pupil_index[i])] - time_vector_xsens)
        )

    xsens_start_of_move_index = np.zeros((len(start_of_move_index)))
    xsens_end_of_move_index = np.zeros((len(start_of_move_index)))
    for i in range(len(start_of_move_index)):
        xsens_start_of_move_index[i] = np.argmin(
            np.abs(time_vector_pupil_offset[int(start_of_move_index[i])] - time_vector_xsens)
        )
        xsens_end_of_move_index[i] = np.argmin(
            np.abs(time_vector_pupil_offset[int(end_of_move_index[i])] - time_vector_xsens)
        )
    return xsens_start_of_jump_index, xsens_end_of_jump_index, xsens_start_of_move_index, xsens_end_of_move_index, time_vector_pupil_offset

def optim_time(time_vector_pupil, time_vector_xsens, start_of_jump_pupil_index_this_time, end_of_jump_pupil_index_this_time, candidate_start_xsens_this_time, candidate_end_xsens_this_time, diff_time, time_offset, pupil_start_index_optim, pupil_end_index_optim, candidate_start_xsens_index_optim, candidate_end_xsens_index_optim):
    """
    This function optimizes the time offset between the Xsens and the Pupil data.
    Xsens begining and end of jumps are determined with the acceleration profile.
    Pupil begining and end of jumps are determined with the manual labeling of the eye-tracking data.
    """
    x0 = - (time_vector_xsens[candidate_start_xsens_this_time[0]] - time_vector_pupil[start_of_jump_pupil_index_this_time.astype(int)[0]])

    time_diff = SX.sym("time_diff", 1)
    f = sum1(
        (
                (time_vector_pupil[start_of_jump_pupil_index_this_time.astype(int)] - time_diff)
                - np.reshape(time_vector_xsens[candidate_start_xsens_this_time], len(candidate_start_xsens_this_time))
        )
        ** 2
    ) + sum1(
        (
                (time_vector_pupil[end_of_jump_pupil_index_this_time.astype(int)] - time_diff)
                - np.reshape(time_vector_xsens[candidate_end_xsens_this_time], len(candidate_end_xsens_this_time))
        )
        ** 2
    )
    nlp = {"x": time_diff, "f": f}
    opts = {"ipopt.print_level": 0, "print_time": 0}
    solver = nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=x0)

    if sol["f"] < diff_time:
        diff_time = np.array([sol["f"]])[0][0][0]
        time_offset = np.array([sol["x"]])[0][0][0]
        pupil_start_index_optim = start_of_jump_pupil_index_this_time
        pupil_end_index_optim = end_of_jump_pupil_index_this_time
        candidate_start_xsens_index_optim = candidate_start_xsens_this_time
        candidate_end_xsens_index_optim = candidate_end_xsens_this_time

    return diff_time, time_offset, pupil_start_index_optim, pupil_end_index_optim, candidate_start_xsens_index_optim, candidate_end_xsens_index_optim


def sync_jump(
    Xsens_sensorFreeAcceleration,
    start_of_jump_pupil_index,
    end_of_jump_pupil_index,
    start_of_move_index,
    end_of_move_index,
    FLAG_SYNCHRO_PLOTS,
    output_file_name,
    csv_eye_tracking,
    Xsens_ms,
    max_threshold,
    air_time_threshold,
    Xsens_jump_idx,
    Pupil_jump_idx,
):
    """
    Synchronize the Xsens and Pupil data, by minimizing the difference between the start and end timestamps.
    The Pupil timestamps are identified through labeling and the Xsens timestamps are identified through the acceleration profile.
    It returns the pupil time vector that is shifted to match the Xsens timestamps. 
    """

    # remove nans at the begining of the trial from csv_eye_tracking
    i = 0
    while np.isnan(csv_eye_tracking[i, 0]):
        i += 1
    csv_eye_tracking = csv_eye_tracking[i:, :]

    time_vector_pupil = (csv_eye_tracking[:, 0] - csv_eye_tracking[0, 0]) / 1e9
    time_vector_xsens = (Xsens_ms - Xsens_ms[0]) / 1000

    # moving average of the acceleration to smooth the signal
    moving_average_window_size = 3
    Xsens_sensorFreeAcceleration_averaged = moving_average(Xsens_sensorFreeAcceleration[:, 6:9], moving_average_window_size)
    Xsens_sensorFreeAcceleration_averaged_norm = np.linalg.norm(Xsens_sensorFreeAcceleration_averaged, axis=1)

    # If there is no acceleration we consider that the athlete is in the air. (we do not measure the gravitational accleration due to the sensor's box being also in free fall)
    idx_jump_candidates_xsens = Xsens_sensorFreeAcceleration_averaged_norm < max_threshold
    idx_jump_candidates_xsens = idx_jump_candidates_xsens.astype(int)
    diff = idx_jump_candidates_xsens[1:] - idx_jump_candidates_xsens[:-1]
    candidate_jump_start_xsens = np.where(diff == 1)[0]
    candidate_jump_end_xsens = np.where(diff == -1)[0]
    if candidate_jump_start_xsens[0] > candidate_jump_end_xsens[0]:
        candidate_jump_end_xsens = candidate_jump_end_xsens[1:]
    if candidate_jump_start_xsens[-1] > candidate_jump_end_xsens[-1]:
        candidate_jump_start_xsens = candidate_jump_start_xsens[:-1]

    i = 0
    while time_vector_xsens[candidate_jump_end_xsens[i]] - time_vector_xsens[candidate_jump_start_xsens[i]] < 0:
        candidate_jump_end_xsens = candidate_jump_end_xsens[1:]
        if candidate_jump_end_xsens.shape[0] < candidate_jump_start_xsens.shape[0]:
            candidate_jump_start_xsens = candidate_jump_start_xsens[:-1]
        i += 1

    candidate_start_xsens = []
    candidate_end_xsens = []
    for i in range(len(candidate_jump_start_xsens)):
        if time_vector_xsens[candidate_jump_end_xsens[i]] - time_vector_xsens[candidate_jump_start_xsens[i]] > air_time_threshold:
            candidate_start_xsens += [candidate_jump_start_xsens[i]]
            candidate_end_xsens += [candidate_jump_end_xsens[i]]

    candidate_start_xsens = np.array(candidate_start_xsens)
    candidate_end_xsens = np.array(candidate_end_xsens)

    if candidate_start_xsens.shape[0] == 0:
        plt.figure()
        plot_xsens_threshold_selection(time_vector_xsens, Xsens_sensorFreeAcceleration_averaged_norm,
                                       moving_average_window_size, idx_jump_candidates_xsens, candidate_start_xsens,
                                       candidate_end_xsens)
        plt.show()

    if FLAG_SYNCHRO_PLOTS:
        plot_synchro(
            moving_average_window_size,
            idx_jump_candidates_xsens,
            candidate_start_xsens,
            candidate_end_xsens,
            start_of_jump_pupil_index,
            end_of_jump_pupil_index,
            Xsens_sensorFreeAcceleration_averaged_norm,
            np.arange(0, len(start_of_jump_pupil_index)),  # pupil_start_index_optim,
            np.arange(0, len(candidate_start_xsens)),  # xsens_start_of_jump_index,
            np.arange(0, len(candidate_start_xsens)),  # xsens_end_of_jump_index,
            np.arange(0, len(candidate_start_xsens)),  # candidate_start_xsens_index_optim,
            np.arange(0, len(candidate_start_xsens)),  # candidate_end_xsens_index_optim,
            start_of_move_index,
            np.arange(0, len(candidate_start_xsens)),  # xsens_start_of_move_index,
            np.arange(0, len(candidate_start_xsens)),  # xsens_end_of_move_index,
            time_vector_pupil,
            time_vector_xsens,
            0,  #time_offset,
            output_file_name,
        )

    diff_time = 10000
    time_offset = 0
    pupil_start_index_optim = 0
    pupil_end_index_optim = 0
    candidate_start_xsens_index_optim = 0
    candidate_end_xsens_index_optim = 0

    if len(Pupil_jump_idx) == 0:
        if min(len(candidate_start_xsens), len(start_of_jump_pupil_index)) > 2:
            nb_jumps_considered = min(len(candidate_start_xsens), len(start_of_jump_pupil_index)) - 2
        else:
            nb_jumps_considered = min(len(candidate_start_xsens), len(start_of_jump_pupil_index))

        comb_tuple_xsens = list(itertools.combinations(range(len(candidate_start_xsens)), nb_jumps_considered))
        comb_xsens = []
        for comb_i in comb_tuple_xsens:
            temp_comb_xsens = np.zeros((len(candidate_start_xsens),))
            for j in comb_i:
                temp_comb_xsens[j] = True
            comb_xsens += [temp_comb_xsens.astype(int)]

        comb_tuple_pupil = list(itertools.combinations(range(len(start_of_jump_pupil_index)), nb_jumps_considered))
        comb_pupil = []
        for comb_i in comb_tuple_pupil:
            temp_comb_pupil = np.zeros((len(start_of_jump_pupil_index),))
            for j in comb_i:
                temp_comb_pupil[j] = True
            comb_pupil += [temp_comb_pupil.astype(int)]

        combine_lists = list(itertools.product(comb_xsens, comb_pupil))

    else:
        temp_comb_xsens = np.zeros((len(candidate_start_xsens),))
        temp_comb_pupil = np.zeros((len(start_of_jump_pupil_index),))
        for i in Pupil_jump_idx:
            temp_comb_pupil[i] = 1
        for i in Xsens_jump_idx:
            temp_comb_xsens[i] = 1
        combine_lists = [(np.array(temp_comb_xsens).astype(int), np.array(temp_comb_pupil).astype(int))]

    for i in range(len(combine_lists)):
        comb_xsens_i = combine_lists[i][0]
        comb_xsens_i = comb_xsens_i.astype(bool)
        comb_pupil_i = combine_lists[i][1]
        comb_pupil_i = comb_pupil_i.astype(bool)

        candidate_start_xsens_this_time = candidate_start_xsens[comb_xsens_i]
        candidate_end_xsens_this_time = candidate_end_xsens[comb_xsens_i]
        start_of_jump_pupil_index_this_time = start_of_jump_pupil_index[comb_pupil_i]
        end_of_jump_pupil_index_this_time = end_of_jump_pupil_index[comb_pupil_i]
        diff_time, time_offset, pupil_start_index_optim, pupil_end_index_optim, candidate_start_xsens_index_optim, candidate_end_xsens_index_optim = optim_time(
            time_vector_pupil, time_vector_xsens, start_of_jump_pupil_index_this_time,
            end_of_jump_pupil_index_this_time, candidate_start_xsens_this_time,
            candidate_end_xsens_this_time, diff_time, time_offset, pupil_start_index_optim, pupil_end_index_optim,
            candidate_start_xsens_index_optim, candidate_end_xsens_index_optim)

        xsens_start_of_jump_index, xsens_end_of_jump_index, xsens_start_of_move_index, xsens_end_of_move_index, time_vector_pupil_offset = chose_closest_index_xsens(
            time_vector_pupil,
            time_offset,
            start_of_jump_pupil_index,
            end_of_jump_pupil_index,
            time_vector_xsens,
            start_of_move_index,
            end_of_move_index)


    if FLAG_SYNCHRO_PLOTS:
        plot_synchro(
            moving_average_window_size,
            idx_jump_candidates_xsens,
            candidate_start_xsens,
            candidate_end_xsens,
            start_of_jump_pupil_index,
            end_of_jump_pupil_index,
            Xsens_sensorFreeAcceleration_averaged_norm,
            pupil_start_index_optim,
            xsens_start_of_jump_index,
            xsens_end_of_jump_index,
            candidate_start_xsens_index_optim,
            candidate_end_xsens_index_optim,
            start_of_move_index,
            xsens_start_of_move_index,
            xsens_end_of_move_index,
            time_vector_pupil,
            time_vector_xsens,
            time_offset,
            output_file_name,
        )

    return (
        xsens_start_of_jump_index,
        xsens_end_of_jump_index,
        xsens_start_of_move_index,
        xsens_end_of_move_index,
        time_vector_xsens,
        time_vector_pupil_offset,
        csv_eye_tracking,
    )
