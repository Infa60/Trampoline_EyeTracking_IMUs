import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from IPython import embed
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import itertools
from operator import itemgetter
from unproject_PI_2d_pixel_gaze_estimates import pixelPoints_to_gazeAngles
from gaze_position_gymnasium import get_gaze_position_from_intersection
from sync_jump import moving_average


def Xsens_quat_to_orientation(
        Xsens_orientation,
        Xsens_position,
        elevation,
        azimuth,
        eye_position_height,
        eye_position_depth,
):
    """
    This function computes the orientation of the head, the eye and the gaze orientation in the global coordinate system
    and relative to each other.
    The head orientation has to be computed from the quaternion returned by Xsens since it is the end of the kinematic
    chain.
    """
    Xsens_head_position_calculated = np.zeros((6,))
    Xsens_orthogonal_thorax_position = np.zeros((6,))
    Xsens_orthogonal_head_position = np.zeros((6,))
    Quat_normalized_head = Xsens_orientation[24:28] / np.linalg.norm(
        Xsens_orientation[24:28]
    )
    Quat_normalized_thorax = Xsens_orientation[16:20] / np.linalg.norm(
        Xsens_orientation[16:20]
    )
    Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
                                  Quat_normalized_head[3])
    Quat_thorax = biorbd.Quaternion(Quat_normalized_thorax[0], Quat_normalized_thorax[1], Quat_normalized_thorax[2],
                                    Quat_normalized_thorax[3])
    RotMat_head = biorbd.Quaternion.toMatrix(Quat_head)
    EulAngles_head_global = biorbd.Rotation_toEulerAngles(RotMat_head, 'xyz').to_array()
    RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()
    RotMat_thorax = biorbd.Quaternion.toMatrix(Quat_thorax).to_array()

    Xsens_head_position_calculated[:3] = Xsens_position[18:21]
    Xsens_head_position_calculated[3:] = (
            RotMat_head @ np.array([0, 0, 0.1]) + Xsens_position[18:21]
    )

    Xsens_orthogonal_thorax_position[:3] = Xsens_position[12:15]
    Xsens_orthogonal_thorax_position[3:] = (
            RotMat_head @ np.array([0.1, 0, 0]) + Xsens_position[12:15]
    )

    Xsens_orthogonal_head_position[:3] = Xsens_position[18:21]
    Xsens_orthogonal_head_position[3:] = (
            RotMat_head @ np.array([0.1, 0, 0]) + Xsens_position[18:21]
    )

    eye_position = (
            RotMat_head @ np.array([eye_position_depth, 0, eye_position_height])
            + Xsens_position[18:21]
    )
    gaze_rotMat = biorbd.Rotation_fromEulerAngles(np.array([azimuth, elevation]), "zy").to_array()
    gaze_orientation = gaze_rotMat @ RotMat_head @ np.array([10, 0, 0]) + eye_position

    RotMat_between = np.linalg.inv(RotMat_thorax) @ RotMat_head
    EulAngles_neck = biorbd.Rotation_toEulerAngles(
        biorbd.Rotation(RotMat_between[0, 0], RotMat_between[0, 1], RotMat_between[0, 2],
                        RotMat_between[1, 0], RotMat_between[1, 1], RotMat_between[1, 2],
                        RotMat_between[2, 0], RotMat_between[2, 1], RotMat_between[2, 2]), 'zy').to_array()

    return Xsens_head_position_calculated, eye_position, gaze_orientation, EulAngles_head_global, EulAngles_neck, Xsens_orthogonal_thorax_position, Xsens_orthogonal_head_position  # EulAngles_neck_xsens #


def compute_eye_related_positions(
        Xsens_orientation,
        Xsens_position,
        elevation,
        azimuth,
        eye_position_height,
        eye_position_depth,
        bound_side,
        facing_front_wall=False,
):
    """
    This function computes the position of the eye and the gaze orientation and the gaze projected on the gymnasium in
    the global coordinate system.
    """
    Xsens_head_position_calculated = np.zeros((np.shape(Xsens_position)[0], 6))
    Xsens_orthogonal_thorax_position = np.zeros((np.shape(Xsens_position)[0], 6))
    Xsens_orthogonal_head_position = np.zeros((np.shape(Xsens_position)[0], 6))
    eye_position = np.zeros((np.shape(Xsens_position)[0], 3))
    gaze_orientation = np.zeros((np.shape(Xsens_position)[0], 3))
    EulAngles_head_global = np.zeros((np.shape(Xsens_position)[0], 3))
    EulAngles_neck = np.zeros((np.shape(Xsens_position)[0], 2))
    gaze_position_temporal_evolution_projected = np.zeros((np.shape(Xsens_position)[0], 3))
    wall_index = np.zeros((np.shape(Xsens_position)[0], 3))

    for i_time in range(len(Xsens_orientation)):
        (
            Xsens_head_position_calculated[i_time, :],
            eye_position[i_time, :],
            gaze_orientation[i_time, :],
            EulAngles_head_global[i_time, :],
            EulAngles_neck[i_time, :],
            Xsens_orthogonal_thorax_position[i_time, :],
            Xsens_orthogonal_head_position[i_time, :],
        ) = Xsens_quat_to_orientation(
            Xsens_orientation[i_time, :],
            Xsens_position[i_time, :],
            elevation[i_time],
            azimuth[i_time],
            eye_position_height,
            eye_position_depth,
        )
        gaze_position_temporal_evolution_projected[i_time, :], wall_index[i_time, :] = get_gaze_position_from_intersection(
            eye_position[i_time, :], gaze_orientation[i_time, :], bound_side, facing_front_wall
        )

    return Xsens_head_position_calculated, eye_position, gaze_orientation, gaze_position_temporal_evolution_projected, wall_index, EulAngles_head_global, EulAngles_neck, Xsens_orthogonal_thorax_position, Xsens_orthogonal_head_position


def identify_fixations(time_vector_pupil, gaze_position_temporal_evolution_projected, eye_position, wall_index, folder_name):
    """
    This function identifies the fixations based on an angle threshold of 5 degrees which is projected on the gymnasium
    to be converted in a distance threshold.
    We consider that if all the points in a window of 40 ms are within the threshold, then it is a fixation.
    """
    treshold_block = 10 * np.pi / 180  # Only for plotting the gliding fixations
    treshold_detection = 2.5 * np.pi / 180  # Real fixation threshold
    duration_threshold = 0.04
    
    fixation_timing_candidates = np.array([])
    fixation_timing_candidates_start = np.array([])
    fixation_timing_candidates_end = np.array([])
    last_index = np.argmin(np.abs((time_vector_pupil[-1] - duration_threshold) - time_vector_pupil))
    for i in range(last_index):
        ms_40_idx = np.argmin(np.abs((time_vector_pupil[i] + duration_threshold) - time_vector_pupil))
        if time_vector_pupil[ms_40_idx] - duration_threshold < 0:
            ms_40_idx += 1
        current_time = np.mean(time_vector_pupil[i: ms_40_idx + 1])
        mean_gaze_position_temporal_evolution_projected = np.mean(gaze_position_temporal_evolution_projected[i: ms_40_idx + 1, :], axis=0)
        mean_eye_position = np.mean(eye_position[i: ms_40_idx + 1, :], axis=0)
        position_threshold = np.tan(treshold_detection) * np.linalg.norm(mean_eye_position - mean_gaze_position_temporal_evolution_projected)

        if np.all(np.abs(gaze_position_temporal_evolution_projected[i: ms_40_idx + 1, :] - mean_gaze_position_temporal_evolution_projected) < position_threshold):
            if len(fixation_timing_candidates) == 0:
                fixation_timing_candidates = np.array([current_time])
                fixation_timing_candidates_start = np.array([i])
                fixation_timing_candidates_end = np.array([ms_40_idx])
            else:
                fixation_timing_candidates = np.vstack((fixation_timing_candidates, current_time))
                fixation_timing_candidates_start = np.vstack((fixation_timing_candidates_start, i))
                fixation_timing_candidates_end = np.vstack((fixation_timing_candidates_end, ms_40_idx))
    fixation_idx_candidates = np.zeros((len(time_vector_pupil),))
    for i in range(len(fixation_timing_candidates_start)):
        fixation_idx_candidates[int(fixation_timing_candidates_start[i]): int(fixation_timing_candidates_end[i])] = 1
    diff_index = fixation_idx_candidates[1:] - fixation_idx_candidates[:-1]

    fixation_blocks_start = np.where(diff_index == 1)[0] + 1
    fixation_blocks_end = np.where(diff_index == -1)[0] + 1
    if fixation_idx_candidates[0] == 1.0:
        fixation_blocks_start = np.hstack((0, fixation_blocks_start))
    if fixation_idx_candidates[-1] == 1.0:
        fixation_blocks_end = np.hstack((fixation_blocks_start, len(fixation_idx_candidates) - 1))

    fixation_positions = np.array([])
    fixation_timing = [np.array([]) for _ in range(len(fixation_blocks_start))]
    fixation_index = np.zeros(len(time_vector_pupil))
    position_threshold_block = []
    wall_index_block = []
    for i in range(len(fixation_blocks_start)):
        mean_gaze_position_temporal_evolution_projected = np.mean(
            gaze_position_temporal_evolution_projected[fixation_blocks_start[i]: fixation_blocks_end[i], :], axis=0
        )
        mean_eye_position = np.mean(eye_position[fixation_blocks_start[i]: fixation_blocks_end[i], :], axis=0)
        position_threshold = np.tan(treshold_block) * np.linalg.norm(mean_eye_position - mean_gaze_position_temporal_evolution_projected)
        norm_distace_mean_fixation_vs__point = np.linalg.norm(gaze_position_temporal_evolution_projected[fixation_blocks_start[i]: fixation_blocks_end[i], :] - mean_gaze_position_temporal_evolution_projected, axis=1)
        if not np.all(norm_distace_mean_fixation_vs__point < position_threshold):
            plt.figure()
            colors = ['r', 'g', 'b']
            for i_composantes in range(3):
                plt.plot(gaze_position_temporal_evolution_projected[fixation_blocks_start[i]: fixation_blocks_end[i], i_composantes], color=colors[i_composantes])
                plt.plot(np.ones((fixation_blocks_end[i] - fixation_blocks_start[i], 1)) * mean_gaze_position_temporal_evolution_projected[i_composantes], color=colors[i_composantes])
                plt.plot(np.ones((fixation_blocks_end[i] - fixation_blocks_start[i], 1)) * (mean_gaze_position_temporal_evolution_projected[i_composantes] - position_threshold), '--', color=colors[i_composantes])
                plt.plot(np.ones((fixation_blocks_end[i] - fixation_blocks_start[i], 1)) * (mean_gaze_position_temporal_evolution_projected[i_composantes] + position_threshold), '--', color=colors[i_composantes])
            plt.savefig(f"{folder_name}_sliding_fixation.png")
            # plt.show()
            print("Problem: sliding fixation, see saved figure")

        if len(fixation_positions) == 0:
            fixation_positions = mean_gaze_position_temporal_evolution_projected
        else:
            fixation_positions = np.vstack((fixation_positions, mean_gaze_position_temporal_evolution_projected))
        fixation_timing[i] = time_vector_pupil[fixation_blocks_start[i]: fixation_blocks_end[i]]
        fixation_index[fixation_blocks_start[i]: fixation_blocks_end[i]] = 1
        position_threshold_block += [position_threshold]
        wall_index_block += [int(np.mean(wall_index[fixation_blocks_start[i]: fixation_blocks_end[i]]))]

    if np.shape(fixation_positions) == (3, ):
        fixation_positions = np.reshape(fixation_positions, (1, 3))

    if np.shape(fixation_positions) == (0, ):
        number_of_fixation = 0
    elif len(np.shape(fixation_positions)) == 1:
        number_of_fixation = 1
    else:
        number_of_fixation = np.shape(fixation_positions)[0]

    # fixation duration in relative time
    jump_duration = time_vector_pupil[-1] - time_vector_pupil[0]
    fixation_duration_absolute = []
    fixation_duration_relative = []
    for j in range(number_of_fixation):
        fixation_duration_absolute += [(fixation_timing[j][-1] - fixation_timing[j][0])]
        fixation_duration_relative += [(fixation_timing[j][-1] - fixation_timing[j][0]) / jump_duration]

    if len(fixation_duration_absolute) > 0:
        quiet_eye_duration_absolute = fixation_duration_absolute[-1]
        quiet_eye_duration_relative = fixation_duration_relative[-1]
    else:
        quiet_eye_duration_absolute = np.nan
        quiet_eye_duration_relative = np.nan

    return fixation_positions, fixation_timing, position_threshold_block, wall_index_block, fixation_index, fixation_duration_absolute, fixation_duration_relative, quiet_eye_duration_absolute, quiet_eye_duration_relative, number_of_fixation


def find_neighbouring_candidates(time_vector_pupil, candidates, duration_threshold):
    """
    This function determines if the candidate data points form a block of consecutive data of more than the duration
    threshold.
    """
    index = np.zeros((len(time_vector_pupil)))
    if np.all(candidates[1:-1] == 0):
        return index
    else:
        diff_index_candidates = candidates[1:] - candidates[:-1]
        if candidates[0] == 1 and candidates[1] == 1:
            diff_index_candidates[0] = 1
        if candidates[-1] == 1 and candidates[-2] == 1:
            diff_index_candidates[-1] = -1
        blocks_start = np.where(diff_index_candidates == 1)[0] + 1
        blocks_end = np.where(diff_index_candidates == -1)[0] + 1
        if blocks_end[0] < blocks_start[0]:
            blocks_end = blocks_end[1:]
        for i in range(len(blocks_end)):
            if time_vector_pupil[blocks_end[i]] - time_vector_pupil[blocks_start[i]] > duration_threshold:
                index[blocks_start[i]: blocks_end[i]] = 1
        return index


def identify_head_eye_movements(elevation, azimuth, blink_index, EulAngles_head_global, EulAngles_neck,
                                fixation_index, time_vector_pupil, output_file_name, FLAG_PUPIL_ANGLES_PLOT):
    """
    This function identifies the head and eye movements being: anticipatory, compensatory, spotting, movement detection,
    blinks, and fixations.
    """
    threshold_angle = 20 * np.pi / 180  # anticipatory / compensatory
    head_velocity_threshold = 120 * np.pi / 180  # 120deg/s Dalvin (2004)
    duration_threshold = 0.04
    # position_threshold = 0.5 * np.pi / 180  # movement detection
    position_threshold = 100 * np.pi / 180  # 100deg/sec movement detection

    b, a = signal.butter(4, 0.15)
    azimuth_filtered = np.zeros((len(azimuth)))
    azimuth_filtered[:] = np.nan
    elevation_filtered = np.zeros((len(elevation)))
    elevation_filtered[:] = np.nan
    while_bool = True
    current_index = 0

    if np.all(np.isnan(elevation)):
        elevation_filtered = elevation
        azimuth_filtered = azimuth
        eye_angles = np.vstack((azimuth_filtered, elevation_filtered))
    else:
        if not np.all(~np.isnan(elevation)):
            if np.where(np.isnan(elevation))[0][0] == 0:
                next_number = np.where(~np.isnan(elevation))[0][0]
                current_index += next_number
            else:
                next_number = 0
        else:
            next_number = 0
        while while_bool:
            if np.shape(np.where(np.isnan(elevation[next_number:]))[0]) == (0,):
                next_nan = len(elevation)
                while_bool = False
            else:
                next_nan = np.where(np.isnan(elevation[next_number:]))[0]
                next_nan = next_nan[0] + current_index
                current_index = next_nan
            if len(azimuth[next_number:next_nan]) < 16:
                azimuth_filtered[next_number:next_nan] = azimuth[next_number:next_nan]
                elevation_filtered[next_number:next_nan] = elevation[next_number:next_nan]
            else:
                azimuth_filtered[next_number:next_nan] = signal.filtfilt(b, a, azimuth[next_number:next_nan])
                elevation_filtered[next_number:next_nan] = signal.filtfilt(b, a, elevation[next_number:next_nan])
            next_number = np.where(~np.isnan(elevation[next_nan:]))[0]
            if np.shape(next_number) == (0,):
                while_bool = False
            else:
                next_number = next_number[0] + current_index
                current_index = next_number

            plt.figure()
            plt.plot(azimuth, label="azimuth")
            plt.plot(azimuth_filtered, label="azimuth_filtered")
            plt.plot(elevation, label="elevation")
            plt.plot(elevation_filtered, label="elevation_filtered")
            plt.legend()
            plt.savefig(f'{output_file_name[:-4]}__filter.png')
            # plt.show()
            plt.close("all")

            eye_angles = np.vstack((azimuth_filtered, elevation_filtered))

    eye_displacement_diff_finie = (eye_angles[:, 2:] - eye_angles[:, :-2]) / 2

    eye_movement_diff_finie = np.zeros((eye_angles.shape[1] - 2, 2))
    for i in range(2):
        eye_movement_diff_finie[:, i] = (eye_angles[i, 2:] - eye_angles[i, :-2]) / (
                    time_vector_pupil[2:] - time_vector_pupil[:-2])

    neck_movement_diff_finie = np.zeros((EulAngles_neck.shape[0] - 2, 2))
    for i in range(2):
        neck_movement_diff_finie[:, i] = (EulAngles_neck[2:, i] - EulAngles_neck[:-2, i]) / (
                    time_vector_pupil[2:] - time_vector_pupil[:-2])

    head_movement_diff_finie = np.zeros((EulAngles_head_global.shape[0] - 2, 3))
    for i in range(3):
        head_movement_diff_finie[:, i] = (EulAngles_head_global[2:, i] - EulAngles_head_global[:-2, i]) / (
                    time_vector_pupil[2:] - time_vector_pupil[:-2])

    if FLAG_PUPIL_ANGLES_PLOT:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(time_vector_pupil, eye_angles[0, :] * 180 / np.pi, '-b', label="eye azimuth")
        ax.plot(time_vector_pupil, EulAngles_neck[:, 0] * 180 / np.pi, '-c', label="neck azimuth")
        ax.plot(time_vector_pupil, eye_angles[1, :] * 180 / np.pi, '-m', label="eye elevation")
        ax.plot(time_vector_pupil, EulAngles_neck[:, 1] * 180 / np.pi, '-r', label="neck elevation")
        # ax.plot(head_movement_diff_finie * 180 / np.pi, '--k', alpha=0.4, label='Head orientation in global')
        plt.xlabel("Time [s]")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.savefig(output_file_name[:-4] + "__time_angles.png")
        # plt.show()

        # angle-angle plot
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(eye_angles[0, :] * 180 / np.pi, eye_angles[1, :] * 180 / np.pi, '-m', label="eye")
        ax.plot(EulAngles_neck[:, 0] * 180 / np.pi, EulAngles_neck[:, 1] * 180 / np.pi, '-r', label="neck")
        plt.xlabel('Azimuth [degrees]')
        plt.ylabel('Elevation [degrees]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        plt.savefig(output_file_name[:-4] + "__angle_angle.png")
        # plt.show()
        plt.close("all")

        # phase diagram
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(eye_movement_diff_finie[:, 0] * 180 / np.pi, eye_angles[0, 1:-1] * 180 / np.pi, '-b',
                 label="eye azimuth")
        ax.plot(neck_movement_diff_finie[:, 0] * 180 / np.pi, EulAngles_neck[1:-1, 0] * 180 / np.pi, '-c',
                 label="neck azimuth")
        ax.plot(eye_movement_diff_finie[:, 1] * 180 / np.pi, eye_angles[1, 1:-1] * 180 / np.pi, '-m',
                 label="eye elevation")
        ax.plot(neck_movement_diff_finie[:, 1] * 180 / np.pi, EulAngles_neck[1:-1, 1] * 180 / np.pi, '-r',
                 label="neck elevation")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.xlabel('Velocity [degrees/s]')
        plt.ylabel('Position [degrees]')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        plt.savefig(output_file_name[:-4] + "__phase_diagram.png")
        # plt.show()
        plt.close("all")

        # eye vs head
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(EulAngles_neck[:, 0] * 180 / np.pi, eye_angles[0, :] * 180 / np.pi, '-r', label="azimuth")
        ax.plot(EulAngles_neck[:, 1] * 180 / np.pi, eye_angles[1, :] * 180 / np.pi, '-m', label="elevation")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.xlabel('Head [degrees]')
        plt.ylabel('Eyes [degrees]')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        plt.savefig(output_file_name[:-4] + "__eye_vs_head.png")
        # plt.show()
        plt.close("all")

    compensatory_candidates = np.zeros((len(time_vector_pupil)))
    anticipatory_candidates = np.zeros((len(time_vector_pupil)))
    angles_diff_move_orientation = np.array([])
    for i in range(np.shape(neck_movement_diff_finie)[0]):
        angle = np.arccos(np.dot(neck_movement_diff_finie[i, :], eye_movement_diff_finie[i, :]) / np.linalg.norm(
            neck_movement_diff_finie[i, :]) / np.linalg.norm(eye_movement_diff_finie[i, :]))

        if np.shape(angles_diff_move_orientation) == (0,):
            angles_diff_move_orientation = angle
        else:
            angles_diff_move_orientation = np.vstack((angles_diff_move_orientation, angle))

        if np.abs(angle) > (np.pi - threshold_angle) and np.abs(angle) < (np.pi + threshold_angle):
            compensatory_candidates[i] = 1
        elif np.abs(angle) < threshold_angle:
            anticipatory_candidates[i] = 1
        elif angle < 0:
            print("Angle négatif !")

    anticipatory_index = find_neighbouring_candidates(time_vector_pupil, anticipatory_candidates, duration_threshold)
    compensatory_index = find_neighbouring_candidates(time_vector_pupil, compensatory_candidates, duration_threshold)
    anticipatory_index = anticipatory_index.astype(bool)
    compensatory_index = compensatory_index.astype(bool)

    head_movement_diff_finie_norm = np.linalg.norm(head_movement_diff_finie, axis=1)
    spotting_candidates = np.zeros((len(head_movement_diff_finie_norm) + 2))
    spotting_candidates[1:-1] = (np.abs(head_movement_diff_finie_norm) < head_velocity_threshold).astype(int)
    spotting_index = find_neighbouring_candidates(time_vector_pupil, spotting_candidates, duration_threshold)
    spotting_index = spotting_index.astype(bool)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time_vector_pupil, np.hstack((0, head_movement_diff_finie_norm, 0)))
    ax.plot(time_vector_pupil[spotting_index], np.hstack((0, head_movement_diff_finie_norm, 0))[spotting_index], '.r')
    ax.plot(np.array([time_vector_pupil[0], time_vector_pupil[-1]]),
             np.array([head_velocity_threshold, head_velocity_threshold]), '--k')
    plt.title("Head velocity (for spotting)")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.savefig(output_file_name[:-4] + "__head_velocity.png")
    # plt.show()
    plt.close("all")

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time_vector_pupil, np.vstack((0, angles_diff_move_orientation, 0)))
    ax.plot(time_vector_pupil[anticipatory_index], np.vstack((0, angles_diff_move_orientation, 0))[anticipatory_index],
             '.m')
    ax.plot(time_vector_pupil[compensatory_index], np.vstack((0, angles_diff_move_orientation, 0))[compensatory_index],
             '.g')
    ax.plot(np.array([time_vector_pupil[0], time_vector_pupil[-1]]), np.array([threshold_angle, threshold_angle]), '--k')
    ax.plot(np.array([time_vector_pupil[0], time_vector_pupil[-1]]),
             np.array([np.pi - threshold_angle, np.pi - threshold_angle]), '--k')
    plt.title("angle between head and eye orientations for anticipatory, compensatory movements")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.savefig(output_file_name[:-4] + "__angle_between_head_and_eye_movements.png")
    # plt.show()
    plt.close("all")

    eye_displacement_norm = np.sqrt(eye_displacement_diff_finie[0, :] ** 2 + eye_displacement_diff_finie[1, :] ** 2)
    movement_detection_candidates = np.zeros((len(eye_displacement_norm) + 2))
    diff_time_vector_pupil = (time_vector_pupil[2:] - time_vector_pupil[:-2]) / 2
    movement_detection_candidates[1:-1] = (np.abs(eye_displacement_norm) < (position_threshold * diff_time_vector_pupil)).astype(int)

    diff_index_movement_detection_candidates = movement_detection_candidates[1:] - movement_detection_candidates[:-1]
    movement_detection_blocks_start = np.where(diff_index_movement_detection_candidates == 1)[0] + 1
    movement_detection_blocks_end = np.where(diff_index_movement_detection_candidates == -1)[0] + 1

    movement_detection_positions = np.array([])
    movement_detection_timing = [np.array([]) for _ in range(len(movement_detection_blocks_start))]
    movement_detection_index = np.zeros((len(time_vector_pupil)))
    for i in range(len(movement_detection_blocks_start)):
        if time_vector_pupil[movement_detection_blocks_end[i]] - time_vector_pupil[
            movement_detection_blocks_start[i]] > duration_threshold:
            mean_eye_angles = np.nanmean(
                eye_angles[:, movement_detection_blocks_start[i]: movement_detection_blocks_end[i]], axis=1)
            if len(movement_detection_positions) == 0:
                movement_detection_positions = mean_eye_angles
            else:
                movement_detection_positions = np.vstack((movement_detection_positions, mean_eye_angles))
            movement_detection_timing[i] = time_vector_pupil[
                                           movement_detection_blocks_start[i]: movement_detection_blocks_end[i]]
            for j in range(movement_detection_blocks_start[i], movement_detection_blocks_end[i]):
                if spotting_index[j] == 0:
                    movement_detection_index[j] = 1
    movement_detection_index = movement_detection_index.astype(bool)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time_vector_pupil, np.hstack((0, eye_displacement_norm, 0)) * 180 / np.pi)
    ax.plot(time_vector_pupil[movement_detection_index],
             np.hstack((0, eye_displacement_norm, 0))[movement_detection_index] * 180 / np.pi, '.r')
    ax.plot(np.array([time_vector_pupil[0], time_vector_pupil[-1]]),
             np.array([position_threshold * 180 / np.pi, position_threshold * 180 / np.pi]), '--k')
    plt.title("Eye_displacement(for movement detection)")
    plt.ylabel('Displacement [degrees]')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.savefig(output_file_name[:-4] + "__eye_displacement.png")
    # plt.show()
    plt.close("all")

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(eye_angles[0, :], '-b', label="eye azimuth")
    ax.plot(EulAngles_neck[:, 0], '-c', label="neck azimuth")
    ax.plot(eye_angles[1, :], '-m', label="eye elevation")
    ax.plot(EulAngles_neck[:, 1], '-r', label="neck elevation")

    ax.plot(anticipatory_index, '-', color='tab:green', label="anticipatory movement")
    ax.plot(compensatory_index, '-', color='tab:purple', label="compensatory movement")
    ax.plot(spotting_index, '-', color='tab:orange', label="spotting")
    ax.plot(movement_detection_index, '-', color='tab:pink', label="movement detection")
    ax.plot(blink_index, '-', color='k', label="blink")
    ax.plot(fixation_index, '-', color='tab:brown', label='fixation')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.savefig(output_file_name[:-4] + "__eye_head_movements.png")
    # plt.show()
    plt.close("all")

    neck_position_norm = np.sqrt(EulAngles_neck[:-1, 1] ** 2 + EulAngles_neck[:-1, 0] ** 2)
    eye_position_norm = np.sqrt(eye_angles[1, :-1] ** 2 + eye_angles[0, :-1] ** 2)
    dt = time_vector_pupil[1:] - time_vector_pupil[:-1]
    neck_amplitude = np.nansum(neck_position_norm * dt)
    eye_amplitude = np.nansum(eye_position_norm * dt)

    neck_amplitude_percentile = np.percentile(neck_position_norm, 90)
    eye_amplitude_percentile = np.percentile(eye_position_norm, 90)

    max_neck_amplitude = np.nanmax(np.sqrt(EulAngles_neck[:, 1] ** 2 + EulAngles_neck[:, 0] ** 2))
    max_eye_amplitude = np.nanmax(np.sqrt(eye_angles[1, :] ** 2 + eye_angles[0, :] ** 2))

    pourcentage_anticipatory = np.sum(anticipatory_index) / len(anticipatory_index)
    pourcentage_compensatory = np.sum(compensatory_index) / len(compensatory_index)
    pourcentage_spotting = np.sum(spotting_index) / len(spotting_index)
    pourcentage_movement_detection = np.sum(movement_detection_index) / len(movement_detection_index)
    pourcentage_blinks = np.sum(blink_index) / len(blink_index)

    return (neck_amplitude, eye_amplitude, max_neck_amplitude, max_eye_amplitude, neck_amplitude_percentile,
           eye_amplitude_percentile, pourcentage_anticipatory, pourcentage_compensatory, pourcentage_spotting,
           pourcentage_movement_detection, pourcentage_blinks, anticipatory_index, compensatory_index, spotting_index,
           movement_detection_index, blink_index)


def update(
        i,
        CoM_trajectory,
        Xsens_position,
        Xsens_orthogonal_thorax_position,
        Xsens_orthogonal_head_position,
        eye_position,
        gaze_position_temporal_evolution_projected,
        lines,
        CoM_point,
        line_eye_orientation,
        eyes_point,
        intersection_point,
        thorax_orthogonal,
        head_orthogonal,
        links,
):
    """
    This function is called periodically from FuncAnimation to update the animation at each frame.
    """
    CoM_point[0][0].set_data(np.array([CoM_trajectory[i, 0]]), np.array([CoM_trajectory[i, 1]]))
    CoM_point[0][0].set_3d_properties(np.array([CoM_trajectory[i, 2]]))

    for i_line, line in enumerate(lines):
        line[0].set_data(
            np.array([Xsens_position[i, 3 * links[i_line, 0]], Xsens_position[i, 3 * links[i_line, 1]]]),
            np.array([Xsens_position[i, 3 * links[i_line, 0] + 1], Xsens_position[i, 3 * links[i_line, 1] + 1]]),
        )
        line[0].set_3d_properties(
            np.array([Xsens_position[i, 3 * links[i_line, 0] + 2], Xsens_position[i, 3 * links[i_line, 1] + 2]])
        )

    thorax_orthogonal[0][0].set_data(
        np.array([Xsens_orthogonal_thorax_position[i, 0], Xsens_orthogonal_thorax_position[i, 3]]),
        np.array([Xsens_orthogonal_thorax_position[i, 1], Xsens_orthogonal_thorax_position[i, 4]]),
    )
    thorax_orthogonal[0][0].set_3d_properties(
        np.array([Xsens_orthogonal_thorax_position[i, 2], Xsens_orthogonal_thorax_position[i, 5]])
    )

    head_orthogonal[0][0].set_data(
        np.array([Xsens_orthogonal_head_position[i, 0], Xsens_orthogonal_head_position[i, 3]]),
        np.array([Xsens_orthogonal_head_position[i, 1], Xsens_orthogonal_head_position[i, 4]]),
    )
    head_orthogonal[0][0].set_3d_properties(
        np.array([Xsens_orthogonal_head_position[i, 2], Xsens_orthogonal_head_position[i, 5]])
    )

    eyes_point[0][0].set_data(np.array([eye_position[i, 0]]), np.array([eye_position[i, 1]]))
    eyes_point[0][0].set_3d_properties(np.array([eye_position[i, 2]]))

    if gaze_position_temporal_evolution_projected is not None:
        line_eye_orientation[0][0].set_data(
            np.array([eye_position[i, 0], gaze_position_temporal_evolution_projected[i, 0]]),
            np.array([eye_position[i, 1], gaze_position_temporal_evolution_projected[i, 1]]),
        )
        line_eye_orientation[0][0].set_3d_properties(np.array([eye_position[i, 2], gaze_position_temporal_evolution_projected[i, 2]]))
        intersection_point[0][0].set_data(np.array([gaze_position_temporal_evolution_projected[i, 0]]), np.array([gaze_position_temporal_evolution_projected[i, 1]]))
        intersection_point[0][0].set_3d_properties(np.array([gaze_position_temporal_evolution_projected[i, 2]]))

    return


def plot_gaze_trajectory(
        gaze_position_temporal_evolution_projected,
        fixation_positions,
        fixation_timing,
        time_vector_pupil,
        bound_side,
        position_threshold_block,
        output_file_name,
        facing_front_wall,
):
    """
    This function plots the gaze trajectory and the fixation positions projected on the gymnasium in 3D.
    """
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if not facing_front_wall:
        plot_gymnasium(bound_side, ax)
    else:
        plot_gymnasium_symmetrized(bound_side, ax)

    N = len(gaze_position_temporal_evolution_projected[:, 0]) - 1
    for j in range(N):
        ax.plot(
            gaze_position_temporal_evolution_projected[j: j + 2, 0],
            gaze_position_temporal_evolution_projected[j: j + 2, 1],
            gaze_position_temporal_evolution_projected[j: j + 2, 2],
            color=plt.cm.viridis(j / N),
        )

    if not facing_front_wall:
        for i in range(len(fixation_positions)):
            radius = position_threshold_block[i]
            phi = np.linspace(0, 2 * np.pi, 100)
            theta = np.linspace(0, np.pi, 100)
            X = radius * np.outer(np.cos(phi), np.sin(theta)) + fixation_positions[i, 0]
            Y = radius * np.outer(np.sin(phi), np.sin(theta)) + fixation_positions[i, 1]
            Z = radius * np.outer(np.ones(np.size(phi)), np.cos(theta)) + fixation_positions[i, 2]
            timing = (np.median(fixation_timing[i]) - time_vector_pupil[0]) / (
                    time_vector_pupil[-1] - time_vector_pupil[0]
            )
            ax.plot_surface(X, Y, Z, color=plt.cm.viridis(timing), alpha=0.3)
    ax.set_title("Gaze trajectory")

    plt.savefig(output_file_name[:-4] + "_gaze_trajectory.png", dpi=300)
    # plt.show()
    plt.close("all")
    return


def plot_gymnasium(bound_side, ax):
    """
    Plot the gymnasium in 3D with the walls and trampoline bed.
    """
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=10.0, azim=-90)

    ax.set_xlim3d([-8.0, 8.0])
    ax.set_ylim3d([-8.0, 8.0])
    ax.set_zlim3d([-3.0, 13.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Front right, to front left (bottom)
    plt.plot(np.array([7.193, 7.360]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")
    # Front right, to back right (bottom)
    plt.plot(np.array([-8.881, 7.193]), np.array([-bound_side, -bound_side]), np.array([0, 0]), "-k")
    # Front left, to back left (bottom)
    plt.plot(np.array([-8.881, 7.360]), np.array([bound_side, bound_side]), np.array([0, 0]), "-k")
    # Back right, to back left (bottom)
    plt.plot(np.array([-8.881, -8.881]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")

    # Front right, to front left (ceiling)
    plt.plot(
        np.array([7.193, 7.360]),
        np.array([-bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )

    # Front right, to back right (ceiling)
    plt.plot(
        np.array([-8.881, 7.193]),
        np.array([-bound_side, -bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )
    # Front left, to back left (ceiling)
    plt.plot(
        np.array([-8.881, 7.360]),
        np.array([bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )
    # Back right, to back left (ceiling)
    plt.plot(
        np.array([-8.881, -8.881]),
        np.array([-bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )

    # Front right bottom, to front right ceiling
    plt.plot(np.array([7.193, 7.193]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Front left bottom, to front left ceiling
    plt.plot(np.array([7.360, 7.360]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Back right bottom, to back right ceiling
    plt.plot(np.array([-8.881, -8.881]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Back left bottom, to back left ceiling
    plt.plot(np.array([-8.881, -8.881]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")

    # Trampoline
    X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
    Z = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, color="k", alpha=0.4)
    return

def plot_gymnasium_symmetrized(bound_side, ax):
    """
    Plot the gymnasium in 3D with the walls and trampoline bed.
    """
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=10.0, azim=-90)

    ax.set_xlim3d([-8.0, 8.0])
    ax.set_ylim3d([-8.0, 8.0])
    ax.set_zlim3d([-3.0, 13.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Front right, to front left (bottom)
    plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")
    # Front right, to back right (bottom)
    plt.plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 0]), "-k")
    # Front left, to back left (bottom)
    plt.plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 0]), "-k")
    # Back right, to back left (bottom)
    plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")

    # Front right, to front left (ceiling)
    plt.plot(
        np.array([7.2, 7.2]),
        np.array([-bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k")

    # Front right, to back right (ceiling)
    plt.plot(
        np.array([-7.2, 7.2]),
        np.array([-bound_side, -bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )
    # Front left, to back left (ceiling)
    plt.plot(
        np.array([-7.2, 7.2]),
        np.array([bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )
    # Back right, to back left (ceiling)
    plt.plot(
        np.array([-7.2, -7.2]),
        np.array([-bound_side, bound_side]),
        np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
        "-k",
    )

    # Front right bottom, to front right ceiling
    plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Front left bottom, to front left ceiling
    plt.plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Back right bottom, to back right ceiling
    plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
    # Back left bottom, to back left ceiling
    plt.plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")

    # Trampoline
    X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
    Z = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, color="k", alpha=0.4)
    return


def animate(
        time_vector_pupil,
        Xsens_orientation,
        Xsens_orientation_facing_front_wall,
        Xsens_position,
        Xsens_position_facing_front_wall,
        CoM_trajectory,
        elevation,
        azimuth,
        blink_index,
        eye_position_height,
        eye_position_depth,
        links,
        num_joints,
        output_file_name,
        folder_name,
        max_frame=0,
        FLAG_ANIMAITON=True,
        FLAG_GAZE_TRAJECTORY=True,
        FLAG_GENERATE_STATS_METRICS=True,
        FLAG_PUPIL_ANGLES_PLOT=True,
):
    """
    This function creates an animation of the athlete's body orientation and gaze orientation.
    It also creates a 3D plot of the projected gaze trajectory.
    And it computes the gaze metrics.
    """

    bound_side = 3 + 121 * 0.0254 / 2

    if FLAG_ANIMAITON:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_box_aspect([1, 1, 1])

        plot_gymnasium(bound_side, ax)

        # Initialization of the figure for the animation
        CoM_point = [ax.plot(0, 0, 0, ".r")]
        eyes_point = [ax.plot(0, 0, 0, ".c")]
        thorax_orthogonal = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-r")]
        head_orthogonal = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-r")]
        intersection_point = [ax.plot(0, 0, 0, ".c", markersize=10)]
        lines = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-k") for _ in range(len(links)-1)]
        line_eye_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-b")]

        ax.set_title("3D Gymnasium")

    # Creating the Animation object
    if max_frame == 0:
        frame_range = range(len(Xsens_position))
    else:
        frame_range = range(max_frame)

    (
        Xsens_head_position_calculated,
        eye_position,
        gaze_orientation,
        gaze_position_temporal_evolution_projected,
        wall_index,
        EulAngles_head_global,
        EulAngles_neck,
        Xsens_orthogonal_thorax_position,
        Xsens_orthogonal_head_position,
    ) = compute_eye_related_positions(
        Xsens_orientation,
        Xsens_position,
        elevation,
        azimuth,
        eye_position_height,
        eye_position_depth,
        bound_side,
        facing_front_wall=False
    )

    # Compute the same gaze metrics but for the athlete facing the front wall
    (
        _,  #Xsens_head_position_calculated_facing_front_wall,
        eye_position_facing_front_wall,
        _,  # gaze_orientation_facing_front_wall,
        gaze_position_temporal_evolution_projected_facing_front_wall,
        wall_index_facing_front_wall,
        _,  # EulAngles_head_global,
        _,  # EulAngles_neck,
        Xsens_orthogonal_thorax_position_facing_front_wall,
        Xsens_orthogonal_head_position_facing_front_wall,
    ) = compute_eye_related_positions(
        Xsens_orientation_facing_front_wall,
        Xsens_position_facing_front_wall,
        elevation,
        azimuth,
        eye_position_height,
        eye_position_depth,
        bound_side,
        facing_front_wall=True
    )

    (fixation_positions,
     fixation_timing,
     position_threshold_block,
     wall_index_block,
     fixation_index,
     fixation_duration_absolute,
     fixation_duration_relative,
     quiet_eye_duration_absolute,
     quiet_eye_duration_relative,
     number_of_fixation,
     ) = identify_fixations(
    time_vector_pupil,
    gaze_position_temporal_evolution_projected,
    eye_position,
    wall_index,
    folder_name,
    )

    if FLAG_GENERATE_STATS_METRICS:

        (
            neck_amplitude,
            eye_amplitude,
            max_neck_amplitude,
            max_eye_amplitude,
            neck_amplitude_percentile,
            eye_amplitude_percentile,
            pourcentage_anticipatory,
            pourcentage_compensatory,
            pourcentage_spotting,
            pourcentage_movement_detection,
            pourcentage_blinks,
            anticipatory_index,
            compensatory_index,
            spotting_index,
            movement_detection_index,
            blink_index,
        )= identify_head_eye_movements(
            elevation,
            azimuth,
            blink_index,
            EulAngles_head_global,
            EulAngles_neck,
            fixation_index,
            time_vector_pupil,
            output_file_name,
            FLAG_PUPIL_ANGLES_PLOT,
        )

    if FLAG_ANIMAITON:
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frame_range,
            fargs=(
                CoM_trajectory,
                Xsens_position,
                Xsens_orthogonal_thorax_position,
                Xsens_orthogonal_head_position,
                eye_position,
                gaze_position_temporal_evolution_projected,
                lines,
                CoM_point,
                line_eye_orientation,
                eyes_point,
                intersection_point,
                thorax_orthogonal,
                head_orthogonal,
                links,
            ),
            blit=False,
        )

        anim.save(output_file_name, fps=60, extra_args=["-vcodec", "libx264"], dpi=300)
        plt.show()
        plt.close("all")

    if FLAG_GAZE_TRAJECTORY:
        plot_gaze_trajectory(
            gaze_position_temporal_evolution_projected,
            fixation_positions,
            fixation_timing,
            time_vector_pupil,
            bound_side,
            position_threshold_block,
            output_file_name,
            facing_front_wall=False,
        )

        plot_gaze_trajectory(
            gaze_position_temporal_evolution_projected_facing_front_wall,
            None,
            None,
            time_vector_pupil,
            bound_side,
            None,
            output_file_name[:-4] + "_facing_front_wall.png",
            facing_front_wall=True,
        )


    return (
        number_of_fixation,
        fixation_duration_absolute,
        fixation_duration_relative,
        fixation_positions,
        fixation_timing,
        fixation_index,
        quiet_eye_duration_absolute,
        quiet_eye_duration_relative,
        gaze_position_temporal_evolution_projected,
        gaze_position_temporal_evolution_projected_facing_front_wall,
        neck_amplitude,
        eye_amplitude,
        max_neck_amplitude,
        max_eye_amplitude,
        neck_amplitude_percentile,
        eye_amplitude_percentile,
        pourcentage_anticipatory,
        pourcentage_compensatory,
        pourcentage_spotting,
        pourcentage_movement_detection,
        pourcentage_blinks,
        anticipatory_index,
        compensatory_index,
        spotting_index,
        movement_detection_index,
        blink_index,
        position_threshold_block,
        wall_index_block,
        Xsens_head_position_calculated,
        eye_position,
        gaze_orientation,
        wall_index,
        wall_index_facing_front_wall,
        EulAngles_head_global,
        EulAngles_neck,
        Xsens_orthogonal_thorax_position,
        Xsens_orthogonal_head_position,
    )
