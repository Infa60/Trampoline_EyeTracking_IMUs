import biorbd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from IPython import embed
import pandas as pd
import argparse
import os
import sys

from sync_jump import sync_jump
from CoM_transfo import CoM_transfo
from get_data_at_same_timestamps import get_data_at_same_timestamps
from animate_JCS import animate
from remove_data_during_blinks import remove_data_during_blinks, home_made_blink_confidence_threshold
from set_initial_orientation import rotate_pelvis_to_initial_orientation, get_initial_gaze_orientation
sys.path.append('../trampoline_bed_labeling/')
from create_gaussian_heatMap import run_create_heatmaps, load_pupil
sys.path.append('../stick_figure_for_figures/')
from generate_stick_figure import generate_stick_figure


def load_anthropo(anthropo_name):
    """
    This function loads the anthropometric parameters of the subject from a txt file.
    The anthropometric parameters are the same as the measurements needed by xsens.
    """
    global eye_position_height, eye_position_depth
    anthropo_table = np.char.split(pd.read_csv(anthropo_name, sep="\t").values.astype("str"), sep=",")
    hip_height = float(anthropo_table[2][0][1]) / 100
    eye_position_height = float(anthropo_table[11][0][1]) / 100
    eye_position_depth = float(anthropo_table[12][0][1]) / 100
    return hip_height


def load_xsens(file_dir, xsens_file_name):
    """
    This function loads the xsens data which were exported using Xsens .mvnx and unpacked with the Matlab code main_gaze_mapping.m.
    """

    # Order of the links between the points (joint coordinates system) provided by Xsens
    links = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5],
                      [5, 6],
                      [4, 7],
                      [7, 8],
                      [8, 9],
                      [9, 10],
                      [4, 11],
                      [11, 12],
                      [12, 13],
                      [13, 14],
                      [0, 15],
                      [15, 16],
                      [16, 17],
                      [17, 18],
                      [0, 19],
                      [19, 20],
                      [20, 21],
                      [21, 22]])

    Xsens_Subject_name = sio.loadmat(file_dir + xsens_file_name + '/' + "Subject_name.mat")["Subject_name"]
    Xsens_frameRate = sio.loadmat(file_dir + xsens_file_name + '/'  + "frameRate.mat")["frameRate"]
    Xsens_time = sio.loadmat(file_dir + xsens_file_name + '/' + "time.mat")["time"]
    Xsens_index = sio.loadmat(file_dir + xsens_file_name + '/' + "index.mat")["index"]
    Xsens_ms = sio.loadmat(file_dir + xsens_file_name + '/' + "ms.mat")["ms"]

    Xsens_position = sio.loadmat(file_dir + xsens_file_name + '/' + "position.mat")["position"]
    Xsens_orientation = sio.loadmat(file_dir + xsens_file_name + '/' + "orientation.mat")["orientation"]
    Xsens_sensorFreeAcceleration = sio.loadmat(file_dir + xsens_file_name + '/' + "sensorFreeAcceleration.mat")[
        "sensorFreeAcceleration"
    ]
    Xsens_jointAngle = sio.loadmat(file_dir + xsens_file_name + '/' + "jointAngle.mat")["jointAngle"]
    Xsens_jointAngle = Xsens_jointAngle * np.pi / 180
    Xsens_centerOfMass = sio.loadmat(file_dir + xsens_file_name + '/' + "centerOfMass.mat")["centerOfMass"]
    Xsens_global_JCS_positions = sio.loadmat(file_dir + xsens_file_name + '/' + "global_JCS_positions.mat")[
        "global_JCS_positions"
    ]
    Xsens_global_JCS_orientations = sio.loadmat(file_dir + xsens_file_name + '/' + "global_JCS_orientations.mat")[
        "global_JCS_orientations"
    ]

    num_joints = int(round(np.shape(Xsens_position)[1]) / 3)

    return Xsens_ms, Xsens_position, Xsens_orientation, Xsens_centerOfMass, links, num_joints, Xsens_sensorFreeAcceleration, Xsens_global_JCS_orientations, Xsens_jointAngle


def run_analysis(
    home_path,
    subject_name,
    move_names,
    repetition_number,
    move_orientation,
    xsens_file_name,
    movie_name,
    eye_tracking_data_path,
    subject_expertise,
    gaze_position_labels,
    out_path,
    anthropo_name,
    max_threshold,
    air_time_threshold,
    Xsens_jump_idx,
    Pupil_jump_idx,
    GENERATE_VIDEO_CONFIDENCE_THRESHOLD,
    GENERATE_HEATMAPS,
    FLAG_SYNCHRO_PLOTS,
    FLAG_COM_PLOTS,
    FLAG_ANIMAITON,
    FLAG_PUPIL_ANGLES_PLOT,
    FLAG_GAZE_TRAJECTORY,
    FLAG_GENERATE_STATS_METRICS,
    FLAG_ANALYSIS,
    GENERATE_STICK_FIGURE_FOR_GRAPHS,
    API_KEY,
):
    """
    This function is the main function of the analysis pipeline. It is called by the main.py script.
    """
    if FLAG_ANALYSIS:
        hip_height = load_anthropo(anthropo_name)

        file_dir = home_path + f'/disk/Eye-tracking/XsensData/{subject_name}/exports_shoulder_height/'
        Xsens_ms, Xsens_position, Xsens_orientation, Xsens_centerOfMass, links, num_joints, Xsens_sensorFreeAcceleration, Xsens_global_JCS_orientations, Xsens_jointAngle = load_xsens(file_dir, xsens_file_name)

    (
        curent_AOI_label,
        csv_eye_tracking,
        csv_blinks,
        start_of_move_index,
        end_of_move_index,
        start_of_jump_index,
        end_of_jump_index,
        start_of_move_index_image,
        end_of_move_index_image,
        start_of_jump_index_image,
        end_of_jump_index_image,
        time_stamps_eye_tracking_index_on_pupil,
        SCENE_CAMERA_SERIAL_NUMBER,
    ) = load_pupil(gaze_position_labels, eye_tracking_data_path)

    blink_duration_threshold = 0.1
    csv_eye_tracking_confident = remove_data_during_blinks(csv_eye_tracking, csv_blinks, blink_duration_threshold)
    # csv_eye_tracking_confident = home_made_blink_confidence_threshold(csv_eye_tracking, csv_blinks, blink_duration_threshold)

    if GENERATE_HEATMAPS:
        move_summary_heatmaps = run_create_heatmaps(
            subject_name,
            subject_expertise,
            move_names,
            move_orientation,
            repetition_number,
            movie_name,
            out_path,
            start_of_move_index_image,
            end_of_move_index_image,
            curent_AOI_label,
            csv_eye_tracking_confident,
            gaze_position_labels,
        )

    if FLAG_ANALYSIS:
        # If the position of the pelvis in Xsens in not (0, 0, 0), for all frames, the data were not exported with the right options.
        if np.logical_and(np.any(Xsens_position[:, :2] > 0.01), np.any(Xsens_position[:, :2] < -0.01)):
            plt.figure()
            plt.plot(Xsens_position[:, :3])
            plt.show()
            raise RuntimeError("Warning: Xsens not well exported, see graph of the pelvis position")

        sync_output_save_name = f"{out_path}/{subject_name}/{movie_name}__sync.png"
        (
            xsens_start_of_jump_index,
            xsens_end_of_jump_index,
            xsens_start_of_move_index,
            xsens_end_of_move_index,
            time_vector_xsens,
            time_vector_pupil_offset,
            csv_eye_tracking_confident_synced,
            csv_blinks
        ) = sync_jump(
            Xsens_sensorFreeAcceleration,
            start_of_jump_index,
            end_of_jump_index,
            start_of_move_index,
            end_of_move_index,
            FLAG_SYNCHRO_PLOTS,
            sync_output_save_name,
            csv_eye_tracking_confident,
            csv_blinks,
            Xsens_ms,
            max_threshold,
            air_time_threshold,
            Xsens_jump_idx,
            Pupil_jump_idx,
        )

        pelvis_resting_frames = np.arange(Xsens_frames_zero[0], Xsens_frames_zero[1])
        Xsens_position_rotated, Xsens_orientation_rotated = rotate_pelvis_to_initial_orientation(num_joints, move_orientation, Xsens_position, Xsens_orientation, pelvis_resting_frames)

        (
            time_vector_pupil_per_move,
            Xsens_orientation_per_move,
            Xsens_orientation_facing_front_wall_per_move,
            Xsens_position_rotated_per_move,
            Xsens_position_facing_front_wall_per_move,
            Xsens_jointAngle_per_move,
            Xsens_CoM_per_move,
            elevation_per_move,
            azimuth_per_move,
        ) = get_data_at_same_timestamps(
            Xsens_orientation_rotated,
            Xsens_position_rotated,
            Xsens_jointAngle,
            xsens_start_of_move_index,
            xsens_end_of_move_index,
            time_vector_xsens,
            start_of_move_index,
            end_of_move_index,
            time_vector_pupil_offset,
            csv_eye_tracking_confident_synced,
            time_stamps_eye_tracking_index_on_pupil,
            Xsens_centerOfMass,
            SCENE_CAMERA_SERIAL_NUMBER,
            API_KEY,
            num_joints,
            move_orientation,
            FLAG_PUPIL_ANGLES_PLOT,
        )

        Xsens_position_no_level_CoM_corrected_rotated_per_move, CoM_trajectory_per_move = CoM_transfo(
            time_vector_pupil_per_move, Xsens_position_rotated_per_move, Xsens_CoM_per_move, num_joints, hip_height, FLAG_COM_PLOTS
        )
        Xsens_position_facing_front_wall_no_level_CoM_corrected_rotated_per_move, _= CoM_transfo(
            time_vector_pupil_per_move, Xsens_position_facing_front_wall_per_move, Xsens_CoM_per_move, num_joints, hip_height, FLAG_COM_PLOTS
        )
        
        for j in range(len(Xsens_position_rotated_per_move)):
            print(f'Analysis of {move_names[j]} {repetition_number[j]}')
            if '/' in move_names[j]:
                move_surname = move_names[j][:-1]
            else:
                move_surname = move_names[j]

            if not os.path.exists(home_path + f"/disk/Eye-tracking/Results/{subject_name}"):
                os.makedirs(home_path + f"/disk/Eye-tracking/Results/{subject_name}")
            if not os.path.exists(home_path + f"/disk/Eye-tracking/Results/{subject_name}/{move_surname}"):
                os.makedirs(home_path + f"/disk/Eye-tracking/Results/{subject_name}/{move_surname}")

            folder_name = (
                    f"{out_path}/{subject_name}/{move_surname}/{movie_name}__{move_surname}__{repetition_number[j]}"
            )
            output_file_name = (
                    f"{out_path}/{subject_name}/{move_surname}/{movie_name}__{move_surname}__{repetition_number[j]}.mp4"
            )

            if GENERATE_STICK_FIGURE_FOR_GRAPHS:
                generate_stick_figure(
                    Xsens_orientation_per_move[j],
                    Xsens_position_no_level_CoM_corrected_rotated_per_move[j],
                    links,
                    move_surname,
                    repetition_number[j],
                )

            (number_of_fixation,
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
            blinks_index,
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
            ) = animate(
                    time_vector_pupil_per_move[j],
                    Xsens_orientation_per_move[j],
                    Xsens_orientation_facing_front_wall_per_move[j],
                    Xsens_position_no_level_CoM_corrected_rotated_per_move[j],
                    Xsens_position_facing_front_wall_no_level_CoM_corrected_rotated_per_move[j],
                    CoM_trajectory_per_move[j],
                    elevation_per_move[j],
                    azimuth_per_move[j],
                    eye_position_height,
                    eye_position_depth,
                    links,
                    num_joints,
                    csv_blinks,
                    output_file_name,
                    folder_name,
                    0,
                    blink_duration_threshold,
                    FLAG_ANIMAITON,
                    FLAG_GAZE_TRAJECTORY,
                    FLAG_GENERATE_STATS_METRICS,
                    FLAG_PUPIL_ANGLES_PLOT,
                )

            # Save the data in a dictionnary for the stats analysis and plots.
            move_summary = {"subject_expertise": subject_expertise,
            "subject_name": subject_name,
            "number_of_fixation" : number_of_fixation,
            "fixation_duration_absolute" : fixation_duration_absolute,
            "fixation_duration_relative" : fixation_duration_relative,
            "fixation_positions" : fixation_positions,
            "fixation_timing" : fixation_timing,
            "fixation_index" : fixation_index,
            "quiet_eye_duration_absolute" : quiet_eye_duration_absolute,
            "quiet_eye_duration_relative" : quiet_eye_duration_relative,
            "gaze_position_temporal_evolution_projected" : gaze_position_temporal_evolution_projected,
            "gaze_position_temporal_evolution_projected_facing_front_wall" : gaze_position_temporal_evolution_projected_facing_front_wall,
            "position_threshold_block" : position_threshold_block,
            "wall_index_block" : wall_index_block,
            "move_orientation": move_orientation[j],
            "neck_amplitude" : neck_amplitude,
            "eye_amplitude" : eye_amplitude,
            "max_neck_amplitude" : max_neck_amplitude,
            "max_eye_amplitude" : max_eye_amplitude,
            "neck_amplitude_percentile" : neck_amplitude_percentile,
            "eye_amplitude_percentile" : eye_amplitude_percentile,
            "pourcentage_anticipatory" : pourcentage_anticipatory,
            "pourcentage_compensatory" : pourcentage_compensatory,
            "pourcentage_spotting" : pourcentage_spotting,
            "pourcentage_movement_detection" : pourcentage_movement_detection,
            "pourcentage_blinks" : pourcentage_blinks,
            "anticipatory_index" : anticipatory_index,
            "compensatory_index" : compensatory_index,
            "spotting_index" : spotting_index,
            "movement_detection_index" : movement_detection_index,
            "blinks_index" : blinks_index,
            "Xsens_head_position_calculated" : Xsens_head_position_calculated,
            "eye_position" : eye_position,
            "gaze_orientation" : gaze_orientation,
            "wall_index" : wall_index,
            "wall_index_facing_front_wall": wall_index_facing_front_wall,
            "EulAngles_head_global" : EulAngles_head_global,
            "EulAngles_neck" : EulAngles_neck,
            "Xsens_orthogonal_thorax_position" : Xsens_orthogonal_thorax_position,
            "Xsens_orthogonal_head_position" : Xsens_orthogonal_head_position,
            "trampoline_bed_proportions" : move_summary_heatmaps[j]["trampoline_bed_proportions"],
            "trampoline_proportions" : move_summary_heatmaps[j]["trampoline_proportions"],
            "wall_front_proportions" : move_summary_heatmaps[j]["wall_front_proportions"],
            "wall_back_proportions" : move_summary_heatmaps[j]["wall_back_proportions"],
            "ceiling_proportions" : move_summary_heatmaps[j]["ceiling_proportions"],
            "side_proportions" : move_summary_heatmaps[j]["side_proportions"],
            "self_proportions" : move_summary_heatmaps[j]["self_proportions"],
            "blink_proportions": move_summary_heatmaps[j]["blink_proportions"],
            "percetile_heatmaps" : move_summary_heatmaps[j]["percetile_heatmaps"],
            "distance_heatmaps" : move_summary_heatmaps[j]["distance_heatmaps"],}

            with open(output_file_name[:-4] + "__eyetracking_metrics.pkl", 'wb') as handle:
                pickle.dump(move_summary, handle)

            plt.close("all")


# ------------------- Code beginning ------------------- #

GENERATE_HEATMAPS = True # False #
GENERATE_VIDEO_CONFIDENCE_THRESHOLD = False  # True #
FLAG_SYNCHRO_PLOTS = True # False  #
FLAG_COM_PLOTS = False  # True #
FLAG_ANIMAITON = True  # False #
FLAG_PUPIL_ANGLES_PLOT = True # False  #
FLAG_GAZE_TRAJECTORY = True  # False  #
FLAG_GENERATE_STATS_METRICS = True  # False #
FLAG_ANALYSIS = True  # False #
FLAG_TURN_ATHLETES_FOR_PGO = True
GENERATE_STICK_FIGURE_FOR_GRAPHS = False


parser = argparse.ArgumentParser("Enter Pupils API_KEY")
parser.add_argument("API_KEY", action="store", help="Pupils API_KEY")
args = parser.parse_args()
API_KEY = args.API_KEY
# API_KEY = "VPNzqxefqpunjUdzKWb3Fr9hQM368y7Q6Lqkc4KVxLHT"

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"
else:
    raise ValueError("Home path not found, please provide an appropriate path")

csv_name = home_path + "/disk/Eye-tracking/Trials_name_mapping.csv"
trial_table = np.char.split(pd.read_csv(csv_name, sep="\t").values.astype("str"), sep=",")

for i_trial in range(len(trial_table)):

    if trial_table[i_trial][0][22] != "True":
        continue

    subject_name = trial_table[i_trial][0][0]
    move_names = trial_table[i_trial][0][1].split(" ")
    repetition_number = trial_table[i_trial][0][2].split(" ")
    move_orientation = [int(x) for x in trial_table[i_trial][0][3].split(" ")]
    xsens_file_name = trial_table[i_trial][0][4]
    eye_tracking_folder = trial_table[i_trial][0][10]
    movie_name = trial_table[i_trial][0][11].replace(".", "_")
    subject_expertise = trial_table[i_trial][0][13]
    if trial_table[i_trial][0][16] != "":
        max_threshold = float(trial_table[i_trial][0][16])
    else:
        max_threshold = 2
    if trial_table[i_trial][0][17] != "":
        air_time_threshold = float(trial_table[i_trial][0][17])
    else:
        air_time_threshold = 0.25
    Xsens_jump_idx = ([int(x) for x in trial_table[i_trial][0][18].split(" ")] if trial_table[i_trial][0][18] != "" else [])
    Pupil_jump_idx = ([int(x) for x in trial_table[i_trial][0][19].split(" ")] if trial_table[i_trial][0][19] != "" else [])
    # No zero for Pupil, we decided to trust their zero since it is odd to find a personal zero for each subject
    Pupil_frames_zero = ([int(x) for x in trial_table[i_trial][0][20].split(" ")] if trial_table[i_trial][0][20] != "" else [0, 30])
    Xsens_frames_zero = ([int(x) for x in trial_table[i_trial][0][21].split(" ")] if trial_table[i_trial][0][21] != "" else [0, 30])

    print(f'Analysis of trial {xsens_file_name} started')

    points_labeled_path = home_path + "/disk/Eye-tracking/PupilData/points_labeled/"
    gaze_position_labels = points_labeled_path + movie_name + "_labeling_points.pkl"
    out_path = home_path + "/disk/Eye-tracking/Results"  # Results_20ms_threshold
    anthropo_name = (
        home_path
        + f"/disk/Eye-tracking/Xsens_measurements/{subject_name}_anthropo.csv"
    )
    eye_tracking_data_path = home_path + "/disk/Eye-tracking/PupilData/CloudExport/" + subject_name + '/' + eye_tracking_folder + "/"

    run_analysis(
        home_path,
        subject_name,
        move_names,
        repetition_number,
        move_orientation,
        xsens_file_name,
        movie_name,
        eye_tracking_data_path,
        subject_expertise,
        gaze_position_labels,
        out_path,
        anthropo_name,
        max_threshold,
        air_time_threshold,
        Xsens_jump_idx,
        Pupil_jump_idx,
        GENERATE_VIDEO_CONFIDENCE_THRESHOLD,
        GENERATE_HEATMAPS,
        FLAG_SYNCHRO_PLOTS,
        FLAG_COM_PLOTS,
        FLAG_ANIMAITON,
        FLAG_PUPIL_ANGLES_PLOT,
        FLAG_GAZE_TRAJECTORY,
        FLAG_GENERATE_STATS_METRICS,
        FLAG_ANALYSIS,
        GENERATE_STICK_FIGURE_FOR_GRAPHS,
        API_KEY,
    )

    plt.close("all")
