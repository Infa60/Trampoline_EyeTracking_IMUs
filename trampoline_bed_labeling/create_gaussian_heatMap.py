
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import pickle
import os
import pandas as pd
import casadi as cas
from IPython import embed
import sys
sys.path.append("../metrics")
from remove_data_during_blinks import remove_data_during_blinks_pupil, home_made_blink_confidence_threshold, remove_data_during_blinks_manual_labeling
from rectangle_labeler_video_suplementary_info import load_csv

def get_blinks(filename_timestamps, filename_info, filename_curent_label, curent_label, csv_eye_tracking):

    timestamp_image_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')
    timestamp_image = np.zeros((len(timestamp_image_read, )))
    for i in range(len(timestamp_image_read)):
        timestamp_image[i] = timestamp_image_read[i][0][2]
    info = np.char.split(pd.read_csv(filename_info, sep='\t').values.astype('str'), sep=',')
    for i in range(len(info)):
        if "scene_camera_serial_number" in info[i][0][0]:
            serial_number_str = info[i][0][0]
    num_quote = 0
    for pos, char in enumerate(serial_number_str):
        if char == '"':
            num_quote += 1
            if num_quote == 3:
                SCENE_CAMERA_SERIAL_NUMBER = serial_number_str[pos + 1:pos + 6]
                break

    last_slash = filename_curent_label.rfind('/')
    last_underscore = filename_curent_label[:filename_curent_label.rfind('_')].rfind('_')
    trial_name = filename_curent_label[last_slash + 1:last_underscore]
    blinks_labeled_file_name = filename_curent_label[:last_slash - 14] + "blinks_labeled/" + trial_name + "_labeling_blinks.pkl"

    with open(blinks_labeled_file_name, "rb") as handle:
        active_blinks, time_stamps_left_eye = pickle.load(handle)

    time_stamps_eye_tracking_index_on_pupil = np.zeros((len(timestamp_image),))
    for i in range(len(timestamp_image)):
        time_stamps_eye_tracking_index_on_pupil[i] = np.argmin(np.abs(csv_eye_tracking[:, 0] - float(timestamp_image[i])))

    # Don't mess with begining as an acrobatics, this is a labeling error, not a real behavior
    curent_label["Not an acrobatics"][0] = 1

    zeros_clusters_index = curent_label["Not an acrobatics"][:-1] - curent_label["Not an acrobatics"][1:]
    zeros_clusters_index = np.hstack((0, zeros_clusters_index))

    end_of_cluster_index_image = np.where(zeros_clusters_index == -1)[0].tolist()
    start_of_cluster_index_image = np.where(zeros_clusters_index == 1)[0].tolist()

    # If there are not the same numbers of acroabtics start and end in labeling, there is a problem
    if len(end_of_cluster_index_image) != len(start_of_cluster_index_image):
        plt.figure()
        plt.plot(zeros_clusters_index, label="Diff accrobatics")
        plt.plot(curent_label["Not an acrobatics"], '-r', label="Not an acrobatics labeled")
        plt.legend()
        plt.show()
        raise RuntimeError("Labeling problem, see graph to help fix it")
    return active_blinks, time_stamps_left_eye, start_of_cluster_index_image, end_of_cluster_index_image, time_stamps_eye_tracking_index_on_pupil, SCENE_CAMERA_SERIAL_NUMBER

def load_pupil(gaze_position_labels, eye_tracking_data_path):
    """
    Load the data from Pupil Labs for the orientation of the gaze.
    """

    with open(gaze_position_labels, "rb") as handle:
        points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(handle)

    filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
    filename_info = eye_tracking_data_path + 'info.json'

    active_blinks, time_stamps_left_eye, start_of_cluster_index_image, end_of_cluster_index_image, time_stamps_eye_tracking_index_on_pupil, SCENE_CAMERA_SERIAL_NUMBER = get_blinks(filename_timestamps, filename_info, gaze_position_labels, curent_AOI_label, csv_eye_tracking)

    start_of_move_index_image = []
    end_of_move_index_image = []
    start_of_jump_index_image = []
    end_of_jump_index_image = []
    for i in range(len(start_of_cluster_index_image)):
        if curent_AOI_label["Jump"][start_of_cluster_index_image[i] + 1] == 1:
            start_of_jump_index_image += [start_of_cluster_index_image[i]]
            end_of_jump_index_image += [end_of_cluster_index_image[i]]
        elif curent_AOI_label["Trampoline"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Trampoline bed"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Wall front"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Wall back"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Wall right"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Wall left"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Self"][start_of_cluster_index_image[i] + 1] == 1 or \
                curent_AOI_label["Ceiling"][start_of_cluster_index_image[i] + 1] == 1:
            start_of_move_index_image += [start_of_cluster_index_image[i]]
            end_of_move_index_image += [end_of_cluster_index_image[i]]

    end_of_move_index = time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image]
    start_of_move_index = time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image]
    end_of_jump_index = time_stamps_eye_tracking_index_on_pupil[end_of_jump_index_image]
    start_of_jump_index = time_stamps_eye_tracking_index_on_pupil[start_of_jump_index_image]

    return curent_AOI_label, csv_eye_tracking, active_blinks, time_stamps_left_eye, start_of_move_index, end_of_move_index, start_of_jump_index, end_of_jump_index, start_of_move_index_image, end_of_move_index_image, start_of_jump_index_image, end_of_jump_index_image, time_stamps_eye_tracking_index_on_pupil, SCENE_CAMERA_SERIAL_NUMBER


def load_pupil_jumps_only(gaze_jumps_labels, eye_tracking_data_path):
    """
    Load the data from Pupil Labs for the orientation of the gaze.
    """
    with open(gaze_jumps_labels, "rb") as handle:
        curent_jumps_label = pickle.load(handle)[0]

    filename = eye_tracking_data_path + 'gaze.csv'
    filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
    filename_info = eye_tracking_data_path + 'info.json'

    csv_read = np.char.split(pd.read_csv(filename, sep='\t').values.astype('str'), sep=',')
    timestamps_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')

    csv_eye_tracking = load_csv(csv_read, timestamps_read)

    active_blinks, time_stamps_left_eye, start_of_cluster_index_image, end_of_cluster_index_image, time_stamps_eye_tracking_index_on_pupil, SCENE_CAMERA_SERIAL_NUMBER = get_blinks(filename_timestamps, filename_info, gaze_jumps_labels, curent_jumps_label, csv_eye_tracking)

    start_of_move_index_image = []
    end_of_move_index_image = []
    start_of_jump_index_image = []
    end_of_jump_index_image = []
    for i in range(len(start_of_cluster_index_image)):
        if curent_jumps_label["Jump"][start_of_cluster_index_image[i] + 1] == 1:
            start_of_jump_index_image += [start_of_cluster_index_image[i]]
            end_of_jump_index_image += [end_of_cluster_index_image[i]]
        elif curent_jumps_label["Acrobatics"][start_of_cluster_index_image[i] + 1] == 1:
            start_of_move_index_image += [start_of_cluster_index_image[i]]
            end_of_move_index_image += [end_of_cluster_index_image[i]]

    end_of_move_index = time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image]
    start_of_move_index = time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image]
    end_of_jump_index = time_stamps_eye_tracking_index_on_pupil[end_of_jump_index_image]
    start_of_jump_index = time_stamps_eye_tracking_index_on_pupil[start_of_jump_index_image]

    return curent_jumps_label, csv_eye_tracking, active_blinks, time_stamps_left_eye, start_of_move_index, end_of_move_index, start_of_jump_index, end_of_jump_index, start_of_move_index_image, end_of_move_index_image, start_of_jump_index_image, end_of_jump_index_image, time_stamps_eye_tracking_index_on_pupil, SCENE_CAMERA_SERIAL_NUMBER


def points_to_percentile(centers):
    """
    Select the 90th percentile point.
    """
    mean_centers = np.mean(centers, axis=0)
    distance = np.linalg.norm(centers - mean_centers, axis=1)
    percentile = np.percentile(distance, 90)
    return distance, percentile

def points_to_ellipse_width(centers, ellipse_fig_name):
    """
    Optimize the parameters of an ellipse containing 90% of the heatmap points.
    Reports the width and height of the ellipse.
    """

    centers = np.array(centers)
    mean_centers = np.mean(centers, axis=0)
    distance = np.linalg.norm(centers - mean_centers, axis=1)
    percentile = np.percentile(distance, 90)
    indices_sorted = np.argsort(distance)
    first_false_index = np.where(np.logical_not(distance[indices_sorted] < percentile))[0][0]
    indices = indices_sorted[:first_false_index]

    # Generate a good intial guess for the ellipse parameters
    eigvals, eigvecs = np.linalg.eigh(np.cov(centers[:, 0], centers[:, 1]))
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta_gauss = np.arctan2(vy, vx)
    width_gauss, height_gauss = 2 * 2 * np.sqrt(eigvals)

    success_out = False
    i_loop = 0
    while success_out == False:
        # State the optimization problem with the following variables
        # Angle
        # width of the ellipse
        # height of the ellipse
        # x center of the ellipse
        # y center of the ellipse
        ellipse_param = cas.MX.sym("parameters", 5)
        x0 = np.array([theta_gauss, width_gauss, height_gauss, mean_centers[0], mean_centers[1]])

        # Objective (minimize area of the ellipse)
        f = np.pi * ellipse_param[1] * ellipse_param[2]  # ellipse air = pi * a * b

        # Constraints (all point are inside the ellipse)
        g = []
        lbg = []
        ubg = []
        for i, indices_this_time in enumerate(indices):

            cos_angle = cas.cos(np.pi-ellipse_param[0])
            sin_angle = cas.sin(np.pi-ellipse_param[0])

            xc = centers[indices_this_time, 0] - ellipse_param[3]
            yc = centers[indices_this_time, 1] - ellipse_param[4]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            rad_cc = (xct ** 2 / (ellipse_param[1] / 2.) ** 2) + (yct ** 2 / (ellipse_param[2] / 2.) ** 2)

            g += [rad_cc]

            lbg += [-cas.inf]
            ubg += [1]

        nlp = {"x": ellipse_param, "f": f, "g": cas.vertcat(*g)}
        opts = {"ipopt.print_level": 0}
        solver = cas.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=x0, lbx=[-2*np.pi, 0, 0, 0, 0], ubx=[np.pi, 214*2, 428*2, 214, 428], lbg=lbg, ubg=ubg)

        if solver.stats()['success']:
            success_out = True
            theta_opt = sol['x'][0]
            width_opt = sol['x'][1]
            height_opt = sol['x'][2]
            center_x_opt = sol['x'][3]
            center_y_opt = sol['x'][4]
        else:
            theta_gauss = theta_gauss * np.random.uniform(0.9, 1.1)
            width_gauss = width_gauss * np.random.uniform(0.9, 1.1)
            height_gauss = height_gauss * np.random.uniform(0.9, 1.1)
            print('Ellipse did not converge, trying again')
        if i_loop == 5:
            up_extreme = centers[np.where(centers[:, 1] == centers[indices, 1].max()), :][0][0]
            down_extreme = centers[np.where(centers[:, 1] == centers[indices, 1].min()), :][0][0]
            left_extreme = centers[np.where(centers[:, 0] == centers[indices, 0].min()), :][0][0]
            right_extreme = centers[np.where(centers[:, 0] == centers[indices, 0].max()), :][0][0]
            width_gauss = np.linalg.norm(up_extreme - down_extreme)
            height_gauss = np.linalg.norm(left_extreme - right_extreme)
            theta_gauss_vert = np.arccos(np.dot(up_extreme - down_extreme, np.array([0, 1])) / np.linalg.norm(up_extreme - down_extreme))
            theta_gauss_hor = np.arccos(np.dot(left_extreme - right_extreme, np.array([1, 0])) / np.linalg.norm(left_extreme - right_extreme))
            theta_gauss = np.mean([theta_gauss_vert, theta_gauss_hor])
        if i_loop > 10:
            embed()
        i_loop += 1

    # Plotting the original points and the fitted ellipse
    fig, ax = plt.subplots(1, 1)
    ax.plot(centers[:, 0], centers[:, 1], '.k')
    ax.plot(centers[indices, 0], centers[indices, 1], '.r')
    ellipse_1 = Ellipse(xy=(mean_centers[0], mean_centers[1]), width=width_gauss, height=height_gauss, angle=theta_gauss*180/np.pi, facecolor='blue', alpha=0.5)
    ellipse_2 = Ellipse(xy=(float(center_x_opt[0]), float(center_y_opt)), width=float(width_opt), height=float(height_opt), angle=float(theta_opt*180/np.pi), facecolor='green', alpha=0.5)
    ax.add_patch(ellipse_1)
    ax.add_patch(ellipse_2)
    ax.set_aspect('equal')
    plt.savefig(ellipse_fig_name, pdi=300)
    # plt.show()
    plt.close(fig)

    # Plot an example of ellipse fitting for supplementary material
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(centers[:, 0], centers[:, 1], '.', markersize=2, color='k', label='All gaze points')
    # ax.plot(centers[indices, 0], centers[indices, 1], '.', markersize=2, color='b', label='Gaze points under\n' + r'$90^{th}$ percentile distance')
    # ax.plot(mean_centers[0], mean_centers[1], 'x', color='r', label='Mean gaze point')
    # ellipse_2 = Ellipse(xy=(float(center_x_opt[0]), float(center_y_opt)), width=float(width_opt), height=float(height_opt), angle=float(theta_opt*180/np.pi), facecolor='blue', alpha=0.3, label='Fitted ellipse')
    # ax.add_patch(ellipse_2)
    # put_lines_on_fig(style='-k')
    # ax.set_aspect('equal')
    # ax.set_xlabel('X [cm]')
    # ax.set_ylabel('Y [cm]')
    # ax.set_title("Example of ellipse fitting on the \ngaze points on the trampoline bed")
    # ax.set_xlim([0, 214])
    # ax.set_ylim([0, 428])
    # ax.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', borderaxespad=0., frameon=False)
    # plt.savefig('/home/charbie/disk/Eye-tracking/plots' + '/ellipse_fitting_example.png', pdi=900)
    # # plt.show()
    return distance, width_opt, height_opt

def points_to_gaussian_heatmap(centers, height, width, scale):
    """
    Create a gaussian heatmap of the gaze points on the trampoline bed.
    The gaze points are taken from the labeled videos.
    This function is strongly insired from from : https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python
    """

    gaussians = []
    for x, y in centers:
        s = np.eye(2) * scale
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T

    # evaluate kernels at grid points
    zz = sum(g.pdf(xxyy) for g in gaussians)

    img = zz.reshape((height, width))
    return img

def put_lines_on_fig(style='-w'):
    """
    Draw the same lines as the red lines on the trampoline bed.
    """
    plt.plot(np.array([0, 214]), np.array([0, 0]), style, linewidth=1)
    plt.plot(np.array([214, 214]), np.array([214, 428]), style, linewidth=1)
    plt.plot(np.array([0, 214]), np.array([428, 428]), style, linewidth=1)
    plt.plot(np.array([0, 0]), np.array([0, 428]), style, linewidth=1)

    plt.plot(np.array([53, 53]), np.array([0, 428]), style, linewidth=1)
    plt.plot(np.array([161, 161]), np.array([0, 428]), style, linewidth=1)
    plt.plot(np.array([0, 214]), np.array([107, 107]), style, linewidth=1)
    plt.plot(np.array([0, 214]), np.array([322, 322]), style, linewidth=1)
    plt.plot(np.array([53, 161]), np.array([160, 160]), style, linewidth=1)
    plt.plot(np.array([53, 161]), np.array([268, 268]), style, linewidth=1)
    plt.plot(np.array([107 - 25, 107 + 25]), np.array([214, 214]), style, linewidth=1)
    plt.plot(np.array([107, 107]), np.array([214 - 25, 214 + 25]), style, linewidth=1)
    return

def run_create_heatmaps(subject_name, subject_expertise, move_names, move_orientation, repetition_number, movie_name, out_path, start_of_move_index_image, end_of_move_index_image, curent_AOI_label, csv_eye_tracking, gaze_position_labels):
    """
    Plots the heatmap and saves the data for later use.
    """
    if not os.path.exists(f'{out_path}/{subject_name}'):
        os.makedirs(f'{out_path}/{subject_name}')

    image_width = 214
    image_height = 428
    gaussian_width = 50

    curent_AOI_label["Blinks"] = np.zeros(np.shape(curent_AOI_label["Trampoline"]))

    if len(move_names) != len(start_of_move_index_image) or len(move_names) != len(end_of_move_index_image):
        plt.figure()
        plt.plot(curent_AOI_label["Not an acrobatics"], '-k', label="Not an acrobatics")
        plt.plot(curent_AOI_label["Jump"], '-r', label="Jump")
        plt.legend()
        plt.show()
        embed()
        raise RuntimeError(f"Not the right number of skills! "
                           f"\nlen(move_names) = {len(move_names)}"
                           f"\nlen(start_of_move_index_image) = {len(start_of_move_index_image)}"
                           f"\nlen(end_of_move_index_image) = {len(end_of_move_index_image)}")

    move_summary = [{} for i in range(len(move_names))]
    move_summary_light = [{} for i in range(len(move_names))]

    centers_gaze_bed = [[] for i in range(len(move_names))]
    percetile_heatmaps = []
    distance_heatmaps = []
    width_ellipse_heatmaps = []
    height_ellipse_heatmaps = []
    for i in range(len(move_names)):
        if not os.path.exists(f'{out_path}/{subject_name}/{move_names[i]}'):
            os.makedirs(f'{out_path}/{subject_name}/{move_names[i]}')
        start = start_of_move_index_image[i]
        end = end_of_move_index_image[i]
        centers_gaze_bed_i = []
        gaze_total_move = end - start
        number_of_trampoline_bed = 0
        number_of_trampoline = 0
        number_of_wall_front = 0
        number_of_wall_back = 0
        number_of_ceiling = 0
        number_of_side = 0
        number_of_self = 0
        number_of_blinks = 0
        for j in range(start, end):
            index_gaze = np.where(csv_eye_tracking[:, 4] == j)[0]
            if np.all(np.isnan(csv_eye_tracking[index_gaze, 1])) and np.all(np.isnan(csv_eye_tracking[index_gaze, 2])):
                number_of_blinks += 1
                curent_AOI_label["Blinks"][j] = 1
            elif curent_AOI_label["Trampoline bed"][j] == 1:
                for k in index_gaze:
                    if csv_eye_tracking[k, 5] != 0 and csv_eye_tracking[k, 6] != 0:
                        if move_orientation[i] < 0:
                            centers_gaze_bed_i.append((image_width - csv_eye_tracking[k, 5], image_height - csv_eye_tracking[k, 6]))
                        else:
                            centers_gaze_bed_i.append((csv_eye_tracking[k, 5], csv_eye_tracking[k, 6]))
                number_of_trampoline_bed += 1
            elif curent_AOI_label["Wall front"][j] == 1:
                if move_orientation[i] < 0:
                    number_of_wall_back += 1
                else:
                    number_of_wall_front += 1
            elif curent_AOI_label["Wall back"][j] == 1:
                if move_orientation[i] < 0:
                    number_of_wall_front += 1
                else:
                    number_of_wall_back += 1
            elif curent_AOI_label["Ceiling"][j] == 1:
                number_of_ceiling += 1
            elif curent_AOI_label["Trampoline"][j] == 1:
                number_of_trampoline += 1
            elif curent_AOI_label["Wall right"][j] == 1 or curent_AOI_label["Wall left"][j] == 1:
                number_of_side += 1
            elif curent_AOI_label["Self"][j] == 1:
                number_of_self += 1

        if len(centers_gaze_bed_i) > 0:
            centers_gaze_bed[i] = centers_gaze_bed_i

            plt.figure()
            put_lines_on_fig()
            img = points_to_gaussian_heatmap(centers_gaze_bed[i], image_height, image_width, gaussian_width)
            plt.imshow(img, cmap=plt.get_cmap('plasma'))
            plt.title(f"{subject_name}({subject_expertise}): {move_names[i]}")
            plt.axis('off')

            # plt.figure()
            # plt.plot(np.arange(start, end), curent_AOI_label["Trampoline bed"][start:end], label="Trampoline bed")
            # plt.plot(np.arange(start, end), curent_AOI_label["Trampoline"][start:end], label="Trampoline")
            # plt.plot(np.arange(start, end), curent_AOI_label["Wall front"][start:end], label="Wall front")
            # plt.plot(np.arange(start, end), curent_AOI_label["Wall back"][start:end], label="Wall back")
            # plt.plot(np.arange(start, end), curent_AOI_label["Wall right"][start:end], label="Wall right")
            # plt.plot(np.arange(start, end), curent_AOI_label["Wall left"][start:end], label="Wall left")
            # plt.plot(np.arange(start, end), curent_AOI_label["Self"][start:end], label="Self")
            # plt.plot(np.arange(start, end), curent_AOI_label["Blinks"][start:end], label="Blinks")
            # plt.legend()
            # plt.show()

            distance, pertentile = points_to_percentile(centers_gaze_bed[i])
            ellipse_fig_name = f"{out_path}/{subject_name}/{move_names[i]}/{movie_name}_ellipse_{repetition_number[i]}.png"
            distance, width_ellipse, height_ellipse = points_to_ellipse_width(centers_gaze_bed[i], ellipse_fig_name)
            distance_heatmaps += [distance]
            percetile_heatmaps += [pertentile]
            width_ellipse_heatmaps += [width_ellipse]
            height_ellipse_heatmaps += [height_ellipse]
        else:
            distance_heatmaps += [np.nan]
            percetile_heatmaps += [np.nan]
            width_ellipse_heatmaps += [np.nan]
            height_ellipse_heatmaps += [np.nan]

        trampoline_bed_proportions = number_of_trampoline_bed / gaze_total_move
        trampoline_proportions = number_of_trampoline / gaze_total_move
        wall_front_proportions = number_of_wall_front / gaze_total_move
        wall_back_proportions = number_of_wall_back / gaze_total_move
        ceiling_proportions = number_of_ceiling / gaze_total_move
        side_proportions = number_of_side / gaze_total_move
        self_proportions = number_of_self / gaze_total_move
        blink_proportions = number_of_blinks / gaze_total_move

        move_summary[i] = {"movement_name": move_names[i],
                           "subject_name": subject_name,
                           "movie_name": movie_name,
                           "centers": centers_gaze_bed_i,
                           "heat_map": img,
                           "trampoline_bed_proportions": trampoline_bed_proportions,
                           "trampoline_proportions": trampoline_proportions,
                           "wall_front_proportions": wall_front_proportions,
                           "wall_back_proportions": wall_back_proportions,
                           "ceiling_proportions": ceiling_proportions,
                           "side_proportions": side_proportions,
                           "self_proportions" : self_proportions,
                           "blink_proportions" : blink_proportions,
                           "percetile_heatmaps": percetile_heatmaps[i],
                           "distance_heatmaps": distance_heatmaps[i],
                           "width_ellipse_heatmaps": width_ellipse_heatmaps[i],
                           "height_ellipse_heatmaps": height_ellipse_heatmaps[i]}

        move_summary_light[i] = {"movement_name": move_names[i],
                           "subject_name": subject_name,
                           "movie_name": movie_name,
                           "centers": centers_gaze_bed_i,
                           "trampoline_bed_proportions": trampoline_bed_proportions,
                           "trampoline_proportions": trampoline_proportions,
                           "wall_front_proportions": wall_front_proportions,
                           "wall_back_proportions": wall_back_proportions,
                           "ceiling_proportions": ceiling_proportions,
                           "side_proportions": side_proportions,
                           "self_proportions" : self_proportions,
                           "blink_proportions": blink_proportions,
                           "percetile_heatmaps": percetile_heatmaps[i],
                           "distance_heatmaps": distance_heatmaps[i],
                           "width_ellipse_heatmaps": width_ellipse_heatmaps[i],
                           "height_ellipse_heatmaps": height_ellipse_heatmaps[i]}

        with open(f'{out_path}/{subject_name}/{move_names[i]}/{movie_name}_heat_map_{repetition_number[i]}.pkl', 'wb') as handle:
            pickle.dump(move_summary[i], handle)

        plt.savefig(f"{out_path}/{subject_name}/{move_names[i]}/{movie_name}_heat_map_{repetition_number[i]}.png", format="png")
        # plt.show()
        print(f"Generated heatmap {subject_name}({subject_expertise}): {move_names[i]}")

    with open(f'{gaze_position_labels[:-20]}_heat_map.pkl', 'wb') as handle:
        pickle.dump(move_summary, handle)

    return move_summary_light


def __main__():
    """
    This is the main function if you only want to create a heatmap from the labeled videos.
    """
    if os.path.exists('/home/user'):
        root_path = '/home/user'
    elif os.path.exists('/home/fbailly'):
        root_path = '/home/fbailly'
    elif os.path.exists('/home/charbie'):
        root_path = '/home/charbie'

    csv_name = root_path + "/disk/Eye-tracking/Trials_name_mapping.csv"
    csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')

    for i_trial in range(len(csv_table)):
        if csv_table[i_trial][0][22] != 'True':
            continue
        movie_path = "/home/user/disk/Eye-tracking/PupilData/points_labeled/"
        movie_name = csv_table[i_trial][0][11].replace('.', '_')
        out_path = '/home/user/disk/Eye-tracking/Results'
        subject_name = csv_table[i_trial][0][0]
        move_names = csv_table[i_trial][0][1].split(" ")
        repetition_number = csv_table[i_trial][0][2].split(" ")
        move_orientation = [int(x) for x in csv_table[i_trial][0][3].split(" ")]
        eye_tracking_folder = csv_table[i_trial][0][10]
        subject_expertise = csv_table[i_trial][0][13]
        eye_tracking_data_path = root_path + "/disk/Eye-tracking/PupilData/CloudExport/" + subject_name + '/' + eye_tracking_folder + "/"

        points_labeled_path = root_path + "/disk/Eye-tracking/PupilData/points_labeled/"
        gaze_position_labels = points_labeled_path + movie_name + "_labeling_points.pkl"

        (curent_AOI_label,
         csv_eye_tracking,
         active_blinks,
         time_stamps_left_eye,
         start_of_move_index,
         end_of_move_index,
         start_of_jump_index,
         end_of_jump_index,
         start_of_move_index_image,
         end_of_move_index_image,
         start_of_jump_index_image,
         end_of_jump_index_image,
         time_stamps_eye_tracking_index_on_pupil,
         SCENE_CAMERA_SERIAL_NUMBER, ) = load_pupil(gaze_position_labels, eye_tracking_data_path)

        csv_eye_tracking_confident = remove_data_during_blinks_manual_labeling(csv_eye_tracking, active_blinks, time_stamps_left_eye)

        run_create_heatmaps(subject_name, subject_expertise, move_names, move_orientation, repetition_number, movie_name,
                        out_path, start_of_move_index_image, end_of_move_index_image, curent_AOI_label,
                        csv_eye_tracking_confident, gaze_position_labels)
