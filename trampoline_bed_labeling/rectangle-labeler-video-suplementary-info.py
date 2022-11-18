
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


def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, frame_counter, frames_clone, active_points, rectangle_color, number_of_points_to_label
    small_image_clone_eye_before = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
    small_image_clone_eye_before[:] = frames_clone[frame_counter][:]
    eye_frames = np.where(csv_eye_tracking[:, 4] == frame_counter)[0]

    if np.sum(active_points[frame_counter, :]) > 0:
        looking_at_trampo_bed()

    for i in range(len(eye_frames)):
        # small_image_clone_eye_before[int(csv_eye_tracking[eye_frames[i], 2] - 3):
        #                              int(csv_eye_tracking[eye_frames[i], 2] + 4), \
        #                             int(csv_eye_tracking[eye_frames[i], 1] - 3):
        #                             int(csv_eye_tracking[eye_frames[i], 1] + 4), :] = np.array([0, 255, 0])
        for iy in range(int(csv_eye_tracking[eye_frames[i], 2] - 3), int(csv_eye_tracking[eye_frames[i], 2] +4)):
            for ix in range(int(csv_eye_tracking[eye_frames[i], 1] - 3), int(csv_eye_tracking[eye_frames[i], 1] + 4)):
                cv2.circle(small_image_clone_eye_before, (ix, iy), 1, color=(0, 255, 0), thickness=-1)
    small_image_clone_eye_before = cv2.resize(small_image_clone_eye_before, (int(round(width / ratio_image)), int(round(height / ratio_image))))

    for i in range(number_of_points_to_label):
        if active_points[frame_counter, i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, frame_counter]), int(points_labels[label_keys[i]][1, frame_counter]))
            cv2.circle(small_image_clone_eye_before, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
            cv2.putText(small_image_clone_eye_before, point_label_names[i], (mouse_click_position[0]+3, mouse_click_position[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            for j in neighbors[i]:
                if active_points[frame_counter, j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, frame_counter]), int(points_labels[label_keys[j]][1, frame_counter]))
                    cv2.line(small_image_clone_eye_before, mouse_click_position, line_position, line_color, thickness=1)

    cv2.imshow(Image_name, small_image_clone_eye_before)

    return

def xy_to_rtheta(x0, x1, y0, y1):
    if (y1 - y0) == 0:
        theta = np.pi/2
    else:
        theta = np.arctan((x0 - x1) / (y1 - y0))
    r = x0 * np.cos(theta) + y0 * np.sin(theta)
    return (r, theta)


def image_treatment(*args):

    this_frame_points_labels = np.zeros((2, number_of_points_to_label))
    for i, key in enumerate(label_keys):
        if active_points[frame_counter, i] == True:
            this_frame_points_labels[:, i] = points_labels[key][:, frame_counter]

    if np.sum(active_points[frame_counter, :]) != 4:
        empty_trampo_bed_image()
    else:
        lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index = find_lines_to_search_for(this_frame_points_labels)

        if len(unique_lines_index) > 0 or len(unique_borders_index) > 0:
            lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index = discriminate_lines_vert_vs_horz(lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index)
            points = find_points_next_frame(lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index)
            if np.asarray(points).sum() == 0:
                empty_trampo_bed_image()
            else:
                distort_to_rectangle(lines_new_vert_index, lines_new_horz_index)
        else:
            empty_trampo_bed_image()

    return

def find_lines_to_search_for(this_frame_points_labels):

    threashold_r = 8
    threashold_theta = 0.1

    lines = []
    lines_rtheta = []
    lines_points_index = []
    borders = []
    borders_rtheta = []
    borders_points_index = []
    lines_index = []
    borders_index = []
    for i in range(number_of_points_to_label):
        if active_points[frame_counter, i] == True:
            x0 = int(this_frame_points_labels[0, i])
            y0 = int(this_frame_points_labels[1, i])
            for j in neighbors[i]:
                if j > i:
                    if active_points[frame_counter, j] == True:
                        x1 = int(this_frame_points_labels[0, j])
                        y1 = int(this_frame_points_labels[1, j])
                        if [i, j] in borders_pairs or [j, i] in borders_pairs:
                            print(f"Borders : {i} {j}")
                            borders += [np.array([x0, x1, y0, y1])]
                            borders_rtheta += [xy_to_rtheta(x0, x1, y0, y1)]
                            borders_points_index += [[i, j]]
                            borders_index += [lines_definitions[i, j]]
                        else:
                            print(f"Lines : {i} {j}")
                            lines += [np.array([x0, x1, y0, y1])]
                            lines_rtheta += [xy_to_rtheta(x0, x1, y0, y1)]
                            lines_points_index += [[i, j]]
                            lines_index += [lines_definitions[i, j]]
    unique_lines = []
    unique_lines_rtheta = []
    unique_lines_points_index = []
    unique_lines_index = []
    for i in range(len(lines)):
        if not unique_lines:
            unique_lines = [lines[0]]
            unique_lines_rtheta = [lines_rtheta[0]]
            unique_lines_points_index += [lines_points_index[i]]
            unique_lines_index += [lines_index[i]]
        else:
            is_unique = True
            for j in range(len(unique_lines)):
                if abs(lines_rtheta[i][0] - lines_rtheta[j][0]) < threashold_r and abs(lines_rtheta[i][1] - lines_rtheta[j][1]) < threashold_theta:
                    is_unique = False
            if is_unique:
                unique_lines += [lines[i]]
                unique_lines_rtheta += [lines_rtheta[i]]
                unique_lines_points_index += [lines_points_index[i]]
                unique_lines_index += [lines_index[i]]

    unique_borders = []
    unique_borders_rtheta = []
    unique_borders_points_index = []
    unique_borders_index = []
    for i in range(len(borders)):
        if not unique_borders:
            unique_borders = [borders[0]]
            unique_borders_rtheta = [borders_rtheta[0]]
            unique_borders_points_index += [borders_points_index[i]]
            unique_borders_index += [borders_index[i]]
        else:
            is_unique = True
            for j in range(len(unique_borders)):
                if abs(borders_rtheta[i][0] - borders_rtheta[j][0]) < threashold_r and abs(borders_rtheta[i][1] - borders_rtheta[j][1]) < threashold_theta:
                    is_unique = False
            if is_unique:
                unique_borders += [borders[i]]
                unique_borders_rtheta += [borders_rtheta[i]]
                unique_borders_points_index += [borders_points_index[i]]
                unique_borders_index += [borders_index[i]]

    return unique_lines, unique_borders, unique_lines_points_index, unique_borders_points_index, unique_lines_index, unique_borders_index


def discriminate_lines_vert_vs_horz(lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index):

    lines_new_vert = np.array([])
    lines_new_horz = np.array([])
    lines_new_vert_index = []
    lines_new_horz_index = []

    lines_borders_last_frame = lines_last_frame + borders_last_frame
    lines_borders_points_index = lines_points_index + borders_points_index
    lines_borders_index = unique_lines_index + unique_borders_index

    for i, line in enumerate(lines_borders_last_frame):

        line_rtheta = xy_to_rtheta(line[0], line[1], line[2], line[3])

        present_pair_reversed = [0, 0]
        present_pair_reversed[:] = lines_borders_points_index[i][:]
        present_pair_reversed.reverse()

        if lines_borders_points_index[i] in vert_pairs or present_pair_reversed in vert_pairs:
            if len(lines_new_vert) == 0:
                lines_new_vert = np.array([line_rtheta[0], line_rtheta[1]])
            else:
                lines_new_vert = np.vstack((lines_new_vert, np.array([line_rtheta[0], line_rtheta[1]])))
            lines_new_vert_index += [lines_borders_index[i]]
        elif lines_borders_points_index[i] in horz_pairs or present_pair_reversed in horz_pairs:
            if len(lines_new_horz) == 0:
                lines_new_horz = np.array([line_rtheta[0], line_rtheta[1]])
            else:
                lines_new_horz = np.vstack((lines_new_horz, np.array([line_rtheta[0], line_rtheta[1]])))
            lines_new_horz_index += [lines_borders_index[i]]

    return lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index


def find_points_next_frame(lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index):

    def intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.
        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],

            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    # if len(np.shape(lines_new_vert)) > 1:
    #     lines_new_vert_order = np.argsort(np.abs(lines_new_vert[:, 0]))
    #     lines_new_vert = lines_new_vert[lines_new_vert_order]
    #
    # if len(np.shape(lines_new_horz)) > 1:
    #     lines_new_horz_order = np.argsort(np.abs(lines_new_horz[:, 0]))
    #     lines_new_horz = lines_new_horz[lines_new_horz_order]

    # Finding the intersection points of the lines
    points = []
    num_lines_new_vert = lines_new_vert.shape[0]
    num_lines_new_horz = lines_new_horz.shape[0]
    active_points[frame_counter, :] = False
    for i in range(num_lines_new_vert):
        for j in range(num_lines_new_horz):
            point = intersection(lines_new_vert[i], lines_new_horz[j])
            points.append(point)
            point_index = int(points_definitions[int(lines_new_vert_index[i]), int(lines_new_horz_index[j])])
            points_labels[str(point_index)][:, frame_counter] = point[0]
            active_points[frame_counter, point_index] = True

    # Drawing the lines and points
    lines_to_plot = []
    if lines_new_horz.shape != lines_new_vert.shape:
        points = [[0, 0] for i in range(4)]
        return points
    lines = np.vstack((lines_new_horz, lines_new_vert))
    for line in lines:
        if len(line) > 0:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # cv2.line(image_lines_points, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # plt.plot(np.array([x1, x2]), np.array([y1, y2]), '-b')
            # lines_to_plot += [(x1, x2, y1, y2)]

    # for p in points:
    #     cv2.circle(image_lines_points, (p[0][0], p[0][1]), 5, 255, 2)
    #     plt.plot(p[0][0], p[0][1], '.g')

    # Displaying the results
    # cv2.imshow(Image_name_approx, image_lines_points)
    # cv2.waitKey(0)
    # plt.show()

    return points


def resize_image_for_disposition(wraped, rectangle_number):

    wraped_choped = wraped[int(rectangle_points_position_definition[rectangle_number][0, 1]): int(rectangle_points_position_definition[rectangle_number][3, 1]),
                            int(rectangle_points_position_definition[rectangle_number][0, 0]): int(rectangle_points_position_definition[rectangle_number][1, 0]),
                            :]

    # plt.figure()
    # plt.imshow(wraped_choped)
    # plt.show()

    if rectangle_number == 0:
        trampo_bed_shape_image = np.zeros(np.shape(wraped_choped), dtype=np.uint8)
        trampo_bed_shape_image[:] = wraped_choped[:]
    elif rectangle_number == 1:
        zeros_top = np.ones((428-322, 214, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
    elif rectangle_number == 2:
        zeros_bottom = np.ones((107, 214, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
    elif rectangle_number == 3:
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.hstack((zeros_left, wraped_choped))
    elif rectangle_number == 4:
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.hstack((wraped_choped, zeros_right))
    elif rectangle_number == 5:
        zeros_top = np.ones((428-322, 161, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 6:
        zeros_top = np.ones((428-322, 161, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 7:
        zeros_bottom = np.ones((107, 161, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 8:
        zeros_bottom = np.ones((107, 161, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 9:
        zeros_bottom = np.ones((107, 214, 3)) * 0.5
        zeros_top = np.ones((428 - 322, 214, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
    elif rectangle_number == 10:
        zeros_bottom = np.ones((107, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 11:
        zeros_top = np.ones((428-322, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 12:
        zeros_bottom = np.ones((107, 161, 3)) * 0.5
        zeros_top = np.ones((428 - 322, 161, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 13:
        zeros_bottom = np.ones((107, 161, 3)) * 0.5
        zeros_top = np.ones((428 - 322, 161, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 14:
        zeros_bottom = np.ones((160, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 15:
        zeros_top = np.ones((428 - 268, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 16:
        zeros_bottom = np.ones((107, 161-53, 3)) * 0.5
        zeros_top = np.ones((428-322, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 17:
        zeros_top = np.ones((428-107, 214, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
    elif rectangle_number == 18:
        zeros_bottom = np.ones((322, 214, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
    elif rectangle_number == 19:
        zeros_bottom = np.ones((268, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 20:
        zeros_top = np.ones((428 - 160, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 21:
        zeros_bottom = np.ones((160, 161-53, 3)) * 0.5
        zeros_top = np.ones((428 - 322, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 22:
        zeros_bottom = np.ones((107, 161 - 53, 3)) * 0.5
        zeros_top = np.ones((428 - 268, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 23:
        zeros_bottom = np.ones((160, 161 - 53, 3)) * 0.5
        zeros_top = np.ones((428 - 268, 161 - 53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 24:
        zeros_right = np.ones((428, 214-53, 3)) * 0.5
        trampo_bed_shape_image = np.hstack((wraped_choped, zeros_right))
    elif rectangle_number == 25:
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.hstack((zeros_left, wraped_choped))
    elif rectangle_number == 26:
        zeros_top = np.ones((428-322, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 27:
        zeros_top = np.ones((428-322, 214-161, 3)) * 0.5
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 28:
        zeros_bottom = np.ones((107, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 29:
        zeros_bottom = np.ones((107, 53, 3)) * 0.5
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 30:
        zeros_top = np.ones((428-107, 161, 3)) * 0.5
        zeros_right = np.ones((428, 214 - 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 31:
        zeros_top = np.ones((428-107, 214-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 32:
        zeros_bottom = np.ones((322, 161, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 33:
        zeros_bottom = np.ones((322, 161, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 34:
        zeros_top = np.ones((428-107, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 35:
        zeros_bottom = np.ones((322, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 36:
        zeros_bottom = np.ones((107, 53, 3)) * 0.5
        zeros_top = np.ones((428-322, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 37:
        zeros_bottom = np.ones((107, 53, 3)) * 0.5
        zeros_top = np.ones((428-322, 53, 3)) * 0.5
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 38:
        zeros_top = np.ones((428-107, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 39:
        zeros_top = np.ones((428-107, 214-161, 3)) * 0.5
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((wraped_choped, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 40:
        zeros_bottom = np.ones((322, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-53, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 41:
        zeros_bottom = np.ones((322, 214-161, 3)) * 0.5
        zeros_left = np.ones((428, 161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
    elif rectangle_number == 42:
        zeros_bottom = np.ones((268, 161-53, 3)) * 0.5
        zeros_top = np.ones((428-322, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    elif rectangle_number == 43:
        zeros_bottom = np.ones((107, 161-53, 3)) * 0.5
        zeros_top = np.ones((428-160, 161-53, 3)) * 0.5
        zeros_left = np.ones((428, 53, 3)) * 0.5
        zeros_right = np.ones((428, 214-161, 3)) * 0.5
        trampo_bed_shape_image = np.vstack((zeros_bottom, wraped_choped))
        trampo_bed_shape_image = np.vstack((trampo_bed_shape_image, zeros_top))
        trampo_bed_shape_image = np.hstack((zeros_left, trampo_bed_shape_image))
        trampo_bed_shape_image = np.hstack((trampo_bed_shape_image, zeros_right))
    else:
        print("Mauvais numéro de rectange")

    trampo_bed_shape_image = trampo_bed_shape_image.astype(np.uint8)

    return trampo_bed_shape_image

def empty_trampo_bed_image():
    global trampo_bed_shape_image
    trampo_bed_shape_image = np.ones((428, 214, 3), dtype=np.uint8) * 125
    cv2.line(trampo_bed_shape_image, (53, 0), (53, 428), (180, 180, 180), 2)
    cv2.line(trampo_bed_shape_image, (161, 0), (161, 428), (180, 180, 180), 2)
    cv2.line(trampo_bed_shape_image, (0, 107), (214, 107), (180, 180, 180), 2)
    cv2.line(trampo_bed_shape_image, (0, 322), (214, 322), (180, 180, 180), 2)
    cv2.line(trampo_bed_shape_image, (53, 160), (161, 160), (180, 180, 180), 2)
    cv2.line(trampo_bed_shape_image, (53, 268), (161, 268), (180, 180, 180), 2)
    return

def distort_to_rectangle(lines_new_vert_index, lines_new_horz_index):
    global trampo_bed_shape_image

    def four_point_transform(image_to_distort, four_vertices_transform, position_corners_to_map):
        dst =  position_corners_to_map.astype("float32")
        M = cv2.getPerspectiveTransform(four_vertices_transform, dst)
        wraped = cv2.warpPerspective(image_to_distort, M, (width_small, height_small), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) # wraped_width, wraped_height
        wraped = np.round(wraped)
        return wraped

    def which_rectangle_is_visible(lines_new_vert_index, lines_new_horz_index):

        for rectangle_number in range(len(rectangle_lines_definitions)):
            sum_rectangle_lines = 0
            for j in range(4):
                if rectangle_lines_definitions[rectangle_number][j] in lines_new_vert_index or rectangle_lines_definitions[rectangle_number][j] in lines_new_horz_index:
                    sum_rectangle_lines += 1
            if sum_rectangle_lines == 4:
                break
        if rectangle_number == len(rectangle_lines_definitions)-1 and sum_rectangle_lines != 4:
            position_corners_to_map = None
            four_vertices_index = None
        else:
            position_corners_to_map = rectangle_points_position_definition[rectangle_number, :]
            four_vertices_index = rectangle_points_definitions[rectangle_number]

        return position_corners_to_map, four_vertices_index, rectangle_number

    position_corners_to_map, four_vertices_index, rectangle_number = which_rectangle_is_visible(lines_new_vert_index, lines_new_horz_index)

    if position_corners_to_map is None:
        return
    else:
        four_vertices = np.zeros((4, 2), dtype="float32")
        for i in range(4):
            four_vertices[i, :] = points_labels[str(int(four_vertices_index[i]))][:, frame_counter]

        min_left = np.min(four_vertices[0, :])
        max_right = np.max(four_vertices[0, :])
        min_top = np.min(four_vertices[1, :])
        max_bottom = np.max(four_vertices[1, :])

        eye_frames = np.where(csv_eye_tracking[:, 4] == frame_counter)[0]
        for i in range(len(eye_frames)):
            image_clone_eye = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
            image_clone_eye[:] = frames_clone[frame_counter][:]

            for iy in range(int(csv_eye_tracking[eye_frames[i], 2] - 3), int(csv_eye_tracking[eye_frames[i], 2] +4)):
                for ix in range(int(csv_eye_tracking[eye_frames[i], 1] - 3), int(csv_eye_tracking[eye_frames[i], 1] + 4)):
                    cv2.circle(image_clone_eye, (ix, iy), 1, color=(0, 255, 0), thickness=-1)
            small_image_clone_eye = cv2.resize(image_clone_eye, (int(round(width / ratio_image)), int(round(height / ratio_image))))

            image_to_distort = np.zeros(np.shape(small_image_clone_eye), dtype=np.uint8)
            image_to_distort[:] = small_image_clone_eye[:]

            four_vertices_transform = np.zeros((4, 2))
            four_vertices_transform[:, :] = four_vertices[:, :]
            size_width = width_small
            size_height = height_small
            image_corners_position = np.array([[0, 0], [size_width, 0], [size_width, size_height], [0, size_height]])
            if min_left < 0:
                missing_pixels = int(abs(min_left))
                zeros_left = np.ones((missing_pixels, size_width, 3)) * 0.5
                image_to_distort = np.hstack((zeros_left, image_to_distort))
                size_height += missing_pixels
                four_vertices_transform[:, 0] += missing_pixels
                image_corners_position[: ,0] += missing_pixels
            if max_right > width_small:
                missing_pixels = int(max_right - width_small)
                zeros_right = np.ones((missing_pixels, size_width, 3)) * 0.5
                image_to_distort = np.hstack((image_to_distort, zeros_right))
                size_height += missing_pixels
            if min_top < 0:
                missing_pixels = int(abs(min_top))
                zeros_top = np.ones((missing_pixels, size_height, 3)) * 0.5
                image_to_distort = np.vstack((zeros_top, image_to_distort))
                size_width += missing_pixels
                four_vertices_transform[:, 1] += missing_pixels
                image_corners_position[:, 1] += missing_pixels
            if max_bottom > height_small:
                missing_pixels = int(max_bottom - height_small)
                zeros_bottom = np.ones((missing_pixels, size_height, 3)) * 0.5
                image_to_distort = np.vstack((image_to_distort, zeros_bottom))
                size_width += missing_pixels

            four_vertices_transform = four_vertices_transform.astype(np.float32)
            wraped = four_point_transform(image_to_distort, four_vertices_transform, position_corners_to_map)
            mask = cv2.inRange(wraped, (0, 255, 0), (0, 255, 0))
            if len(np.where(mask != 0)[0]) > 0:
                fixation_region_y, fixation_region_x = np.where(mask != 0)
                fixation_pixel = np.round(np.array([np.mean(fixation_region_x), np.mean(fixation_region_y)]))
                csv_eye_tracking[eye_frames[i], 5] = fixation_pixel[0]
                csv_eye_tracking[eye_frames[i], 6] = fixation_pixel[1]
            else:
                fixation_pixel = None
            trampo_bed_shape_image = resize_image_for_disposition(wraped, rectangle_number)


            print("four_vertices_transform : ", four_vertices_transform)

        for i in range(len(eye_frames)):
            cv2.circle(trampo_bed_shape_image, (int(csv_eye_tracking[eye_frames[i], 5]), int(csv_eye_tracking[eye_frames[i], 6])), 3, color=(0, 255, 255), thickness=-1)

        cv2.line(trampo_bed_shape_image, (53, 0), (53, 428), (180, 180, 180), 2)
        cv2.line(trampo_bed_shape_image, (161, 0), (161, 428), (180, 180, 180), 2)
        cv2.line(trampo_bed_shape_image, (0, 107), (214, 107), (180, 180, 180), 2)
        cv2.line(trampo_bed_shape_image, (0, 322), (214, 322), (180, 180, 180), 2)
        cv2.line(trampo_bed_shape_image, (53, 160), (161, 160), (180, 180, 180), 2)
        cv2.line(trampo_bed_shape_image, (53, 268), (161, 268), (180, 180, 180), 2)
        return

def mouse_click(event, x, y, flags, param):
    global points_labels, current_click

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]][:, frame_counter] = np.array([x, y])
        draw_points_and_lines()
    return

def put_text():
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (53, 214)
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    if curent_AOI_label["Wall front"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Wall front', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Wall back"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Wall back', org, font,fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Wall right"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Wall right', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Wall left"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Wall left', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Self"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Self', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Ceiling"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, 'Ceiling', org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Not an acrobatics"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, "Not an acrobatics", org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Trampoline"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, "Trampoline", org, font, fontScale, color, thickness, cv2.LINE_AA)
    elif curent_AOI_label["Jump"][frame_counter] == 1:
        cv2.putText(trampo_bed_shape_image, "Jump", org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.putText(trampo_bed_shape_image, "0", (0+3, 0+13), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "1", (53+3, 0+13), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "2", (161-22, 0+13), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "3", (214-22, 0+13), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "4", (0+3, 107-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "5", (53+3, 107-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "6", (161-22, 107-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "7", (214-22, 107-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "8", (53+3, 160-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "9", (161-22, 160-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "q", (53+3, 268-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "w", (161-22, 268-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "e", (0+3, 322-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "r", (53+3, 322-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "t", (161-22, 322-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "y", (214-22, 322-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "u", (0+3, 428-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "i", (53+3, 428-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "o", (161-22, 428-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)
    cv2.putText(trampo_bed_shape_image, "p", (214-22, 428-3), font, fontScale, (180, 180, 180), thickness, cv2.LINE_AA)

    cv2.imshow("Distorted", trampo_bed_shape_image)
    return

def looking_at_wall_front(*args):
    curent_AOI_label["Wall front"][frame_counter:] = 1
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_wall_back(*args):
    curent_AOI_label["Wall back"][frame_counter:] = 1
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_wall_right(*args):
    curent_AOI_label["Wall right"][frame_counter:] = 1
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_wall_left(*args):
    curent_AOI_label["Wall left"][frame_counter:] = 1
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_self(*args):
    curent_AOI_label["Self"][frame_counter:] = 1
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_ceiling(*args):
    curent_AOI_label["Ceiling"][frame_counter:] = 1
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_trampo_bed(*args):
    curent_AOI_label["Trampoline bed"][frame_counter] = 1
    curent_AOI_label["Wall back"][frame_counter] = 0
    curent_AOI_label["Wall front"][frame_counter] = 0
    curent_AOI_label["Wall right"][frame_counter] = 0
    curent_AOI_label["Wall left"][frame_counter] = 0
    curent_AOI_label["Self"][frame_counter] = 0
    curent_AOI_label["Ceiling"][frame_counter] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter] = 0
    curent_AOI_label["Trampoline"][frame_counter] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_trampo(*args):
    curent_AOI_label["Trampoline"][frame_counter:] = 1
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_not_an_acrobatics(*args):
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 1
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Jump"][frame_counter:] = 0
    return

def looking_at_jump(*args):
    curent_AOI_label["Jump"][frame_counter:] = 1
    curent_AOI_label["Trampoline"][frame_counter:] = 0
    curent_AOI_label["Wall back"][frame_counter:] = 0
    curent_AOI_label["Wall front"][frame_counter:] = 0
    curent_AOI_label["Wall right"][frame_counter:] = 0
    curent_AOI_label["Wall left"][frame_counter:] = 0
    curent_AOI_label["Self"][frame_counter:] = 0
    curent_AOI_label["Ceiling"][frame_counter:] = 0
    curent_AOI_label["Trampoline bed"][frame_counter:] = 0
    curent_AOI_label["Not an acrobatics"][frame_counter:] = 0
    return

def point_choice(*args):
    global current_click
    looking_at_trampo_bed()
    num_point = args[1]
    current_click = num_point
    if active_points[frame_counter, num_point]:
        active_points[frame_counter, num_point] = False
    else:
        active_points[frame_counter, num_point] = True
    draw_points_and_lines()
    return

############################### code beginning #######################################################################
global image_clone, small_image, number_of_points_to_label, width_small, height_small, frame_counter, label_keys, points_labels, frames_clone
global ratio_image, Image_name, borders_points, curent_AOI_label, csv_eye_tracking, point_label_names

circle_radius = 5
line_color = (1, 1, 1)
number_of_points_to_label = 20
circle_colors = sns.color_palette(palette="viridis", n_colors=number_of_points_to_label)
for i in range(number_of_points_to_label):
    col_0 = int(circle_colors[i][0] * 255)
    col_1 = int(circle_colors[i][1] * 255)
    col_2 = int(circle_colors[i][2] * 255)
    circle_colors[i] = (col_0, col_1, col_2)

Image_name = "Video"
Trackbar_name = "Frames"
ratio_image = 1.5

# $$$$$$$$$$$$$$$$$$

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = "/home/user/disk/Eye-tracking/PupilData/CloudExport/")

movie_path = "/home/user/disk/Eye-tracking/PupilData/undistorted_videos/"
last_slash = file_path.rfind('/')
movie_name = file_path[last_slash+1 : -4].replace('.', '_')

movie_file = movie_path + movie_name + "_undistorted_images.pkl"

second_last_slash = file_path[:last_slash].rfind('/')
eye_tracking_data_path = '/home/user/disk/Eye-tracking/PupilData/CloudExport/' + file_path[second_last_slash+1:last_slash+1]
filename = eye_tracking_data_path  + 'gaze.csv'
filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
filename_info = eye_tracking_data_path + 'info.json'

with open(filename_info, 'r') as f:
  json_info = json.load(f)


# frames, num_frames = load_video_frames(movie_file)
file = open(movie_file, "rb")
frames = pickle.load(file)
num_frames = len(frames)
frames_clone = frames.copy()
frames_clone = frames.copy()

csv_read = np.char.split(pd.read_csv(filename, sep='\t').values.astype('str'), sep=',')
timestamps_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')

time_stamps_eye_tracking = np.zeros((len(timestamps_read), ))
for i in range(len(timestamps_read)):
    time_stamps_eye_tracking[i] = float(timestamps_read[i][0][2])

csv_eye_tracking = np.zeros((len(csv_read), 7))
for i in range(len(csv_read)):
    csv_eye_tracking[i, 0] = float(csv_read[i][0][2]) # timestemp
    csv_eye_tracking[i, 1] = int(round(float(csv_read[i][0][3]))) # pos_x
    csv_eye_tracking[i, 2] = int(round(float(csv_read[i][0][4]))) # pos_y
    csv_eye_tracking[i, 3] = float(csv_read[i][0][5]) # confidence
    csv_eye_tracking[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - time_stamps_eye_tracking)) # closest image timestemp

# 2 -> 0: gaze_timestamp
# 3 -> 1: norm_pos_x
# 4 -> 2: norm_pos_y
# 5 -> 3: confidence
# 4: closest image time_stamp
# 5: pos_x_bedFrame -> computed from labeling and distortion
# 6: pos_y_bedFrame -> computed from labeling and distortion

point_label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p"]
points_labels = {"0": np.zeros((2, len(frames))),
                "1": np.zeros((2, len(frames))),
                "2": np.zeros((2, len(frames))),
                "3": np.zeros((2, len(frames))),
                "4": np.zeros((2, len(frames))),
                "5": np.zeros((2, len(frames))),
                "6": np.zeros((2, len(frames))),
                "7": np.zeros((2, len(frames))),
                "8": np.zeros((2, len(frames))),
                "9": np.zeros((2, len(frames))),
                "10": np.zeros((2, len(frames))),
                "11": np.zeros((2, len(frames))),
                "12": np.zeros((2, len(frames))),
                "13": np.zeros((2, len(frames))),
                "14": np.zeros((2, len(frames))),
                "15": np.zeros((2, len(frames))),
                "16": np.zeros((2, len(frames))),
                "17": np.zeros((2, len(frames))),
                "18": np.zeros((2, len(frames))),
                "19": np.zeros((2, len(frames)))}

curent_AOI_label = {"Trampoline": np.zeros((len(frames), )),
                "Trampoline bed": np.zeros((len(frames), )),
                "Wall front": np.zeros((len(frames), )),
                "Wall back": np.zeros((len(frames), )),
                "Wall right": np.zeros((len(frames), )),
                "Wall left": np.zeros((len(frames), )),
                "Self": np.zeros((len(frames), )),
                "Ceiling": np.zeros((len(frames), )),
                "Not an acrobatics": np.ones((len(frames), )),
                "Jump": np.ones((len(frames), ))}
label_keys = [key for key in points_labels.keys()]
current_click = 0
active_points = np.zeros((num_frames, number_of_points_to_label))
neighbors = [[1, 2, 3, 4, 12, 16],  # 0
             [0, 2, 3, 5, 8, 10, 13, 17],  # 1
             [0, 1, 3, 6, 9, 11, 14, 18],  # 2
             [0, 1, 2, 7, 15, 19],  # 3
             [5, 6, 7, 0, 12, 16],  # 4
             [4, 6, 7, 1, 8, 10, 13, 17],  # 5
             [4, 5, 7, 2, 9, 11, 14, 18],  # 6
             [4, 5, 6, 3, 15, 19],  # 7
             [9, 1, 5, 10, 13, 17],  # 8
             [8, 2, 6, 11, 14, 18],  # 9
             [11, 1, 5, 8, 13, 17],  # 10
             [10, 2, 6, 9, 14, 18],  # 11
             [13, 14, 15, 0, 4, 16],  # 12
             [12, 14, 15, 1, 5, 8, 10, 17],  # 13
             [12, 13, 15, 2, 6, 9, 11, 18],  # 14
             [12, 13, 14, 3, 7, 19],  # 15
             [17, 18, 19, 0, 4, 12],  # 16
             [16, 18, 19, 1, 5, 8, 10, 13],  # 17
             [16, 17, 19, 2, 6, 9, 11, 14],  # 18
             [16, 17, 18, 3, 7, 15]] # 19


vert_pairs = [[0, 4],
             [0, 12],
             [0, 16],
             [4, 12],
             [4, 16],
             [12, 16],
             [1, 5],
             [1, 8],
             [1, 10],
             [1, 13],
             [1, 17],
             [5, 8],
             [5, 10],
             [5, 13],
             [5, 17],
             [8, 10],
             [8, 13],
             [8, 17],
             [10, 13],
             [10, 17],
             [13, 17],
             [2, 6],
             [2, 9],
             [2, 11],
             [2, 14],
             [2, 18],
             [6, 9],
             [6, 11],
             [6, 14],
             [6, 18],
             [9, 11],
             [9, 14],
             [9, 18],
             [11, 14],
             [11, 18],
             [14, 18],
             [3, 7],
             [3, 15],
             [3, 19],
             [7, 15],
             [7, 19],
             [15, 19]]

horz_pairs = [[0, 1],
             [0, 2],
             [0, 3],
             [1, 2],
             [1, 3],
             [2, 3],
             [4, 5],
             [4, 6],
             [4, 7],
             [5, 6],
             [5, 7],
             [6, 7],
             [8, 9],
             [10, 11],
             [12, 13],
             [12, 14],
             [12, 15],
             [13, 14],
             [13, 15],
             [14, 15],
             [16, 17],
             [16, 18],
             [16, 19],
             [17, 18],
             [17, 19],
             [18, 19]]

borders_points = [0, 3, 16, 19]
borders_lines = [0, 1, 2, 3, 4, 7, 12, 15, 16, 17, 18, 19]


borders_pairs = [[0, 1],
                 [0, 2],
                 [0, 3],
                 [0, 4],
                 [0, 12],
                 [0, 16],
                 [4, 12],
                 [4, 16],
                 [12, 16],
                 [1, 2],
                 [1, 3],
                 [2, 3],
                 [3, 7],
                 [3, 15],
                 [3, 19],
                 [7, 15],
                 [7, 19],
                 [15, 19],
                 [16, 17],
                 [16, 18],
                 [16, 19],
                 [17, 18],
                 [17, 19],
                 [18, 19]]

lines_definitions = np.zeros((20, 20))
lines_definitions[:, :] = np.nan
lines_definitions[0, [1, 2, 3]] = 0
lines_definitions[[1, 2, 3], 0] = 0
lines_definitions[1, [0, 2, 3]] = 0
lines_definitions[[0, 2, 3], 1] = 0
lines_definitions[2, [0, 1, 3]] = 0
lines_definitions[[0, 1, 3], 2] = 0
lines_definitions[3, [0, 1, 2]] = 0
lines_definitions[[0, 1, 2], 3] = 0
lines_definitions[4, [5, 6, 7]] = 1
lines_definitions[[5, 6, 7], 4] = 1
lines_definitions[5, [4, 6, 7]] = 1
lines_definitions[[4, 6, 7], 5] = 1
lines_definitions[6, [4, 5, 7]] = 1
lines_definitions[[4, 5, 7], 6] = 1
lines_definitions[7, [4, 5, 6]] = 1
lines_definitions[[4, 5, 6], 7] = 1
lines_definitions[8, 9] = 2
lines_definitions[9, 8] = 2
lines_definitions[10, 11] = 3
lines_definitions[11, 10] = 3
lines_definitions[12, [13, 14, 15]] = 4
lines_definitions[[13, 14, 15], 12] = 4
lines_definitions[13, [12, 14, 15]] = 4
lines_definitions[[12, 14, 15], 13] = 4
lines_definitions[14, [12, 13, 15]] = 4
lines_definitions[[12, 13, 15], 14] = 4
lines_definitions[15, [12, 13, 14]] = 4
lines_definitions[[12, 13, 14], 15] = 4
lines_definitions[16, [17, 18, 19]] = 5
lines_definitions[[17, 18, 19], 16] = 5
lines_definitions[17, [16, 18, 19]] = 5
lines_definitions[[16, 18, 19], 17] = 5
lines_definitions[18, [16, 17, 19]] = 5
lines_definitions[[16, 17, 19], 18] = 5
lines_definitions[19, [16, 17, 18]] = 5
lines_definitions[[16, 17, 18], 19] = 5
lines_definitions[0, [4, 12, 16]] = 6
lines_definitions[[4, 12, 16], 0] = 6
lines_definitions[4, [0, 12, 16]] = 6
lines_definitions[[0, 12, 16], 4] = 6
lines_definitions[12, [0, 4, 16]] = 6
lines_definitions[[0, 4, 16], 12] = 6
lines_definitions[16, [0, 4, 12]] = 6
lines_definitions[[0, 4, 12], 16] = 6
lines_definitions[1, [5, 8, 10, 13, 17]] = 7
lines_definitions[[5, 8, 10, 13, 17], 1] = 7
lines_definitions[5, [1, 8, 10, 13, 17]] = 7
lines_definitions[[1, 8, 10, 13, 17], 5] = 7
lines_definitions[8, [1, 5, 10, 13, 17]] = 7
lines_definitions[[1, 5, 10, 13, 17], 8] = 7
lines_definitions[10, [1, 5, 8, 13, 17]] = 7
lines_definitions[[1, 5, 8, 13, 17], 10] = 7
lines_definitions[13, [1, 5, 8, 10, 17]] = 7
lines_definitions[ [1, 5, 8, 10, 17], 13] = 7
lines_definitions[17, [1, 5, 8, 10, 13]] = 7
lines_definitions[[1, 5, 8, 10, 13], 17] = 7
lines_definitions[2, [6, 9, 11, 14, 18]] = 8
lines_definitions[[6, 9, 11, 14, 18], 2] = 8
lines_definitions[6, [2, 9, 11, 14, 18]] = 8
lines_definitions[[2, 9, 11, 14, 18], 6] = 8
lines_definitions[9, [2, 6, 11, 14, 18]] = 8
lines_definitions[[2, 6, 11, 14, 18], 9] = 8
lines_definitions[11, [2, 6, 9, 14, 18]] = 8
lines_definitions[[2, 6, 9, 14, 18], 11] = 8
lines_definitions[14, [2, 6, 9, 11, 18]] = 8
lines_definitions[[2, 6, 9, 11, 18], 14] = 8
lines_definitions[18, [2, 6, 9, 11, 14]] = 8
lines_definitions[[2, 6, 9, 11, 14], 18] = 8
lines_definitions[3, [7, 15, 19]] = 9
lines_definitions[[7, 15, 19], 3] = 9
lines_definitions[7, [3, 15, 19]] = 9
lines_definitions[[3, 15, 19], 7] = 9
lines_definitions[15, [3, 7, 19]] = 9
lines_definitions[[3, 7, 19], 15] = 9
lines_definitions[19, [3, 7, 15]] = 9
lines_definitions[[3, 7, 15], 19] = 9

points_definitions = np.zeros((12, 12))
points_definitions[:, :] = np.nan
points_definitions[0, 6] = 0
points_definitions[6, 0] = 0
points_definitions[0, 7] = 1
points_definitions[7, 0] = 1
points_definitions[0, 8] = 2
points_definitions[8, 0] = 2
points_definitions[0, 9] = 3
points_definitions[9, 0] = 3
points_definitions[1, 6] = 4
points_definitions[6, 1] = 4
points_definitions[1, 7] = 5
points_definitions[7, 1] = 5
points_definitions[1, 8] = 6
points_definitions[8, 1] = 6
points_definitions[1, 9] = 7
points_definitions[9, 1] = 7
points_definitions[2, 7] = 8
points_definitions[7, 2] = 8
points_definitions[2, 8] = 9
points_definitions[8, 2] = 9
points_definitions[3, 7] = 10
points_definitions[7, 3] = 10
points_definitions[3, 8] = 11
points_definitions[8, 3] = 11
points_definitions[4, 6] = 12
points_definitions[6, 4] = 12
points_definitions[4, 7] = 13
points_definitions[7, 4] = 13
points_definitions[4, 8] = 14
points_definitions[8, 4] = 14
points_definitions[4, 9] = 15
points_definitions[9, 4] = 15
points_definitions[5, 6] = 16
points_definitions[6, 5] = 16
points_definitions[5, 7] = 17
points_definitions[7, 5] = 17
points_definitions[5, 8] = 18
points_definitions[8, 5] = 18
points_definitions[5, 9] = 19
points_definitions[9, 5] = 19

rectangle_points_definitions = np.zeros((44, 4))
rectangle_points_definitions[0, :] = np.array([0, 3, 19, 16])
rectangle_points_definitions[1, :] = np.array([0, 3, 15, 12])
rectangle_points_definitions[2, :] = np.array([4, 7, 19, 16])
rectangle_points_definitions[3, :] = np.array([1, 3, 19, 17])
rectangle_points_definitions[4, :] = np.array([0, 2, 18, 16])
rectangle_points_definitions[5, :] = np.array([0, 2, 14, 12])
rectangle_points_definitions[6, :] = np.array([1, 3, 15, 13])
rectangle_points_definitions[7, :] = np.array([4, 6, 18, 16])
rectangle_points_definitions[8, :] = np.array([5, 7, 19, 17])
rectangle_points_definitions[9, :] = np.array([4, 7, 15, 12])
rectangle_points_definitions[10, :] = np.array([5, 6, 18, 17])
rectangle_points_definitions[11, :] = np.array([1, 2, 14, 13])
rectangle_points_definitions[12, :] = np.array([4, 6, 14, 12])
rectangle_points_definitions[13, :] = np.array([5, 7, 15, 13])
rectangle_points_definitions[14, :] = np.array([8, 9, 18, 17])
rectangle_points_definitions[15, :] = np.array([1, 2, 11, 10])
rectangle_points_definitions[16, :] = np.array([5, 6, 14, 13])
rectangle_points_definitions[17, :] = np.array([0, 3, 7, 4])
rectangle_points_definitions[18, :] = np.array([12, 15, 19, 16])
rectangle_points_definitions[19, :] = np.array([10, 11, 18, 17])
rectangle_points_definitions[20, :] = np.array([1, 2, 9, 8])
rectangle_points_definitions[21, :] = np.array([8, 9, 14, 13])
rectangle_points_definitions[22, :] = np.array([5, 6, 11, 10])
rectangle_points_definitions[23, :] = np.array([8, 9, 11, 10])
rectangle_points_definitions[24, :] = np.array([0, 1, 17, 16])
rectangle_points_definitions[25, :] = np.array([2, 3, 19, 18])
rectangle_points_definitions[26, :] = np.array([0, 1, 13, 12])
rectangle_points_definitions[27, :] = np.array([2, 3, 15, 14])
rectangle_points_definitions[28, :] = np.array([4, 5, 17, 16])
rectangle_points_definitions[29, :] = np.array([6, 7, 19, 18])
rectangle_points_definitions[30, :] = np.array([0, 2, 6, 4])
rectangle_points_definitions[31, :] = np.array([1, 3, 7, 5])
rectangle_points_definitions[32, :] = np.array([12, 14, 18, 16])
rectangle_points_definitions[33, :] = np.array([13, 15, 19, 17])
rectangle_points_definitions[34, :] = np.array([1, 2, 6, 5])
rectangle_points_definitions[35, :] = np.array([13, 14, 18, 17])
rectangle_points_definitions[36, :] = np.array([4, 5, 13, 12])
rectangle_points_definitions[37, :] = np.array([6, 7, 15, 14])
rectangle_points_definitions[38, :] = np.array([0, 1, 5, 4])
rectangle_points_definitions[39, :] = np.array([2, 3, 7, 6])
rectangle_points_definitions[40, :] = np.array([12, 13, 17, 16])
rectangle_points_definitions[41, :] = np.array([14, 15, 19, 18])
rectangle_points_definitions[42, :] = np.array([10, 11, 14, 13])
rectangle_points_definitions[43, :] = np.array([5, 6, 9, 8])

rectangle_lines_definitions = np.zeros((44, 4))
rectangle_lines_definitions[0, :] = np.array([0, 5, 6, 9])
rectangle_lines_definitions[1, :] = np.array([6, 9, 0, 4])
rectangle_lines_definitions[2, :] = np.array([6, 9, 1, 5])
rectangle_lines_definitions[3, :] = np.array([7, 9, 0, 5])
rectangle_lines_definitions[4, :] = np.array([6, 8, 0, 5])
rectangle_lines_definitions[5, :] = np.array([6, 8, 0, 4])
rectangle_lines_definitions[6, :] = np.array([7, 9, 0, 4])
rectangle_lines_definitions[7, :] = np.array([1, 5, 6, 8])
rectangle_lines_definitions[8, :] = np.array([1, 5, 7, 9])
rectangle_lines_definitions[9, :] = np.array([1, 4, 6, 9])
rectangle_lines_definitions[10, :] = np.array([7, 8, 1, 5])
rectangle_lines_definitions[11, :] = np.array([0, 4, 7, 8])
rectangle_lines_definitions[12, :] = np.array([6, 8, 1, 4])
rectangle_lines_definitions[13, :] = np.array([1, 4, 7, 9])
rectangle_lines_definitions[14, :] = np.array([7, 8, 2, 5])
rectangle_lines_definitions[15, :] = np.array([0, 3, 7, 8])
rectangle_lines_definitions[16, :] = np.array([1, 4, 7, 8])
rectangle_lines_definitions[17, :] = np.array([0, 1, 6, 9])
rectangle_lines_definitions[18, :] = np.array([6, 9, 4, 5])
rectangle_lines_definitions[19, :] = np.array([3, 5, 7, 8])
rectangle_lines_definitions[20, :] = np.array([7, 8, 2, 0])
rectangle_lines_definitions[21, :] = np.array([2, 4, 7, 8])
rectangle_lines_definitions[22, :] = np.array([1, 3, 7, 8])
rectangle_lines_definitions[23, :] = np.array([2, 3, 7, 8])
rectangle_lines_definitions[24, :] = np.array([6, 7, 0, 5])
rectangle_lines_definitions[25, :] = np.array([8, 9, 0, 5])
rectangle_lines_definitions[26, :] = np.array([6, 7, 0, 4])
rectangle_lines_definitions[27, :] = np.array([8, 9, 0, 4])
rectangle_lines_definitions[28, :] = np.array([6, 7, 1, 5])
rectangle_lines_definitions[29, :] = np.array([1, 5, 8, 9])
rectangle_lines_definitions[30, :] = np.array([6, 8, 0, 1])
rectangle_lines_definitions[31, :] = np.array([7, 9, 0, 1])
rectangle_lines_definitions[32, :] = np.array([4, 5, 6, 8])
rectangle_lines_definitions[33, :] = np.array([4, 5, 7, 9])
rectangle_lines_definitions[34, :] = np.array([7, 8, 0, 1])
rectangle_lines_definitions[35, :] = np.array([4, 5, 7, 8])
rectangle_lines_definitions[36, :] = np.array([1, 4, 6, 7])
rectangle_lines_definitions[37, :] = np.array([1, 4, 8, 9])
rectangle_lines_definitions[38, :] = np.array([0, 1, 6, 7])
rectangle_lines_definitions[39, :] = np.array([0, 1, 8, 9])
rectangle_lines_definitions[40, :] = np.array([4, 5, 6, 7])
rectangle_lines_definitions[41, :] = np.array([4, 5, 8, 9])
rectangle_lines_definitions[42, :] = np.array([3, 4, 7, 8])
rectangle_lines_definitions[43, :] = np.array([1, 2, 7, 8])

rectangle_points_position_definition = np.zeros((44, 4, 2))
rectangle_points_position_definition[0, :, :] = np.array([[0, 0],
                                                 [214, 0],
                                                 [214, 428],
                                                 [0, 428]])
rectangle_points_position_definition[1, :, :] = np.array([[0, 0],
                                                 [214, 0],
                                                 [214, 322],
                                                 [0, 322]])
rectangle_points_position_definition[2, :, :] = np.array([[0, 107],
                                                 [214, 107],
                                                 [214, 428],
                                                 [0, 428]])
rectangle_points_position_definition[3, :, :] = np.array([[53, 0],
                                                 [214, 0],
                                                 [214, 428],
                                                 [53, 428]])
rectangle_points_position_definition[4, :, :] = np.array([[0, 0],
                                                 [161, 0],
                                                 [161, 428],
                                                 [0, 428]])
rectangle_points_position_definition[5, :, :] = np.array([[0, 0],
                                                 [161, 0],
                                                 [161, 322],
                                                 [0, 322]])
rectangle_points_position_definition[6, :, :] = np.array([[53, 0],
                                                 [214, 0],
                                                 [214, 322],
                                                 [53, 322]])
rectangle_points_position_definition[7, :, :] = np.array([[0, 107],
                                                 [161, 107],
                                                 [161, 428],
                                                 [0, 428]])
rectangle_points_position_definition[8, :, :] = np.array([[53, 107],
                                                 [214, 107],
                                                 [214, 428],
                                                 [53, 428]])
rectangle_points_position_definition[9, :, :] = np.array([[0, 107],
                                                 [214, 107],
                                                 [214, 322],
                                                 [0, 322]])
rectangle_points_position_definition[10, :, :] = np.array([[53, 107],
                                                  [161, 107],
                                                  [161, 428],
                                                  [53, 428]])
rectangle_points_position_definition[11, :, :] = np.array([[53, 0],
                                                  [161, 0],
                                                  [161, 322],
                                                  [53, 322]])
rectangle_points_position_definition[12, :, :] = np.array([[0, 107],
                                                  [161, 107],
                                                  [161, 322],
                                                  [0, 322]])
rectangle_points_position_definition[13, :, :] = np.array([[53, 107],
                                                  [214, 107],
                                                  [214, 322],
                                                  [53, 322]])
rectangle_points_position_definition[14, :, :] = np.array([[53, 160],
                                                  [161, 160],
                                                  [161, 428],
                                                  [53, 428]])
rectangle_points_position_definition[15, :, :] = np.array([[53, 0],
                                                  [161, 0],
                                                  [161, 268],
                                                  [53, 268]])
rectangle_points_position_definition[16, :, :] = np.array([[53, 107],
                                                  [161, 107],
                                                  [161, 322],
                                                  [53, 322]])
rectangle_points_position_definition[17, :, :] = np.array([[0, 0],
                                                  [214, 0],
                                                  [214, 107],
                                                  [0, 107]])
rectangle_points_position_definition[18, :, :] = np.array([[0, 322],
                                                  [214, 322],
                                                  [214, 428],
                                                  [0, 428]])
rectangle_points_position_definition[19, :, :] = np.array([[53, 268],
                                                  [161, 268],
                                                  [161, 428],
                                                  [53, 428]])
rectangle_points_position_definition[20, :, :] = np.array([[53, 0],
                                                  [161, 0],
                                                  [161, 160],
                                                  [53, 160]])
rectangle_points_position_definition[21, :, :] = np.array([[53, 160],
                                                  [161, 160],
                                                  [161, 322],
                                                  [53, 322]])
rectangle_points_position_definition[22, :, :] = np.array([[53, 107],
                                                  [161, 107],
                                                  [161, 268],
                                                  [53, 268]])
rectangle_points_position_definition[23, :, :] = np.array([[53, 160],
                                                  [161, 160],
                                                  [161, 268],
                                                  [53, 268]])
rectangle_points_position_definition[24, :, :] = np.array([[0, 0],
                                                  [53, 0],
                                                  [53, 428],
                                                  [0, 428]])
rectangle_points_position_definition[25, :, :] = np.array([[161, 0],
                                                  [214, 0],
                                                  [214, 428],
                                                  [161, 428]])
rectangle_points_position_definition[26, :, :] = np.array([[0, 0],
                                                  [53, 0],
                                                  [53, 322],
                                                  [0, 322]])
rectangle_points_position_definition[27, :, :] = np.array([[161, 0],
                                                  [214, 0],
                                                  [214, 322],
                                                  [161, 322]])
rectangle_points_position_definition[28, :, :] = np.array([[0, 107],
                                                  [53, 107],
                                                  [53, 428],
                                                  [0, 428]])
rectangle_points_position_definition[29, :, :] = np.array([[161, 107],
                                                  [214, 107],
                                                  [214, 428],
                                                  [161, 428]])
rectangle_points_position_definition[30, :, :] = np.array([[0, 0],
                                                  [161, 0],
                                                  [161, 107],
                                                  [0, 107]])
rectangle_points_position_definition[31, :, :] = np.array([[53, 0],
                                                  [214, 0],
                                                  [214, 107],
                                                  [53, 107]])
rectangle_points_position_definition[32, :, :] = np.array([[0, 322],
                                                  [161, 322],
                                                  [161, 428],
                                                  [0, 428]])
rectangle_points_position_definition[33, :, :] = np.array([[53, 322],
                                                  [214, 322],
                                                  [214, 428],
                                                  [53, 428]])
rectangle_points_position_definition[34, :, :] = np.array([[53, 0],
                                                  [161, 0],
                                                  [161, 107],
                                                  [53, 107]])
rectangle_points_position_definition[35, :, :] = np.array([[53, 322],
                                                  [161, 322],
                                                  [161, 428],
                                                  [53, 428]])
rectangle_points_position_definition[36, :, :] = np.array([[0, 107],
                                                  [53, 107],
                                                  [53, 322],
                                                  [0, 322]])
rectangle_points_position_definition[37, :, :] = np.array([[161, 107],
                                                  [214, 107],
                                                  [214, 322],
                                                  [161, 322]])
rectangle_points_position_definition[38, :, :] = np.array([[0, 0],
                                                  [53, 0],
                                                  [53, 107],
                                                  [0, 107]])
rectangle_points_position_definition[39, :, :] = np.array([[161, 0],
                                                  [214, 0],
                                                  [214, 107],
                                                  [161, 107]])
rectangle_points_position_definition[40, :, :] = np.array([[0, 322],
                                                  [53, 322],
                                                  [53, 428],
                                                  [0, 428]])
rectangle_points_position_definition[41, :, :] = np.array([[161, 322],
                                                  [214, 322],
                                                  [214, 428],
                                                  [161, 428]])
rectangle_points_position_definition[42, :, :] = np.array([[53, 268],
                                                  [161, 268],
                                                  [161, 322],
                                                  [53, 322]])
rectangle_points_position_definition[43, :, :] = np.array([[53, 107],
                                                  [161, 107],
                                                  [161, 160],
                                                  [53, 160]])

def nothing(x):
    return


cv2.namedWindow(Image_name)
cv2.createTrackbar(Trackbar_name, Image_name, 0, num_frames, nothing)
frame_counter = 0
# cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
cv2.createButton("0", point_choice, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("1", point_choice, 1, cv2.QT_PUSH_BUTTON, 1)
cv2.createButton("2", point_choice, 2, cv2.QT_PUSH_BUTTON, 2)
cv2.createButton("3", point_choice, 3, cv2.QT_PUSH_BUTTON, 3)
cv2.createButton("4", point_choice, 4, cv2.QT_PUSH_BUTTON, 4)
cv2.createButton("5", point_choice, 5, cv2.QT_PUSH_BUTTON, 5)
cv2.createButton("6", point_choice, 6, cv2.QT_PUSH_BUTTON, 6)
cv2.createButton("7", point_choice, 7, cv2.QT_PUSH_BUTTON, 7)
cv2.createButton("8", point_choice, 8, cv2.QT_PUSH_BUTTON, 8)
cv2.createButton("9", point_choice, 9, cv2.QT_PUSH_BUTTON, 9)
cv2.createButton("q", point_choice, 10, cv2.QT_PUSH_BUTTON, 10)
cv2.createButton("w", point_choice, 11, cv2.QT_PUSH_BUTTON, 11)
cv2.createButton("e", point_choice, 12, cv2.QT_PUSH_BUTTON, 12)
cv2.createButton("r", point_choice, 13, cv2.QT_PUSH_BUTTON, 13)
cv2.createButton("t", point_choice, 14, cv2.QT_PUSH_BUTTON, 14)
cv2.createButton("y", point_choice, 15, cv2.QT_PUSH_BUTTON, 15)
cv2.createButton("u", point_choice, 16, cv2.QT_PUSH_BUTTON, 16)
cv2.createButton("i", point_choice, 17, cv2.QT_PUSH_BUTTON, 17)
cv2.createButton("o", point_choice, 18, cv2.QT_PUSH_BUTTON, 18)
cv2.createButton("p", point_choice, 19, cv2.QT_PUSH_BUTTON, 19)
cv2.createTrackbar("Rien", "", 0, 1, nothing)
cv2.createButton("Trampoline", looking_at_trampo, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Wall front", looking_at_wall_front, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Wall back", looking_at_wall_back, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Wall right", looking_at_wall_right, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Wall left", looking_at_wall_left, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Self", looking_at_self, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Ceiling", looking_at_ceiling, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Not an acrobatics", looking_at_not_an_acrobatics, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Jump", looking_at_jump, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.setMouseCallback(Image_name, mouse_click)


gaze_position_labels_file = "/home/user/disk/Eye-tracking/PupilData/points_labeled/" + movie_name + "_labeling_points.pkl" # [:-4]
if os.path.exists(gaze_position_labels_file):
    file = open(gaze_position_labels_file, "rb")
    points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(file)
    if "Wall right" not in curent_AOI_label.keys(): ############
        curent_AOI_label["Wall right"] = np.zeros((len(frames),))
        curent_AOI_label["Wall left"] = np.zeros((len(frames),))
        curent_AOI_label["Self"] = np.zeros((len(frames),))
    if "Jump" not in curent_AOI_label.keys():  ############
        curent_AOI_label["Jump"] = np.zeros((len(frames),))

    # $$$$$$$$$$$$$$$$$$


playVideo = True
image_clone = frames[frame_counter].copy()
width, height, rgb = np.shape(image_clone)
small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
width_small, height_small, rgb_small = np.shape(small_image)
cv2.imshow(Image_name, small_image)
while playVideo == True:

    key = cv2.waitKey(0) & 0xFF

    if key == ord('0'):
        point_choice(0, 0)
    elif key == ord('1'):
        point_choice(1, 1)
    elif key == ord('2'):
        point_choice(2, 2)
    elif key == ord('3'):
        point_choice(3, 3)
    elif key == ord('4'):
        point_choice(4, 4)
    elif key == ord('5'):
        point_choice(5, 5)
    elif key == ord('6'):
        point_choice(6, 6)
    elif key == ord('7'):
        point_choice(7, 7)
    elif key == ord('8'):
        point_choice(8, 8)
    elif key == ord('9'):
        point_choice(9, 9)
    elif key == ord('q'):
        point_choice(10, 10)
    elif key == ord('w'):
        point_choice(11, 11)
    elif key == ord('e'):
        point_choice(12, 12)
    elif key == ord('r'):
        point_choice(13, 13)
    elif key == ord('t'):
        point_choice(14, 14)
    elif key == ord('y'):
        point_choice(15, 15)
    elif key == ord('u'):
        point_choice(16, 16)
    elif key == ord('i'):
        point_choice(17, 17)
    elif key == ord('o'):
        point_choice(18, 18)
    elif key == ord('p'):
        point_choice(19, 19)

    if frame_counter % 15: # s'il ya un probleme, au moins on n'a pas tout perdu
        if not os.path.exists(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}'):
            os.makedirs(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}')
        with open(f'/home/user/disk/Eye-tracking/Results/{json_info["wearer_name"]}/{movie_name}_tempo_labeling_points.pkl', 'wb') as handle:
            pickle.dump([points_labels, active_points, curent_AOI_label, csv_eye_tracking], handle)

    # $$$$$$$$$$$$$$$$$$

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
        small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(Image_name, small_image_gray)
        draw_points_and_lines()

    elif key == ord('.'):  # if `>` then advance
        image_treatment()
        put_text()
        if frame_counter < num_frames-1:
            frame_counter += 1
        cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
        image_clone = np.zeros(np.shape(frames_clone[frame_counter]), dtype=np.uint8)
        image_clone[:] = frames_clone[frame_counter][:]
        small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
        small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(Image_name, small_image_gray)
        draw_points_and_lines()

    elif key == ord('x'):  # if `x` then quit
        playVideo = False

cv2.destroyAllWindows()

with open("/home/user/disk/Eye-tracking/PupilData/points_labeled/" + movie_name + "_labeling_points.pkl", 'wb') as handle:
    pickle.dump([points_labels, active_points, curent_AOI_label, csv_eye_tracking], handle)



# $$$$$$$$$$$$$$$$$$
