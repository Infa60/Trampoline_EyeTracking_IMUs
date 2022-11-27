import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def get_gaze_position_from_intersection(vector_origin, vector_end, bound_side):
    def intersection_plane_vector(vector_origin, vector_end, planes_points, planes_normal_vector):

        vector_orientation = vector_end - vector_origin
        t = (np.dot(planes_points, planes_normal_vector) - np.dot(planes_normal_vector, vector_origin)) / np.dot(
            vector_orientation, planes_normal_vector
        )
        return vector_origin + vector_orientation * np.abs(t)

    def verify_intersection_position(vector_origin, vector_end, wall_index, bound_side):
        vector_orientation = vector_end - vector_origin
        if wall_index == 0:  # trampoline
            t = (0 - vector_origin[2]) / vector_orientation[2]
        elif wall_index == 1:  # wall front
            a = (bound_side - -bound_side) / (7.360 - 7.193)
            b = bound_side - a * 7.360
            t = (b + a * vector_origin[0] - vector_origin[1]) / (vector_orientation[1] - a * vector_orientation[0])
        elif wall_index == 2:  # ceiling
            t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
        elif wall_index == 3:  # wall back
            t = (-8.881 - vector_origin[0]) / vector_orientation[0]
        elif wall_index == 4:  # bound right
            t = (-bound_side - vector_origin[1]) / vector_orientation[1]
        elif wall_index == 5:  # bound left
            t = (bound_side - vector_origin[1]) / vector_orientation[1]

        return vector_origin + vector_orientation * t

    # zero is positioned at the center of the trampoline
    planes_points = np.array(
        [
            [7.193, bound_side, 0],  # trampoline
            [7.193, bound_side, 0],  # wall front
            [7.193, bound_side, 9.4620 - 1.2192],  # ceiling
            [-8.881, bound_side, 0],  # wall back
            [7.193, bound_side, 0],  # bound right
            [7.360, -bound_side, 0],  # bound left
        ]
    )

    planes_normal_vector = np.array(
        [
            [0, 0, 1],  # trampoline
            np.cross(
                np.array([7.193, bound_side, 0]) - np.array([7.360, -bound_side, 0]), np.array([0, 0, -1])
            ).tolist(),  # wall front
            [0, 0, -1],  # ceiling
            [1, 0, 0],  # wall back
            [0, 1, 0],  # bound right
            [0, -1, 0],  # bound left
        ]
    )

    plane_bounds = [
        np.array([[-8.881, 7.360], [-bound_side, bound_side], [0, 0]]),
        np.array([[7.193, 7.360], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-8.881, 7.360], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
        np.array([[-8.881, -8.881], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-8.881, 7.193], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-8.881, 7.360], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
    ]

    intersection = []
    wall_index = None
    intersection_index = np.zeros((len(planes_points)))
    for i in range(len(planes_points)):
        current_interaction = intersection_plane_vector(
            vector_origin, vector_end, planes_points[i, :], planes_normal_vector[i, :]
        )

        if current_interaction is not None:
            bounds_bool = True
            vector_orientation = vector_end - vector_origin
            potential_gaze_orientation = current_interaction - vector_origin
            cross_condition = np.linalg.norm(np.cross(vector_orientation, potential_gaze_orientation))
            dot_condition = np.dot(vector_orientation, potential_gaze_orientation)
            if dot_condition > 0:
                if cross_condition > -0.01 and cross_condition < 0.01:
                    for i_bool in range(3):
                        if (
                            current_interaction[i_bool] > plane_bounds[i][i_bool, 0] - 1
                            and current_interaction[i_bool] < plane_bounds[i][i_bool, 1] + 1
                        ):
                            a = 1
                        else:
                            bounds_bool = False
                else:
                    bounds_bool = False
            else:
                bounds_bool = False

        if bounds_bool:
            intersection += [current_interaction]
            intersection_index[i] = 1
            wall_index = i

    if intersection_index.sum() > 1:
        bound_crossing = np.zeros((len(np.where(intersection_index == 1)[0])))
        for idx, i in enumerate(np.where(intersection_index == 1)[0]):
            for j in range(3):
                if plane_bounds[i][j, 0] - intersection[idx][j] > 0:
                    bound_crossing[idx] += np.abs(plane_bounds[i][j, 0] - intersection[idx][j])
                if plane_bounds[i][j, 1] - intersection[idx][j] < 0:
                    bound_crossing[idx] += np.abs(plane_bounds[i][j, 1] - intersection[idx][j])
            # if idx == 0:
            #     current_point_distances = np.array([np.min(np.abs(intersection[idx] - plane_bounds[i][:, 0]))])
            # else:
            #     current_point_distances = np.vstack((current_point_distances, np.min(np.abs(intersection[idx] - plane_bounds[i][:, 0]))))

        closest_index = np.argmin(bound_crossing)
        wall_index = np.where(intersection_index == 1)[0][closest_index]


        # elif intersection_index.sum() == 0:
        #     gaze_position = None
        #     print("Probleme !")
        # else:
        #     gaze_position = intersection[0]
    if wall_index is not None:
        gaze_position = verify_intersection_position(vector_origin, vector_end, wall_index, bound_side)
    else:
        gaze_position = None

    return gaze_position, wall_index


def unwrap_gaze_position(gaze_position, bound_side):
    # Wall front
    # Bound left  # Trampoline  # Bound right
    # Wall back
    # Ceiling

    if intersection_index[0] == 1:  # trampoline
        gaze_position_x_y = gaze_position[:2]
    elif intersection_index[1] == 1:  # wall front
        # wall front is not normal to the side bounds
        wall_front_vector = np.array([bound_side, 7.193, 0]) - np.array([-bound_side, 7.360, 0])
        gaze_position_2_norm = gaze_position[2]
        y_unknown = np.sqrt(gaze_position_2_norm**2 / (wall_front_vector[1] ** 2 / wall_front_vector[0] ** 2 + 1))
        x_unknown = -wall_front_vector[1] / wall_front_vector[0] * y_unknown
        gaze_position_x_y = (np.array([gaze_position[0], gaze_position[1]]) + np.array([x_unknown, y_unknown])).tolist()
    elif intersection_index[2] == 1:  # ceiling
        gaze_position_x_y = [gaze_position[0], gaze_position[1] + 9.462 + 2 * 8.881]
    elif intersection_index[2] == 1:  # wall back
        gaze_position_x_y = [gaze_position[0], gaze_position[1] - gaze_position[2]]
    elif intersection_index[2] == 1:  # bound right
        gaze_position_x_y = [gaze_position[0] + gaze_position[2], gaze_position[1]]
    elif intersection_index[2] == 1:  # bound left
        gaze_position_x_y = [gaze_position[0] - gaze_position[2], gaze_position[1]]

    return gaze_position_x_y
