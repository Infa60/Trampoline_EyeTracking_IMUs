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
        # elif wall_index == 1:  # wall front
        #     a = (bound_side - -bound_side) / (7.360 - 7.193)
        #     b = bound_side - a * 7.360
        #     t = (b + a * vector_origin[0] - vector_origin[1]) / (vector_orientation[1] - a * vector_orientation[0])
        elif wall_index == 1:  # wall front
            t = (7.2 - vector_origin[0]) / vector_orientation[0]
        elif wall_index == 2:  # ceiling
            t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
        # elif wall_index == 3:  # wall back
        #     t = (-8.881 - vector_origin[0]) / vector_orientation[0]
        elif wall_index == 3:  # wall back
            t = (-7.2 - vector_origin[0]) / vector_orientation[0]
        elif wall_index == 4:  # bound right
            t = (-bound_side - vector_origin[1]) / vector_orientation[1]
        elif wall_index == 5:  # bound left
            t = (bound_side - vector_origin[1]) / vector_orientation[1]

        return vector_origin + vector_orientation * t

    # zero is positioned at the center of the trampoline
    planes_points = np.array(
        # [
        #     [7.193, bound_side, 0],  # trampoline
        #     [7.193, bound_side, 0],  # wall front
        #     [7.193, bound_side, 9.4620 - 1.2192],  # ceiling
        #     [-8.881, bound_side, 0],  # wall back
        #     [7.193, bound_side, 0],  # bound right
        #     [7.360, -bound_side, 0],  # bound left
        # ]
        [
            [7.2, bound_side, 0],  # trampoline
            [7.2, bound_side, 0],  # wall front
            [7.2, bound_side, 9.4620 - 1.2192],  # ceiling
            [-7.2, bound_side, 0],  # wall back
            [7.2, bound_side, 0],  # bound right
            [7.2, -bound_side, 0],  # bound left
        ]
    )

    planes_normal_vector = np.array(
        [
            [0, 0, 1],  # trampoline
            # np.cross(
            #     np.array([7.193, bound_side, 0]) - np.array([7.360, -bound_side, 0]), np.array([0, 0, -1])
            # ).tolist(),  # wall front
            [-1, 0, 0],  # wall front
            [0, 0, -1],  # ceiling
            [1, 0, 0],  # wall back
            [0, 1, 0],  # bound right
            [0, -1, 0],  # bound left
        ]
    )

    # plane_bounds = [
    #     np.array([[-8.881, 7.360], [-bound_side, bound_side], [0, 0]]),
    #     np.array([[7.193, 7.360], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
    #     np.array([[-8.881, 7.360], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
    #     np.array([[-8.881, -8.881], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
    #     np.array([[-8.881, 7.193], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
    #     np.array([[-8.881, 7.360], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
    # ]
    plane_bounds = [
        np.array([[-7.2, 7.2], [-bound_side, bound_side], [0, 0]]),
        np.array([[7.2, 7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-7.2, 7.2], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
        np.array([[-7.2, -7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-7.2, 7.2], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
        np.array([[-7.2, 7.2], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
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
        closest_index = np.argmin(bound_crossing)
        wall_index = np.where(intersection_index == 1)[0][closest_index]

    if wall_index is not None:
        gaze_position = verify_intersection_position(vector_origin, vector_end, wall_index, bound_side)
    else:
        gaze_position = None

    return gaze_position, wall_index

