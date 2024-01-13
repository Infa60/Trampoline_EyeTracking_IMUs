import biorbd
import numpy as np
from IPython import embed

def rotate_xsens(Xsens_position, Xsens_orientation, rotation_matrix, num_joints):
    """
    Actually doing the rotation of the Xsens data
    """

    # rotate the xsens positions around the hip
    Xsens_position_rotated = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            Xsens_position_rotated[i, 3*k : 3*(k+1)] = rotation_matrix @ Xsens_position[i, 3*k : 3*(k+1)]

    Xsens_orientation_rotated = np.zeros(np.shape(Xsens_orientation))
    Xsens_orientation_rotated[:, :] = Xsens_orientation[:, :]
    for i_segment in range(23):
        for i in range(np.shape(Xsens_orientation)[0]):
            Quat_normalized = Xsens_orientation[i, i_segment*4:(i_segment+1)*4] / np.linalg.norm(
                Xsens_orientation[i, i_segment*4:(i_segment+1)*4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2],
                                          Quat_normalized[3])
            RotMat = biorbd.Quaternion.toMatrix(Quat).to_array()
            RotMat_rotated = rotation_matrix @ RotMat

            Quat_rotated = biorbd.Quaternion.fromMatrix(biorbd.Rotation(RotMat_rotated[0, 0], RotMat_rotated[0, 1], RotMat_rotated[0, 2],
                                                                             RotMat_rotated[1, 0], RotMat_rotated[1, 1], RotMat_rotated[1, 2],
                                                                             RotMat_rotated[2, 0], RotMat_rotated[2, 1], RotMat_rotated[2, 2])).to_array()
            Xsens_orientation_rotated[i, i_segment*4:(i_segment+1)*4] = Quat_rotated

    return Xsens_position_rotated, Xsens_orientation_rotated

def rotate_pelvis_to_initial_orientation(num_joints, move_orientation, Xsens_position, Xsens_orientation, pelvis_resting_frames):
    """
    This function realigns the pelvis to be faced to the wall at the beginig of the trial.
    """

    vect_hips = Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] - Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3]
    vect_vert = Xsens_position[pelvis_resting_frames, 0*3:(0+1)*3] - (Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] + Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3])/2
    vect_hips_mean = np.mean(vect_hips, axis=0)
    vect_vert_mean = np.mean(vect_vert, axis=0)
    vect_front_mean = np.cross(vect_hips_mean, vect_vert_mean)
    vect_hips_mean[2] = 0
    if move_orientation[0] == 1:
        desired_vector = np.array([0, -1, 0])
        print("\nMove orientation = 1 \n")
    elif move_orientation[0] == -1:
        desired_vector = np.array([0, 1, 0])
        print("\nMove orientation = -1 \n")
    else:
        print('Error: move_orientation[0] should be 1 or -1')
        embed()

    angle_needed = np.arccos(np.dot(vect_hips_mean, desired_vector) / (np.linalg.norm(vect_hips_mean) * np.linalg.norm(desired_vector)))
    print("\nvect_front_mean = ", vect_front_mean, " \n")

    rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([0, 0, angle_needed]), 'xyz').to_array()
    vec_hip_rotated = rotation_matrix @ vect_hips_mean
    residual_angle = np.arccos(np.dot(vec_hip_rotated, desired_vector) / (np.linalg.norm(vec_hip_rotated) * np.linalg.norm(desired_vector)))

    rotation_matrix_opposite_rotation = biorbd.Rotation.fromEulerAngles(np.array([0, 0, -angle_needed]), 'xyz').to_array()
    vec_hip_rotated_opposite_rotation = rotation_matrix_opposite_rotation @ vect_hips_mean
    residual_angle_opposite_rotation = np.arccos(np.dot(vec_hip_rotated_opposite_rotation, desired_vector) / (np.linalg.norm(vec_hip_rotated_opposite_rotation) * np.linalg.norm(desired_vector)))

    if np.abs(residual_angle) > np.abs(residual_angle_opposite_rotation):
        angle_needed = -angle_needed
        rotation_matrix = rotation_matrix_opposite_rotation
        vec_hip_rotated = vec_hip_rotated_opposite_rotation

    print('\n**** angle_needed : ', angle_needed*180/np.pi, 'degrees ****')
    print('vec_hip_rotated : ', vec_hip_rotated)
    print('Desired vector : ', desired_vector)
    print('\n')

    Xsens_position_rotated, Xsens_orientation_rotated = rotate_xsens(Xsens_position, Xsens_orientation, rotation_matrix, num_joints)

    # import matplotlib.pyplot as plt
    # from animate_JCS import plot_gymnasium
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plot_gymnasium(3 + 121 * 0.0254 / 2, ax)
    # for i in range(num_joints):
    #     ax.plot(Xsens_position_rotated[pelvis_resting_frames[0], 3*i], Xsens_position_rotated[pelvis_resting_frames[0], 3*i+1], Xsens_position_rotated[pelvis_resting_frames[0], 3*i+2], '.b')
    #     ax.plot(Xsens_position[pelvis_resting_frames[0], 3*i], Xsens_position[pelvis_resting_frames[0], 3*i+1], Xsens_position[pelvis_resting_frames[0], 3*i+2], '.k')
    #
    #
    # Quat_normalized_head=Xsens_orientation[pelvis_resting_frames[0], 24:28] / np.linalg.norm(
    #         Xsens_orientation[pelvis_resting_frames[0], 24:28]
    #     )
    # Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
    #                               Quat_normalized_head[3])
    # RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()
    # RotMat_head_rotated = rotation_matrix @ RotMat_head
    #
    # fake_head_orientation_vector_rotated = RotMat_head_rotated @ np.array([1, 0, 0])
    # fake_head_orientation_vector = RotMat_head @ np.array([1, 0, 0])
    #
    # ax.plot(np.array([0, fake_head_orientation_vector_rotated[0]]), np.array([0, fake_head_orientation_vector_rotated[1]]), np.array([0, fake_head_orientation_vector_rotated[2]]), '-b')
    # ax.plot(np.array([0, fake_head_orientation_vector[0]]), np.array([0, fake_head_orientation_vector[1]]), np.array([0, fake_head_orientation_vector[2]]), '-k')
    #
    # ax.plot(np.array([Xsens_position_rotated[pelvis_resting_frames[0], 15*3], Xsens_position_rotated[pelvis_resting_frames[0], 19*3]]),
    #         np.array([Xsens_position_rotated[pelvis_resting_frames[0], 15*3+1], Xsens_position_rotated[pelvis_resting_frames[0], 19*3+1]]),
    #         np.array([Xsens_position_rotated[pelvis_resting_frames[0], 15*3+2], Xsens_position_rotated[pelvis_resting_frames[0], 19*3+2]]), '-m')
    # plt.show()

    return Xsens_position_rotated, Xsens_orientation_rotated


def get_initial_gaze_orientation(eye_resting_frames, azimuth, elevation):
    """
    This function identify the resting gaze orientation since the zero from Pupil is depandant on the face geometry of the subjects.
    """
    azimuth_zero = np.mean(azimuth[eye_resting_frames])
    elevation_zero = np.mean(elevation[eye_resting_frames])
    azimuth_zero_mean = np.mean(azimuth_zero)
    elevation_zero_mean = np.mean(elevation_zero)

    return azimuth_zero_mean, elevation_zero_mean







