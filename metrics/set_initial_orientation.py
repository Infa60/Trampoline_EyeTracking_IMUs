import biorbd
import numpy as np
from IPython import embed

def rotate_pelvis_to_initial_orientation(num_joints, Xsens_position, Xsens_CoM, pelvis_resting_frames):
    """
    This function realigns the pelvis to be faced to the front wall at the beginig of the trial.
    """

    vect_hips = Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] - Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3]
    vect_vert = Xsens_position[pelvis_resting_frames, 0*3:(0+1)*3] - (Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] + Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3])/2
    vect_hips_mean = np.mean(vect_hips, axis=0)
    vect_vert_mean = np.mean(vect_vert, axis=0)
    vect_front_mean = np.cross(vect_hips_mean, vect_vert_mean)
    vect_hips_mean[2] = 0
    desired_vector = np.array([0, -1, 0])
    # desired_vector = np.array([1, 0, 0])
    angle_needed = np.arccos(np.dot(vect_hips_mean, desired_vector) / (np.linalg.norm(vect_hips_mean) * np.linalg.norm(desired_vector)))
    if vect_front_mean[0] < 0:
        angle_needed = -angle_needed
    rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([0, 0, angle_needed]), 'xyz').to_array()
    print('\n****angle_needed : ', angle_needed*180/np.pi, '****')
    print('Hips vector : ', vect_hips_mean)
    print('Desired vector : ', desired_vector)
    print('\n')

    Xsens_position_centered_on_CoM_rotated = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            # Xsens_position_centered_on_CoM_rotated[i, 3*k : 3*(k+1)] = rotation_matrix @ Xsens_position_centered_on_CoM[i, 3*k : 3*(k+1)]
            Xsens_position_centered_on_CoM_rotated[i, 3*k : 3*(k+1)] = rotation_matrix @ Xsens_position[i, 3*k : 3*(k+1)]

    # import matplotlib.pyplot as plt
    # from animate_JCS import plot_gymnasium
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plot_gymnasium(3 + 121 * 0.0254 / 2, ax)
    # for i in range(num_joints):
    #     ax.plot(Xsens_position_centered_on_CoM_rotated[pelvis_resting_frames[0], 3*i], Xsens_position_centered_on_CoM_rotated[pelvis_resting_frames[0], 3*i+1], Xsens_position_centered_on_CoM_rotated[pelvis_resting_frames[0], 3*i+2], '.b')
    #     ax.plot(Xsens_position[pelvis_resting_frames[0], 3*i], Xsens_position[pelvis_resting_frames[0], 3*i+1], Xsens_position[pelvis_resting_frames[0], 3*i+2], '.k')
    # ax.plot(Xsens_CoM[pelvis_resting_frames[0], 0], Xsens_CoM[pelvis_resting_frames[0], 1], Xsens_CoM[pelvis_resting_frames[0], 2], '.r')
    # plt.show()

    return Xsens_position_centered_on_CoM_rotated # Xsens_position_rotated


def get_initial_gaze_orientation(eye_resting_frames, azimuth, elevation):
    """
    This function identify the resting gaze orientation since the zero from Pupil is depandant on the face geometry of the subjects.
    """
    azimuth_zero = np.mean(azimuth[eye_resting_frames])
    elevation_zero = np.mean(elevation[eye_resting_frames])
    azimuth_zero_mean = np.mean(azimuth_zero)
    elevation_zero_mean = np.mean(elevation_zero)

    return azimuth_zero_mean, elevation_zero_mean







