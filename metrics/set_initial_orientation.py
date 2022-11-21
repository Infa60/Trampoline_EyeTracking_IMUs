import biorbd
import numpy as np





def rotate_pelvis_to_initial_orientation(num_joints, Xsens_position, Xsens_CoM, pelvis_resting_frames):
    """
    This function realigns the pelvis to be faced to the front wall at the beginig of the trial.
    """

    vect_hips = Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] - Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3]
    vect_hips_mean = np.mean(vect_hips, axis=0)
    vect_hips_mean[2] = 0
    desired_vector = np.array([0, 1, 0])
    angle_needed = np.arccos(np.dot(vect_hips_mean, desired_vector) / (np.linalg.norm(vect_hips_mean) * np.linalg.norm(desired_vector)))
    rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([0, 0, angle_needed]), 'xyz').to_array()

    Xsens_position_centered_on_CoM = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            Xsens_position_centered_on_CoM[i, 3*k : 3*(k+1)] = Xsens_position[i, 3 * k : 3 * (k + 1)] - Xsens_CoM[i, :3]

    Xsens_position_centered_on_CoM_rotated = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            Xsens_position_centered_on_CoM_rotated[i, 3*k : 3*(k+1)] = rotation_matrix @ Xsens_position_centered_on_CoM[i, 3*k : 3*(k+1)]

    Xsens_position_rotated = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            Xsens_position_rotated[i, 3*k : 3*(k+1)] = Xsens_position_centered_on_CoM_rotated[i, 3 * k : 3 * (k + 1)] + Xsens_CoM[i, :3]

    return Xsens_position_rotated


def get_initial_gaze_orientation(eye_resting_frames, azimuth, elevation):
    """
    This function identify the resting gaze orientation since the zero from Pupil is depandant on the face geometry of the subjects.
    """
    azimuth_zero = np.mean(azimuth[eye_resting_frames])
    elevation_zero = np.mean(elevation[eye_resting_frames])
    azimuth_zero_mean = np.mean(azimuth_zero)
    elevation_zero_mean = np.mean(elevation_zero)

    return azimuth_zero_mean, elevation_zero_mean







