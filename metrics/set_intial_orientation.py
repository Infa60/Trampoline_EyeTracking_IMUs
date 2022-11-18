import biorbd




def rotate_pelvis_to_initial_orientation(num_joints, Xsens_position, pelvis_resting_frames):
    """
    This function realigns the pelvis to be faced to the front wall at the beginig of the trial.
    """
    vect_hips = Xsens_position[15, :, :] - Xsens_position[19, :, :]
    desired_vector = np.array([0, 1, 0])
    angle_needed = np.arccos(np.dot(vect_hips, desired_vector) / (np.linalg.norm(vect_hips) * np.linalg.norm(desired_vector)))

    rotation_matrix = biorbd.Rotation.from_euler_angles(np.array([0, 0, angle_needed]), 'xyz').to_array()


    return rotated_Xsens_position


def set_initial_gaze_orientation(gaze_resting_frames):
    """
    This function identify the resting gaze orientation since the zero from Pupil is depandant on the face geometry of the subjects.
    """


    return







