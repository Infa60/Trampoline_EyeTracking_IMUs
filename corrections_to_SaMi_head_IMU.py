
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import biorbd
import math
import scipy.io as sio
import pickle
from IPython import embed



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

    Xsens_position = sio.loadmat(file_dir + xsens_file_name + '/' + "position.mat")["position"]
    Xsens_orientation = sio.loadmat(file_dir + xsens_file_name + '/' + "orientation_original_file_with_head_sideways.mat")["orientation"]

    num_joints = int(round(np.shape(Xsens_position)[1]) / 3)

    return Xsens_position, Xsens_orientation, links, num_joints


def Xsens_quat_to_orientation(
        Xsens_orientation,
        Xsens_position,
        eye_position_height,
        eye_position_depth,
        RotMat_head_orientation_zero,
):

    right_shoulder_position = Xsens_position[7 * 3:(7 + 1) * 3]
    left_shoulder_position = Xsens_position[11 * 3:(11 + 1) * 3]
    shoulder_vector = left_shoulder_position - right_shoulder_position

    thorax_bottom_position = Xsens_position[3*3:(3+1)*3]
    thorax_up_position = Xsens_position[4*3:(4+1)*3]
    thorax_vector = thorax_up_position - thorax_bottom_position

    perp_thorax = np.cross(shoulder_vector, thorax_vector)
    perp_thorax = perp_thorax / np.linalg.norm(perp_thorax)
    perp_thorax_points = np.hstack((thorax_up_position, thorax_up_position + perp_thorax))


    # Traditionnal computation for the orientation of the thorax ------------------------------------------------------
    Xsens_head_position_calculated = np.zeros((6,))
    Xsens_orthogonal_thorax_position = np.zeros((6,))
    Xsens_orthogonal_head_position = np.zeros((6,))
    Quat_normalized_head = Xsens_orientation[24:28] / np.linalg.norm(
        Xsens_orientation[24:28]
    )
    Quat_normalized_neck = Xsens_orientation[20:24] / np.linalg.norm(
        Xsens_orientation[20:24]
    )
    Quat_normalized_thorax = Xsens_orientation[16:20] / np.linalg.norm(
        Xsens_orientation[16:20]
    )
    Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
                                  Quat_normalized_head[3])
    Quat_neck = biorbd.Quaternion(Quat_normalized_neck[0], Quat_normalized_neck[1], Quat_normalized_neck[2],
                                  Quat_normalized_neck[3])
    Quat_thorax = biorbd.Quaternion(Quat_normalized_thorax[0], Quat_normalized_thorax[1], Quat_normalized_thorax[2],
                                    Quat_normalized_thorax[3])
    RotMat_head = biorbd.Quaternion.toMatrix(Quat_head)
    EulAngles_head_global = biorbd.Rotation_toEulerAngles(RotMat_head, 'xyz').to_array()
    RotMat_neck = biorbd.Quaternion.toMatrix(Quat_neck).to_array()
    RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()
    RotMat_thorax = biorbd.Quaternion.toMatrix(Quat_thorax).to_array()

    # -----------------------------------------------------------------------------------------------------------------
    if RotMat_head_orientation_zero is not None:
        RotMat_head = RotMat_head @ np.linalg.inv(RotMat_head_orientation_zero)
    else:
        RotMat_head = np.eye(3)
    # -----------------------------------------------------------------------------------------------------------------

    Xsens_head_position_calculated[:3] = Xsens_position[18:21]
    Xsens_head_position_calculated[3:] = (
            RotMat_head @ np.array([0, 0, 0.1]) + Xsens_position[18:21]
    )

    Xsens_orthogonal_thorax_position[:3] = Xsens_position[12:15]
    Xsens_orthogonal_thorax_position[3:] = (
            RotMat_thorax @ np.array([0.1, 0, 0]) + Xsens_position[12:15]
    )

    Xsens_orthogonal_head_position[:3] = Xsens_position[18:21]
    Xsens_orthogonal_head_position[3:] = (
            RotMat_head @ np.array([0.1, 0, 0]) + Xsens_position[18:21]
    )

    eye_position = (
            RotMat_head @ np.array([eye_position_depth, 0, eye_position_height])
            + Xsens_position[18:21]
    )
    gaze_orientation = RotMat_head @ np.array([10, 0, 0]) + eye_position

    RotMat_between = np.linalg.inv(RotMat_thorax) @ RotMat_head
    EulAngles_neck = biorbd.Rotation_toEulerAngles(
        biorbd.Rotation(RotMat_between[0, 0], RotMat_between[0, 1], RotMat_between[0, 2],
                        RotMat_between[1, 0], RotMat_between[1, 1], RotMat_between[1, 2],
                        RotMat_between[2, 0], RotMat_between[2, 1], RotMat_between[2, 2]), 'zy').to_array()

    return Xsens_head_position_calculated, eye_position, gaze_orientation, EulAngles_head_global, EulAngles_neck, Xsens_orthogonal_thorax_position, Xsens_orthogonal_head_position, perp_thorax_points


def compute_eye_related_positions(
        Xsens_orientation,
        Xsens_position,
        eye_position_height,
        eye_position_depth,
        RotMat_head_orientation_zero,
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
    perp_thorax_points = np.zeros((np.shape(Xsens_position)[0], 6))

    for i_time in range(len(Xsens_orientation)):
        (
            Xsens_head_position_calculated[i_time, :],
            eye_position[i_time, :],
            gaze_orientation[i_time, :],
            EulAngles_head_global[i_time, :],
            EulAngles_neck[i_time, :],
            Xsens_orthogonal_thorax_position[i_time, :],
            Xsens_orthogonal_head_position[i_time, :],
            perp_thorax_points[i_time, :],
        ) = Xsens_quat_to_orientation(
            Xsens_orientation[i_time, :],
            Xsens_position[i_time, :],
            eye_position_height,
            eye_position_depth,
            RotMat_head_orientation_zero,
        )

    return Xsens_head_position_calculated, eye_position, gaze_orientation, EulAngles_head_global, EulAngles_neck, Xsens_orthogonal_thorax_position, Xsens_orthogonal_head_position, perp_thorax_points


def update(
        i,
        Xsens_position,
        Xsens_orthogonal_thorax_position,
        Xsens_orthogonal_head_position,
        eye_position,
        perp_thorax_points,
        thorax_orthogonal,
        head_orthogonal,
        lines,
        line_gaze_orientation,
        line_perp_thorax,
        links,
        text,
):

    print(i)

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


    for i_line, line in enumerate(lines):
        line[0].set_data(
            np.array([Xsens_position[i, 3 * links[i_line, 0]], Xsens_position[i, 3 * links[i_line, 1]]]),
            np.array([Xsens_position[i, 3 * links[i_line, 0] + 1], Xsens_position[i, 3 * links[i_line, 1] + 1]]),
        )
        line[0].set_3d_properties(
            np.array([Xsens_position[i, 3 * links[i_line, 0] + 2], Xsens_position[i, 3 * links[i_line, 1] + 2]])
        )

        line_gaze_orientation[0][0].set_data(
            np.array([eye_position[i, 0], gaze_orientation[i, 0]]),
            np.array([eye_position[i, 1], gaze_orientation[i, 1]]),
        )
        line_gaze_orientation[0][0].set_3d_properties(np.array([eye_position[i, 2], gaze_orientation[i, 2]]))

    line_perp_thorax[0][0].set_data(
        np.array([perp_thorax_points[i, 0], perp_thorax_points[i, 3]]),
        np.array([perp_thorax_points[i, 1], perp_thorax_points[i, 4]]),
    )
    line_perp_thorax[0][0].set_3d_properties(
        np.array([perp_thorax_points[i, 2], perp_thorax_points[i, 5]])
    )

    text[0].set_text(f"{i}")

    return


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az

FIND_FRAMES_CLAP_FLAG = False

xsens_file_dir = "/home/charbie/disk/Eye-tracking/XsensData/SaMi/exports_shoulder_height/"
xsens_file_name = "SaMi_02"  # = 870  # "SaMi_03" = 1150
Xsens_position, Xsens_orientation, links, num_joints = load_xsens(xsens_file_dir, xsens_file_name)

if not FIND_FRAMES_CLAP_FLAG:
    rotation_nothing = biorbd.Rotation.fromEulerAngles(np.array([0, 0, 0]), "xyz").to_array()
    rotation_nothing_points = rotation_nothing @ np.array([1, 0, 0])

    # Head orientation during the clap, where the athelte is looking in front of her
    head_orientation_zero_frame = 870
    head_orientation_zero = Xsens_orientation[head_orientation_zero_frame, 24:28] / np.linalg.norm(Xsens_orientation[head_orientation_zero_frame, 24:28])
    quaternion_head_orientation_zero = biorbd.Quaternion(head_orientation_zero[0], head_orientation_zero[1], head_orientation_zero[2], head_orientation_zero[3])
    RotMat_head_orientation_zero = biorbd.Quaternion.toMatrix(quaternion_head_orientation_zero).to_array()

    Rotation_head_techniquement = RotMat_head_orientation_zero @ np.array([1, 0, 0])

    new_Xsens_orientation = {"orientation": np.zeros(np.shape(Xsens_orientation))}
    new_Xsens_orientation["orientation"][:, :] = Xsens_orientation[:, :]
    for i in range(len(Xsens_orientation)):

        Quat_normalized_head = Xsens_orientation[i, 24:28] / np.linalg.norm(
            Xsens_orientation[i, 24:28]
        )
        Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
                                      Quat_normalized_head[3])
        RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()

        RotMat_head = RotMat_head @ np.linalg.inv(RotMat_head_orientation_zero)
        New_quaternion_head = biorbd.Quaternion.fromMatrix(biorbd.Rotation(RotMat_head[0, 0], RotMat_head[0, 1], RotMat_head[0, 2],
                                                                           RotMat_head[1, 0], RotMat_head[1, 1], RotMat_head[1, 2],
                                                                           RotMat_head[2, 0], RotMat_head[2, 1], RotMat_head[2, 2])).to_array()
        new_Xsens_orientation["orientation"][i, 24:28] = New_quaternion_head

    sio.savemat(f"/home/charbie/disk/Eye-tracking/XsensData/SaMi/exports_shoulder_height/{xsens_file_name}/orientation.mat", new_Xsens_orientation)
else:
    RotMat_head_orientation_zero = None

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim3d([-1.5, 1.5])
ax.set_ylim3d([-1.5, 1.5])
ax.set_zlim3d([0, 2.5])


# Initialization of the figure for the animation
lines = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-k") for _ in range(len(links)-1)]
thorax_orthogonal = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-r")]
head_orthogonal = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-r")]
line_gaze_orientation = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-b")]
line_perp_thorax = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "-m")]
# line_perp_thorax_rotated = [ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "--c")]

if not FIND_FRAMES_CLAP_FLAG:
    ax.plot(np.array([0, rotation_nothing_points[0]]),
            np.array([0, rotation_nothing_points[1]]),
            np.array([0, rotation_nothing_points[2]]), '--c')
    ax.plot(np.array([0, Rotation_head_techniquement[0]]),
            np.array([0, Rotation_head_techniquement[1]]),
            np.array([0, Rotation_head_techniquement[2]]), '--k')

text = [ax.text(-1, -1, 2, "0")]

max_frame = 0
# Creating the Animation object
if max_frame == 0:
    frame_range = range(len(Xsens_position))
else:
    frame_range = range(max_frame)

# frame_range = range(0, 1000)  # Synchro clap Xsens frames


(
    Xsens_head_position_calculated,
    eye_position,
    gaze_orientation,
    EulAngles_head_global,
    EulAngles_neck,
    Xsens_orthogonal_thorax_position,
    Xsens_orthogonal_head_position,
    perp_thorax_points,
) = compute_eye_related_positions(
    Xsens_orientation,
    Xsens_position,
    0.095,
    0.059,
    RotMat_head_orientation_zero,
)

anim = animation.FuncAnimation(
    fig,
    update,
    frames=frame_range,
    fargs=(
        Xsens_position,
        Xsens_orthogonal_thorax_position,
        Xsens_orthogonal_head_position,
        eye_position,
        perp_thorax_points,
        thorax_orthogonal,
        head_orthogonal,
        lines,
        line_gaze_orientation,
        line_perp_thorax,
        links,
        text,
    ),
    blit=False,
)

anim.save(f"Rotated_head_{xsens_file_name}_rotated.mp4", fps=60, extra_args=["-vcodec", "libx264"], dpi=300)
# anim.save(f"Rotated_head_{xsens_file_name}_clap.mp4", fps=60, extra_args=["-vcodec", "libx264"], dpi=300)
plt.show()















