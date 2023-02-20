import biorbd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import mpl_toolkits.mplot3d.axes3d as p3


def generate_stick_figure(Xsens_orientation, Xsens_position, links, move_surname, repetition_number):

    nb_frames = Xsens_position.shape[0]

    Xsens_head_position_calculated = np.zeros((nb_frames, 6))
    for frame in range(nb_frames):
        Quat_normalized_head = Xsens_orientation[frame, 24:28] / np.linalg.norm(
            Xsens_orientation[frame, 24:28]
        )
        Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
                                      Quat_normalized_head[3])
        RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()

        Xsens_head_position_calculated[frame, :3] = Xsens_position[frame, 18:21]
        Xsens_head_position_calculated[frame, 3:] = (
                RotMat_head @ np.array([0, 0, 0.1]) + Xsens_position[frame, 18:21]
        )

    snapshot_list = [0, 20, 40, 60, 80, 100]
    for snapshot in snapshot_list:
        i = round(snapshot * (nb_frames-1) / 100)

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_box_aspect(aspect = (1,1,1))

        for i_line in range(len(links)-1):
            if i_line != 17:
                ax.plot(np.array([Xsens_position[i, 3 * links[i_line, 0]], Xsens_position[i, 3 * links[i_line, 1]]]),
                        np.array([Xsens_position[i, 3 * links[i_line, 0] + 1], Xsens_position[i, 3 * links[i_line, 1] + 1]]),
                        np.array([Xsens_position[i, 3 * links[i_line, 0] + 2], Xsens_position[i, 3 * links[i_line, 1] + 2]]),
                        '-k')
        ax.plot(np.array([Xsens_head_position_calculated[i, 0], Xsens_head_position_calculated[i, 3]]),
                np.array([Xsens_head_position_calculated[i, 1], Xsens_head_position_calculated[i, 4]]),
                np.array([Xsens_head_position_calculated[i, 2], Xsens_head_position_calculated[i, 5]]),
                '-k')

        ax.set_axis_off()
        ax.view_init(elev=0, azim=-90)
        ax.set_box_aspect(aspect = (1,1,1))

        plt.savefig(f'/home/charbie/disk/Eye-tracking/plots/stick_figures/{move_surname}/SoMe_{snapshot}__{repetition_number}.svg')
    return