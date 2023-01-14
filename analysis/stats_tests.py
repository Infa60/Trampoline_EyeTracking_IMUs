
import numpy as np
import pingouin as pg
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import spm1d
from scipy.interpolate import interp1d
import csv
from IPython import embed


##########################################################################################
# run --- python stats_test.py > stats_output.txt ---  to save the output to a file
##########################################################################################


PRIMARY_ANALYSIS_FLAG = False
TRAJECTORIES_ANALYSIS_FLAG = True
AOI_ANALYSIS_FLAG = False
NECK_EYE_ANALYSIS_FLAG = False # True
SPREADING_HEATMAP_FLAG = False # True
QUALITATIVE_ANALYSIS_FLAG = False # True

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"

results_path = f"{home_path}/disk/Eye-tracking/Results"

with open(home_path + '/disk/Eye-tracking/plots/primary_table.pkl', 'rb') as f:
    primary_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/trajectories_table.pkl', 'rb') as f:
    trajectories_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/AOI_proportions_table.pkl', 'rb') as f:
    AOI_proportions_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/neck_eye_movements_table.pkl', 'rb') as f:
    neck_eye_movements_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/heatmaps_spreading_table.pkl', 'rb') as f:
    heatmaps_spreading_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/qualitative_table.pkl', 'rb') as f:
    qualitative_table = pickle.load(f)


subelite_names = []
elite_names = []
for i in range(len(primary_table)):
    if primary_table[i][1] == 'SubElite':
        if primary_table[i][0] not in subelite_names:
            subelite_names.append(primary_table[i][0])
    if primary_table[i][1] == 'Elite':
        if primary_table[i][0] not in elite_names:
            elite_names.append(primary_table[i][0])

trial_per_athlete_per_move_index = {}
for i in range(1, len(trajectories_table)):
    if trajectories_table[i][0] not in trial_per_athlete_per_move_index.keys():
        basic_dict = {'4-': [], '41': [], '42': [], '43': [], }
        trial_per_athlete_per_move_index[trajectories_table[i][0]] = {'4-': [], '41': [], '42': [], '43': [], }
        trial_per_athlete_per_move_index[trajectories_table[i][0]][trajectories_table[i][2]] += [i]
    else:
        trial_per_athlete_per_move_index[trajectories_table[i][0]][trajectories_table[i][2]] += [i]


with open(home_path + '/disk/Eye-tracking/plots/nb_trial_per_athlete.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Athlete', '4-/', '41/', '42/', '43/'])
    for i, name in  enumerate(trial_per_athlete_per_move_index):
        if name in subelite_names:
            writer.writerow([name,
                             len(trial_per_athlete_per_move_index[name]['4-']),
                             len(trial_per_athlete_per_move_index[name]['41']),
                             len(trial_per_athlete_per_move_index[name]['42']),
                             len(trial_per_athlete_per_move_index[name]['43'])])
    writer.writerow(['-', '-', '-', '-', '-'])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        if name in elite_names:
            writer.writerow([name,
                             len(trial_per_athlete_per_move_index[name]['4-']),
                             len(trial_per_athlete_per_move_index[name]['41']),
                             len(trial_per_athlete_per_move_index[name]['42']),
                             len(trial_per_athlete_per_move_index[name]['43'])])
    f.close()

# ------------------------------------ Primary data frame = Mixed Anova ---------------------------------------- #
if PRIMARY_ANALYSIS_FLAG:
    primary_data_frame = pd.DataFrame(primary_table[1:], columns=primary_table[0])
    primary_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/AllAnalysedMovesGrid.csv")

    # Unequal number of movements per group, unequal number of participants OK!
    list_move_ok_for_now = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                            58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                            68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                            78, 79, 80, 81, 82,
                            83, 84, 85, 86, 87,
                            93, 94, 95, 96, 97,
                            98, 99, 100, 101, 102,
                            114, 115, 116, 117, 118, 119, 120, 121,
                            122, 123, 124, 125, 126, 127, 128, 129,
                            146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                            156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
                            185, 186, 187, 188, 189, 190, 191, 192,
                            193, 194, 195, 196, 197, 198, 199,
                            222, 223, 224, 225, 226, 227,
                            228, 229, 230, 231, 232, 233,
                            244, 245, 246, 247, 248, 249, 250,
                            251, 252, 253, 254, 255]

    print("Attribuer la moyenne a MaBo a la place")


    primary_data_frame_temporary = pd.DataFrame(columns=primary_table[0])
    num_rows = 0
    for i in range(len(AOI_proportions_table)):
        # if primary_data_frame['Name'][i] != 'MaBo':
        if i in list_move_ok_for_now:
            df = {'Name': [primary_data_frame['Name'][i]],
            'Expertise': [primary_data_frame['Expertise'][i]],
            'Acrobatics': [primary_data_frame['Acrobatics'][i]],
            'Fixations duration': [primary_data_frame['Fixations duration'][i]],
            'Number of fixations': [primary_data_frame['Number of fixations'][i]],
            'Quiet eye duration': [primary_data_frame['Quiet eye duration'][i]],
            'Eye amplitude': [primary_data_frame['Eye amplitude'][i]],
            'Neck amplitude': [primary_data_frame['Neck amplitude'][i]],
            'Maximum eye amplitude': [primary_data_frame['Maximum eye amplitude'][i]],
            'Maximum neck amplitude': [primary_data_frame['Maximum neck amplitude'][i]]}
            primary_data_frame_temporary = pd.concat([primary_data_frame_temporary, pd.DataFrame(df)])
            num_rows += 1

    primary_data_frame = primary_data_frame_temporary

    # Eta squared measures the proportion of the total variance in a dependent variable that is associated with the
    # membership of different groups defined by an independent variable. Partial eta squared is a similar measure in which
    # the effects of other independent variables and interactions are partialled out.
    # from : https://eric.ed.gov/?id=EJ927266#:~:text=Eta%20squared%20measures%20the%20proportion,and%20interactions%20are%20partialled%20out.

    print("Mixed ANOVA for Fixations duration")
    out = pg.mixed_anova(data=primary_data_frame, dv='Fixations duration', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Number of fixations")
    out = pg.mixed_anova(data=primary_data_frame, dv='Number of fixations', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Quiet eye duration")
    out = pg.mixed_anova(data=primary_data_frame, dv='Quiet eye duration', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Eye amplitude")
    out = pg.mixed_anova(data=primary_data_frame, dv='Eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Neck amplitude")
    out = pg.mixed_anova(data=primary_data_frame, dv='Neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Maximum eye amplitude")
    out = pg.mixed_anova(data=primary_data_frame, dv='Maximum eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Maximum neck amplitude")
    out = pg.mixed_anova(data=primary_data_frame, dv='Maximum neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


    print("pairwise t-test for Fixations duration")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Fixations duration', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Number of fixations")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Number of fixations', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Quiet eye duration")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Quiet eye duration', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Eye amplitude")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Neck amplitude")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Maximum eye amplitude")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Maximum eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Maximum neck amplitude")
    out = pg.pairwise_tests(data=primary_data_frame, dv='Maximum neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# -------------------------------- Trajectories data frame = SPM1D ? ------------------------------------ #

def find_significant_timings(xi_interp, subelites_data, elites_data):

    admissible_timings = {'4-': np.zeros((len(xi_interp),)), '41': np.zeros((len(xi_interp),)),
                           '42': np.zeros((len(xi_interp),)), '43': np.zeros((len(xi_interp),))}
    significant_timings = {'4-': np.zeros((2,)), '41': np.zeros((2,)), '42': np.zeros((2,)), '43': np.zeros((2,))}

    for j, move in enumerate(['4-', '41', '42', '43']):
        for i in range(len(xi_interp)):
            if np.std(subelites_data[move][:, i]) > 0 and np.std(elites_data[move][:, i]) > 0:
                admissible_timings[move][i] = 1
        begining_of_clusters = np.where(admissible_timings[move][1:] - admissible_timings[move][:-1] == 1)[0] +1
        end_of_clusters = np.where(admissible_timings[move][1:] - admissible_timings[move][:-1] == -1)[0] +1
        if admissible_timings[move][0] == 1:
            begining_of_clusters = np.concatenate(([0], begining_of_clusters))
        if admissible_timings[move][-1] == 1:
            end_of_clusters = np.concatenate((end_of_clusters, [len(xi_interp)]))

        for i in range(len(begining_of_clusters)):
            t = spm1d.stats.ttest2(subelites_data[move][:, begining_of_clusters[i]:end_of_clusters[i]],
                                   elites_data[move][:, begining_of_clusters[i]:end_of_clusters[i]], equal_var=False)
            ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
            if ti.clusters != []:
                # ti.plot()
                # plt.show()
                # embed()
                clusters = ti.clusters
                for k in range(len(clusters)):
                    cluster_x, cluster_z = clusters[k].get_patch_vertices()
                    significant_timings[move] = np.vstack((significant_timings[move],
                                                           np.array([cluster_x[0] + begining_of_clusters[i], cluster_x[1] + begining_of_clusters[i]])))

    return admissible_timings, significant_timings

def plot_trajectories_data_frame(trajectory_curves_per_athelte_per_move, move, title_variable, output_filename, significant_timings, bound_side):

    fig, axs = plt.subplots(1, 2)
    i_subelite = 0
    i_elite = 0
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in subelite_names:
            if trajectory_curves_per_athelte_per_move[name][move] != []:
                axs[0].plot(trajectory_curves_per_athelte_per_move[name][move][0, index_variable], trajectory_curves_per_athelte_per_move[name][move][1, index_variable],
                            color=colors_subelites[i_subelite],
                            label=name)
            i_subelite += 1
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in elite_names:
            if trajectory_curves_per_athelte_per_move[name][move] != []:
                axs[1].plot(trajectory_curves_per_athelte_per_move[name][move][0, index_variable], trajectory_curves_per_athelte_per_move[name][move][1, index_variable],
                            color=colors_elites[i_elite],
                            label=name)
            i_elite += 1

    print("Ajouter significant differences in bold lines ?")

    for j in range(2):
        # Plot trampo bed
        axs[j].add_patch(Rectangle((-3.5 * 0.3048, -7 * 0.3048), 7 * 0.3048, 14 * 0.3048, facecolor='k', alpha=0.2))
        # Plot horizontal lines of the symmetrized gymnasium
        axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2/2, -7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side, bound_side]), np.array([7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2/2 - (9.4620-1.2192), -7.2/2 - (9.4620-1.2192)]), '-k')
        axs[j].plot(np.array([-bound_side, bound_side]), np.array([7.2/2 + 9.4620-1.2192, 7.2/2 + 9.4620-1.2192]), '-k')
        axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2/2 - (9.4620-1.2192) - 7.2, -7.2/2 - (9.4620-1.2192) - 7.2]), '-k')
        axs[j].plot(np.array([-bound_side - (9.4620-1.2192), bound_side]), np.array([-7.2/2, -7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side - (9.4620-1.2192), bound_side]), np.array([7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([bound_side, bound_side + 9.4620 - 1.2192]), np.array([-7.2/2, -7.2/2]), '-k')
        axs[j].plot(np.array([bound_side, bound_side + 9.4620 - 1.2192]), np.array([7.2/2, 7.2/2]), '-k')
        # Plot vertical lines of the symmetrized gymnasium
        axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side - (9.4620-1.2192), -bound_side - (9.4620-1.2192)]), np.array([-7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([bound_side + 9.4620-1.2192, bound_side + 9.4620-1.2192]), np.array([-7.2/2, 7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side, -bound_side]), np.array([7.2/2, 7.2 / 2 + 9.4620-1.2192]), '-k')
        axs[j].plot(np.array([bound_side, bound_side]), np.array([7.2/2, 7.2/2 + 9.4620-1.2192]), '-k')
        axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2/2 - (9.4620-1.2192), 7.2/2]), '-k')
        axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2/2 - (9.4620-1.2192), 7.2/2]), '-k')
        axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2/2 - (9.4620-1.2192) - 7.2, -7.2/2 - (9.4620-1.2192)]), '-k')
        axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2/2 - (9.4620-1.2192) - 7.2, -7.2/2 - (9.4620-1.2192)]), '-k')
        axs[j].axis('equal')
    unwrap_and_plot_gaze_position()

    plt.subplots_adjust(right=0.8)
    plt.suptitle(title_variable + move)
    plt.savefig(output_filename, dpi=300)
    plt.show()
    return

def unwrap_and_plot_gaze_position(gaze_position, wall_index, bound_side):

    gaze_position_x_y = np.zeros((2, wall_index))
    for i in range(1, len(wall_index)):
        if wall_index[i][0] == 1:  # trampoline
            gaze_position_x_y[i] = gaze_position[:2]
        elif wall_index[i][1] == 1:  # wall front
            gaze_position_x_y[i] = [gaze_position[0], 7.2/2 + gaze_position[1]]
        elif wall_index[i][2] == 1:  # ceiling
            gaze_position_x_y[i] = [gaze_position[0], -7.2/2 - (9.4620-1.2192) - 7.2/2 - gaze_position[1]]
        elif wall_index[i][3] == 1:  # wall back
            gaze_position_x_y[i] = [gaze_position[0], -7.2/2 - gaze_position[2]]
        elif wall_index[i][4] == 1:  # bound right
            gaze_position_x_y[i] = [bound_side + gaze_position[2], gaze_position[1]]
        elif wall_index[i][5] == 1:  # bound left
            gaze_position_x_y[i] = [-bound_side - gaze_position[2], gaze_position[1]]

    return gaze_position_x_y


if TRAJECTORIES_ANALYSIS_FLAG:

    bound_side = 3 + 121 * 0.0254 / 2

    nb_interp_points = 500
    xi_interp = np.linspace(0, 1, nb_interp_points)
    trajectory_curves_per_athelte_per_move = {}
    for j, name in enumerate(trial_per_athlete_per_move_index.keys()):
        trajectory_curves_per_athelte_per_move[name] = {}
        for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
            trajectory_curves_per_athelte_per_move[name][move] = []
            index_this_time = trial_per_athlete_per_move_index[name][move]
            trajectory_curve = np.zeros((nb_interp_points, ))
            for k in index_this_time:
                x_index_this_time = np.linspace(0, 1, len(trajectories_table[k][4]))
                trajectory_this_time = unwrap_and_plot_gaze_position(trajectories_table[k][4], trajectories_table[k][5], bound_side)
                trajectory_curve += interp1d(x_index_this_time, trajectory_this_time)(xi_interp)

            if len(index_this_time) > 0:
                trajectory_curves_per_athelte_per_move[name][move] = [trajectory_curve/len(index_this_time)]

    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    subelites_trajectory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_trajectory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    for name in trajectory_curves_per_athelte_per_move.keys():
        for move in trajectory_curves_per_athelte_per_move[name].keys():
            if name in subelite_names:
                if trajectory_curves_per_athelte_per_move[name][move] != []:
                    subelites_trajectory[move] = np.vstack((subelites_trajectory[move], trajectory_curves_per_athelte_per_move[name][move]))
            elif name in elite_names:
                if trajectory_curves_per_athelte_per_move[name][move] != []:
                    elites_trajectory[move] = np.vstack((elites_trajectory[move], trajectory_curves_per_athelte_per_move[name][move]))
            else:
                print(f"Probleme de nom: {name} not recognised")

    for i, move in enumerate(['4-', '41', '42', '43']):
        subelites_trajectory[move] = subelites_trajectory[move][1:]
        elites_trajectory[move] = elites_trajectory[move][1:]

    admissible_timings_x, significant_timings_x = find_significant_timings(xi_interp, subelites_trajectory[0, :], elites_trajectory[0, :])
    admissible_timings_y, significant_timings_y = find_significant_timings(xi_interp, subelites_trajectory[1, :], elites_trajectory[1, :])

    significant_timings = np.logical_or(significant_timings_x, significant_timings_y)
    
    for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
        plot_trajectories_data_frame(trajectory_curves_per_athelte_per_move, move, xi_interp, "Projected gaze trajectory symmetrized (PGOS) ",
                      home_path + '/disk/Eye-tracking/plots/' + f'PGOS_{move}.png',
                      significant_timings)


# ----------------------------------------- AOI data frame = Mixed ANOVA --------------------------------------------- #
# AOI_proportions_table
if AOI_ANALYSIS_FLAG:
    AOI_proportions_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics', 'Trampoline',
                                                            'Wall back front', 'Ceiling', 'Wall sides',
                                                            'Athlete himself', 'Blink'])
    num_rows = 0
    for i in range(len(AOI_proportions_table)):
        # if primary_data_frame['Name'][i] != 'MaBo':
        if i in list_move_ok_for_now:
            df = {'Name': [AOI_proportions_table['Name'][i]],
            'Expertise': [AOI_proportions_table['Expertise'][i]],
            'Acrobatics': [AOI_proportions_table['Acrobatics'][i]],
            'Trampoline': [AOI_proportions_table['Trampoline'][i]],
            'Wall back front': [AOI_proportions_table['Wall front'][i] + AOI_proportions_table['Wall back'][i]],
            'Ceiling': [AOI_proportions_table['Ceiling'][i]],
            'Wall sides': [AOI_proportions_table['Wall sides'][i]],
            'Athlete himself': [AOI_proportions_table['Athlete himself'][i]],
            'Blink': [AOI_proportions_table['Blink'][i]]}
            primary_data_frame_temporary = pd.concat([AOI_proportions_table_temporary, pd.DataFrame(df)])
            num_rows += 1

    AOI_proportions_table = AOI_proportions_table_temporary


    print("Mixed ANOVA for Trampoline")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Trampoline', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Wall front")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Wall front', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Wall back")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Wall back', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Ceiling")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Ceiling', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Wall sides")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Wall sides', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Athlete himself")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Athlete himself', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Blink")
    out = pg.mixed_anova(data=AOI_proportions_table, dv='Blink', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Trampoline")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Trampoline', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Wall front")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Wall front', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Wall back")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Wall back', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Ceiling")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Ceiling', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Wall sides")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Wall sides', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Athlete himself")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Athlete himself', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Blink")
    out = pg.pairwise_tests(data=AOI_proportions_table, dv='Blink', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ------------------------- Neck + eye proportion of movements data frame = Mixed ANOVA ----------------------------- #
if NECK_EYE_ANALYSIS_FLAG:
    neck_eye_movements_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Anticipatory movements', 'Compensatory movements',
                                                               'Spotting movements', 'Movement detection', 'Blinks'])
    num_rows = 0
    for i in range(len(neck_eye_movements_table)):
        if i in list_move_ok_for_now:
            df = {'Name': [neck_eye_movements_table['Name'][i]],
            'Expertise': [neck_eye_movements_table['Expertise'][i]],
            'Acrobatics': [neck_eye_movements_table['Acrobatics'][i]],
            'Anticipatory movements': [neck_eye_movements_table['Anticipatory movements'][i]],
            'Compensatory movements': [neck_eye_movements_table['Compensatory movements'][i]],
            'Spotting movements': [neck_eye_movements_table['Spotting movements'][i]],
            'Movement detection': [neck_eye_movements_table['Movement detection'][i]],
            'Blinks': [neck_eye_movements_table['Blinks'][i]]}
            primary_data_frame_temporary = pd.concat([neck_eye_movements_table_temporary, pd.DataFrame(df)])
            num_rows += 1

    neck_eye_movements_table = neck_eye_movements_table_temporary


    print("Mixed ANOVA for Anticipatory movements")
    out = pg.mixed_anova(data=neck_eye_movements_table, dv='Anticipatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Compensatory movements")
    out = pg.mixed_anova(data=neck_eye_movements_table, dv='Compensatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Spotting movements")
    out = pg.mixed_anova(data=neck_eye_movements_table, dv='Spotting movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Movement detection")
    out = pg.mixed_anova(data=neck_eye_movements_table, dv='Movement detection', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Blinks")
    out = pg.mixed_anova(data=neck_eye_movements_table, dv='Blinks', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Anticipatory movements")
    out = pg.pairwise_tests(data=neck_eye_movements_table, dv='Anticipatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Compensatory movements")
    out = pg.pairwise_tests(data=neck_eye_movements_table, dv='Compensatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Spotting movements")
    out = pg.pairwise_tests(data=neck_eye_movements_table, dv='Spotting movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Movement detection")
    out = pg.pairwise_tests(data=neck_eye_movements_table, dv='Movement detection', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Blinks")
    out = pg.pairwise_tests(data=neck_eye_movements_table, dv='Blinks', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ----------------------------------- Heatmap spreading data frame = Mixed ANOVA -------------------------------------- #

if SPREADING_HEATMAP_FLAG:
    heatmaps_spreading_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Distance from the center of each point of the heatmap',
                                                               'Heat map 90th percentile'])
    num_rows = 0
    for i in range(len(heatmaps_spreading_table)):
        if i in list_move_ok_for_now:
            df = {'Name': [heatmaps_spreading_table['Name'][i]],
                  'Expertise': [heatmaps_spreading_table['Expertise'][i]],
                  'Acrobatics': [heatmaps_spreading_table['Acrobatics'][i]],
                  'Distance from the center of each point of the heatmap': [heatmaps_spreading_table['Distance from the center of each point of the heatmap'][i]],
                  'Heat map 90th percentile': [heatmaps_spreading_table['Heat map 90th percentile'][i]]}
            primary_data_frame_temporary = pd.concat([heatmaps_spreading_table_temporary, pd.DataFrame(df)])
            num_rows += 1

    heatmaps_spreading_table = heatmaps_spreading_table_temporary

    print("Mixed ANOVA for Heat map 90th percentile")
    out = pg.mixed_anova(data=heatmaps_spreading_table, dv='Heat map 90th percentile', within='Acrobatics',
                         between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Heat map 90th percentile")
    out = pg.pairwise_tests(data=heatmaps_spreading_table, dv='Heat map 90th percentile', within='Acrobatics',
                            between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ---------------------------------------- Qualitative data frame = SPM1D -------------------------------------------- #
def nearest_interp(xi, x, y):
    out = np.zeros((len(xi),))
    for i in range(len(xi)):
        index_closest = np.argmin(np.abs(xi[i]-x))
        out[i] = y[index_closest]
    return out


def plot_presence(presence_curves_per_athelte, move, xi_interp, index_variable, title_variable, output_file_name, significant_timings):

    fig, axs = plt.subplots(2, 1)
    i_subelite = 0
    i_elite = 0
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            if presence_curves_per_athelte[name][move] != []:
                axs[0].plot(xi_interp * 100, presence_curves_per_athelte[name][move][index_variable], color=colors_subelites[i_subelite],
                            label=name)
            i_subelite += 1
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            if presence_curves_per_athelte[name][move] != []:
                axs[1].plot(xi_interp * 100, presence_curves_per_athelte[name][move][index_variable], color=colors_elites[i_elite],
                            label=name)
            i_elite += 1

    if len(np.shape(significant_timings[move])) > 1:
        for i in range(1, np.shape(significant_timings[move])[0]):
            print(f"Significant difference for {move}, {title_variable}\n")
            axs[0].axvspan(significant_timings[move][i, 0] * 100/len(xi_interp), significant_timings[move][i, 1] * 100, alpha=0.2, facecolor='k')
            axs[1].axvspan(significant_timings[move][i, 0] * 100/len(xi_interp), significant_timings[move][i, 1] * 100, alpha=0.2, facecolor='k')

    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_xlabel("Normalized time [%]")
    plt.subplots_adjust(right=0.8)
    plt.suptitle(title_variable + move)
    plt.savefig(output_file_name, dpi=300)
    return

if QUALITATIVE_ANALYSIS_FLAG:

    nb_interp_points = 500
    xi_interp = np.linspace(0, 1, nb_interp_points)
    presence_curves_per_athelte = {}
    for j, name in enumerate(trial_per_athlete_per_move_index.keys()):
        presence_curves_per_athelte[name] = {}
        for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
            presence_curves_per_athelte[name][move] = []
            index_this_time = trial_per_athlete_per_move_index[name][move]
            anticipatory_curve = np.zeros((nb_interp_points, ))
            compensatory_curve = np.zeros((nb_interp_points, ))
            spotting_curve = np.zeros((nb_interp_points, ))
            movement_detection_curve = np.zeros((nb_interp_points, ))
            blink_curve = np.zeros((nb_interp_points, ))
            for k in index_this_time:
                x_index_this_time = np.linspace(0, 1, len(qualitative_table[k][4]))
                anticipatory_presence_this_time = qualitative_table[k][4].astype(int)
                compensatory_presence_this_time = qualitative_table[k][5].astype(int)
                spotting_presence_this_time = qualitative_table[k][6].astype(int)
                movement_detection_presence_this_time = qualitative_table[k][7].astype(int)
                blink_presence_this_time = qualitative_table[k][8].astype(int)

                anticipatory_curve += nearest_interp(xi_interp, x_index_this_time, anticipatory_presence_this_time)
                compensatory_curve += nearest_interp(xi_interp, x_index_this_time, compensatory_presence_this_time)
                spotting_curve += nearest_interp(xi_interp, x_index_this_time, spotting_presence_this_time)
                movement_detection_curve += nearest_interp(xi_interp, x_index_this_time, movement_detection_presence_this_time)
                blink_curve += nearest_interp(xi_interp, x_index_this_time, blink_presence_this_time)

            if len(index_this_time) > 0:
                presence_curves_per_athelte[name][move] = [anticipatory_curve/len(index_this_time),
                                                 compensatory_curve/len(index_this_time),
                                                 spotting_curve/len(index_this_time),
                                                 movement_detection_curve/len(index_this_time),
                                                 blink_curve/len(index_this_time)]

    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    subelites_anticipatory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_anticipatory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    subelites_compensatory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_compensatory = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    subelites_spotting = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_spotting = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    subelites_movement_detection = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_movement_detection = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    subelites_blink = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    elites_blink = {'4-': np.zeros((len(xi_interp))), '41': np.zeros((len(xi_interp))), '42': np.zeros((len(xi_interp))), '43': np.zeros((len(xi_interp)))}
    for name in presence_curves_per_athelte.keys():
        for move in presence_curves_per_athelte[name].keys():
            if name in subelite_names:
                if presence_curves_per_athelte[name][move] != []:
                    subelites_anticipatory[move] = np.vstack((subelites_anticipatory[move], presence_curves_per_athelte[name][move][0]))
                    subelites_compensatory[move] = np.vstack((subelites_compensatory[move], presence_curves_per_athelte[name][move][1]))
                    subelites_spotting[move] = np.vstack((subelites_spotting[move], presence_curves_per_athelte[name][move][2]))
                    subelites_movement_detection[move] = np.vstack((subelites_movement_detection[move], presence_curves_per_athelte[name][move][3]))
                    subelites_blink[move] = np.vstack((subelites_blink[move], presence_curves_per_athelte[name][move][4]))
            elif name in elite_names:
                if presence_curves_per_athelte[name][move] != []:
                    elites_anticipatory[move] = np.vstack((elites_anticipatory[move], presence_curves_per_athelte[name][move][0]))
                    elites_compensatory[move] = np.vstack((elites_compensatory[move], presence_curves_per_athelte[name][move][1]))
                    elites_spotting[move] = np.vstack((elites_spotting[move], presence_curves_per_athelte[name][move][2]))
                    elites_movement_detection[move] = np.vstack((elites_movement_detection[move] , presence_curves_per_athelte[name][move][3]))
                    elites_blink[move] = np.vstack((elites_blink[move], presence_curves_per_athelte[name][move][4]))
            else:
                print(f"Probleme de nom: {name} not recognised")

    for i, move in enumerate(['4-', '41', '42', '43']):
        subelites_anticipatory[move] = subelites_anticipatory[move][1:]
        subelites_compensatory[move] = subelites_compensatory[move][1:]
        subelites_spotting[move] = subelites_spotting[move][1:]
        subelites_movement_detection[move] = subelites_movement_detection[move][1:]
        subelites_blink[move] = subelites_blink[move][1:]
        elites_anticipatory[move] = elites_anticipatory[move][1:]
        elites_compensatory[move] = elites_compensatory[move][1:]
        elites_spotting[move] = elites_spotting[move][1:]
        elites_movement_detection[move] = elites_movement_detection[move][1:]
        elites_blink[move] = elites_blink[move][1:]

    admissible_timings_anticipatory, significant_timings_anticipatory = find_significant_timings(xi_interp, subelites_anticipatory, elites_anticipatory)
    admissible_timings_compensatory, significant_timings_compensatory = find_significant_timings(xi_interp, subelites_compensatory, elites_compensatory)
    admissible_timings_spotting, significant_timings_spotting = find_significant_timings(xi_interp, subelites_spotting, elites_spotting)
    admissible_timings_movement_detection, significant_timings_movement_detection = find_significant_timings(xi_interp, subelites_movement_detection, elites_movement_detection)
    admissible_timings_blink, significant_timings_blink = find_significant_timings(xi_interp, subelites_blink, elites_blink)

    for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
        plot_presence(presence_curves_per_athelte, move, xi_interp, 0, "Anticipatory movements ",
                      home_path + '/disk/Eye-tracking/plots/' + f'anticiaptory_presence_{move}.png',
                      significant_timings_anticipatory)
        plot_presence(presence_curves_per_athelte, move, xi_interp, 1, "Compensatory movements ",
                      home_path + '/disk/Eye-tracking/plots/' + f'compensatory_presence_{move}.png',
                      significant_timings_compensatory)
        plot_presence(presence_curves_per_athelte, move, xi_interp, 2, "Spotting ",
                      home_path + '/disk/Eye-tracking/plots/' + f'spotting_presence_{move}.png',
                      significant_timings_spotting)
        plot_presence(presence_curves_per_athelte, move, xi_interp, 3, "Movement detection ",
                      home_path + '/disk/Eye-tracking/plots/' + f'movement_detection_presence_{move}.png',
                      significant_timings_movement_detection)
        plot_presence(presence_curves_per_athelte, move, xi_interp, 4, "Blink ",
                      home_path + '/disk/Eye-tracking/plots/' + f'blink_presence_{move}.png',
                      significant_timings_blink)







