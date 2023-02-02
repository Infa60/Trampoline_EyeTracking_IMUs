
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
# run --- python stats_tests.py > stats_output.txt ---  to save the output to a file
##########################################################################################


PRIMARY_ANALYSIS_FLAG = True
TRAJECTORIES_ANALYSIS_FLAG = True
GENERATE_EACH_ATHLETE_PGOS_GRAPH = True
AOI_ANALYSIS_FLAG = True
NECK_EYE_ANALYSIS_FLAG = True
SPREADING_HEATMAP_FLAG = True
QUALITATIVE_ANALYSIS_FLAG = True

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

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
index_subelites  = []
index_elites = []
for i in range(len(primary_table)):
    if primary_table[i][1] == 'SubElite':
        if primary_table[i][0] not in subelite_names:
            subelite_names.append(primary_table[i][0])
        index_subelites.append(i)
    elif primary_table[i][1] == 'Elite':
        if primary_table[i][0] not in elite_names:
            elite_names.append(primary_table[i][0])
        index_elites.append(i)

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

# Give the mean value to the athelte who did not want to do one move in particular for ANOVAs
primary_table_array = np.array(primary_table)
AOI_proportions_table_array = np.array(AOI_proportions_table)
neck_eye_movements_table_array = np.array(neck_eye_movements_table)
heatmaps_spreading_table_array = np.array(heatmaps_spreading_table)
for i, name in enumerate(trial_per_athlete_per_move_index):
    for j, move in enumerate(trial_per_athlete_per_move_index[name]):
        if len(trial_per_athlete_per_move_index[name][move]) == 0:
            index = (index_elites if name in elite_names else index_subelites)
            primary_table += [[name,
                              'Elite' if name in elite_names else 'SubElite',
                              move,
                              np.nanmean(primary_table_array[index, 3].astype(float)),
                              np.nanmean(primary_table_array[index, 4].astype(float)),
                              np.nanmean(primary_table_array[index, 5].astype(float)),
                              np.nanmean(primary_table_array[index, 6].astype(float)),
                              np.nanmean(primary_table_array[index, 7].astype(float)),
                              np.nanmean(primary_table_array[index, 8].astype(float)),
                              np.nanmean(primary_table_array[index, 9].astype(float)),
                              ]]
            AOI_proportions_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  np.nanmean(AOI_proportions_table_array[index, 3].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 4].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 5].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 6].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 7].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 8].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 9].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 10].astype(float)),
                                  ]]
            neck_eye_movements_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  np.nanmean(neck_eye_movements_table_array[index, 3].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 4].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 5].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 6].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 7].astype(float)),
                                  ]]
            heatmaps_spreading_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  None,
                                  np.nanmean(heatmaps_spreading_table_array[index, 4].astype(float)),
                                  ]]


# ------------------------------------ Primary data frame = Mixed Anova ---------------------------------------- #
if PRIMARY_ANALYSIS_FLAG:
    primary_data_frame = pd.DataFrame(primary_table[1:], columns=primary_table[0])
    primary_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/primary_data_frame.csv")

    primary_data_frame_temporary = pd.DataFrame(columns=primary_table[0])
    for i in range(len(primary_data_frame)):
        df = {'Name': [primary_data_frame['Name'][i]],
        'Expertise': [primary_data_frame['Expertise'][i]],
        'Acrobatics': [primary_data_frame['Acrobatics'][i]],
        'Fixations duration relative': [primary_data_frame['Fixations duration relative'][i]],
        'Number of fixations': [primary_data_frame['Number of fixations'][i]],
        'Quiet eye duration relative': [primary_data_frame['Quiet eye duration relative'][i]],
        'Eye amplitude': [primary_data_frame['Eye amplitude'][i]],
        'Neck amplitude': [primary_data_frame['Neck amplitude'][i]],
        'Maximum eye amplitude': [primary_data_frame['Maximum eye amplitude'][i]],
        'Maximum neck amplitude': [primary_data_frame['Maximum neck amplitude'][i]]}
        primary_data_frame_temporary = pd.concat([primary_data_frame_temporary, pd.DataFrame(df)])

    primary_data_frame = primary_data_frame_temporary

    # Eta squared measures the proportion of the total variance in a dependent variable that is associated with the
    # membership of different groups defined by an independent variable. Partial eta squared is a similar measure in which
    # the effects of other independent variables and interactions are partialled out.
    # from : https://eric.ed.gov/?id=EJ927266#:~:text=Eta%20squared%20measures%20the%20proportion,and%20interactions%20are%20partialled%20out.

    print("Mixed ANOVA for Fixations duration relative")
    out = pg.mixed_anova(data=primary_data_frame, dv='Fixations duration relative', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Number of fixations")
    out = pg.mixed_anova(data=primary_data_frame, dv='Number of fixations', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Quiet eye duration relative")
    out = pg.mixed_anova(data=primary_data_frame, dv='Quiet eye duration relative', within='Acrobatics', between='Expertise', subject='Name')
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


    print("pairwise t-test for Fixations duration relative")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Fixations duration relative', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Number of fixations")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Number of fixations', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Quiet eye duration relative")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Quiet eye duration relative', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Eye amplitude")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Neck amplitude")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Maximum eye amplitude")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Maximum eye amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Maximum neck amplitude")
    out = pg.pairwise_tukey(data=primary_data_frame, dv='Maximum neck amplitude', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# -------------------------------- Trajectories data frame = SPM1D ? ------------------------------------ #

def find_significant_timings(xi_interp, subelites_data, elites_data):

    admissible_timings = {'4-': np.zeros((len(xi_interp),)), '41': np.zeros((len(xi_interp),)),
                           '42': np.zeros((len(xi_interp),)), '43': np.zeros((len(xi_interp),))}
    significant_timings = {'4-': np.zeros((2,)), '41': np.zeros((2,)),
                           '42': np.zeros((2,)), '43': np.zeros((2,))}

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
            ti = t.inference(alpha=0.05, two_tailed=False)
            if hasattr(ti, 'clusters'):
                if ti.clusters != []:
                    # ti.plot()
                    # plt.show()
                    clusters = ti.clusters
                    for k in range(len(clusters)):
                        cluster_x, cluster_z = clusters[k].get_patch_vertices()
                        significant_timings[move] = np.vstack((significant_timings[move],
                                                               np.array([cluster_x[0] + begining_of_clusters[i], cluster_x[1] + begining_of_clusters[i]])))
        if significant_timings[move].shape == (2,):
            significant_timings[move] = None
        else:
            significant_timings[move] = significant_timings[move][1:]
            print(f"Significant timings for {move}: {significant_timings[move]}")
    return admissible_timings, significant_timings

def plot_gymnasium_unwrapped(axs, j, FLAG_3D=False):
    # # Plot trampo bed
    # if FLAG_3D:
    #     X, Y = np.meshgrid([-3.5 * 0.3048, 3.5 * 0.3048], [-7 * 0.3048, 7 * 0.3048])
    #     Z = np.zeros(X.shape)
    #     axs[j].plot_surface(X, Y, Z, color="k", alpha=0.4)
    # else:
    #     axs[j].add_patch(Rectangle((-3.5 * 0.3048, -7 * 0.3048), 7 * 0.3048, 14 * 0.3048, facecolor='k', alpha=0.2))
    # # Plot horizontal lines of the symmetrized gymnasium
    # axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2, -7.2]), '-k')
    # axs[j].plot(np.array([-bound_side, bound_side]), np.array([7.2, 7.2]), '-k')
    # axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2 - (9.4620-1.2192), -7.2 - (9.4620-1.2192)]), '-k')
    # axs[j].plot(np.array([-bound_side, bound_side]), np.array([7.2 + 9.4620-1.2192, 7.2 + 9.4620-1.2192]), '-k')
    # axs[j].plot(np.array([-bound_side, bound_side]), np.array([-7.2 - (9.4620-1.2192) - 7.2, -7.2 - (9.4620-1.2192) - 7.2]), '-k')
    # axs[j].plot(np.array([-bound_side - (9.4620-1.2192), bound_side]), np.array([-7.2, -7.2]), '-k')
    # axs[j].plot(np.array([-bound_side - (9.4620-1.2192), bound_side]), np.array([7.2, 7.2]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side + 9.4620 - 1.2192]), np.array([-7.2, -7.2]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side + 9.4620 - 1.2192]), np.array([7.2, 7.2]), '-k')
    # # Plot vertical lines of the symmetrized gymnasium
    # axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2, 7.2]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2, 7.2]), '-k')
    # axs[j].plot(np.array([-bound_side - (9.4620-1.2192), -bound_side - (9.4620-1.2192)]), np.array([-7.2, 7.2]), '-k')
    # axs[j].plot(np.array([bound_side + 9.4620-1.2192, bound_side + 9.4620-1.2192]), np.array([-7.2, 7.2]), '-k')
    # axs[j].plot(np.array([-bound_side, -bound_side]), np.array([7.2, 7.2 + 9.4620-1.2192]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side]), np.array([7.2, 7.2 + 9.4620-1.2192]), '-k')
    # axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2 - (9.4620-1.2192), 7.2]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2 - (9.4620-1.2192), 7.2]), '-k')
    # axs[j].plot(np.array([-bound_side, -bound_side]), np.array([-7.2 - (9.4620-1.2192) - 7.2, -7.2 - (9.4620-1.2192)]), '-k')
    # axs[j].plot(np.array([bound_side, bound_side]), np.array([-7.2 - (9.4620-1.2192) - 7.2, -7.2 - (9.4620-1.2192)]), '-k')

    # Plot trampo bed
    if FLAG_3D:
        X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
        Z = np.zeros(X.shape)
        axs[j].plot_surface(X, Y, Z, color="k", alpha=0.4)
    else:
        axs[j].add_patch(Rectangle((-7 * 0.3048, -3.5 * 0.3048), 14 * 0.3048, 7 * 0.3048, facecolor='k', alpha=0.2))
    # Plot vertical lines of the symmetrized gymnasium
    axs[j].plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), -7.2 - (9.4620-1.2192)]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([7.2 + 9.4620-1.2192, 7.2 + 9.4620-1.2192]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192) - 2*7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2, -7.2]), np.array([-bound_side - (9.4620-1.2192), bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([-bound_side - (9.4620-1.2192), bound_side]), '-k')
    axs[j].plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
    # Plot horizontal lines of the symmetrized gymnasium
    axs[j].plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([-bound_side - (9.4620-1.2192), -bound_side - (9.4620-1.2192)]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([bound_side + 9.4620-1.2192, bound_side + 9.4620-1.2192]), '-k')
    axs[j].plot(np.array([7.2, 7.2 + 9.4620-1.2192]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2 + 9.4620-1.2192]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), 7.2]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), 7.2]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192)]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192)]), np.array([bound_side, bound_side]), '-k')

    axs[j].text(-7.2 - (9.4620-1.2192) - 2*7.2 + 7.2/2 + 1, bound_side + 0.1, "Ceiling", fontsize=10)
    axs[j].text(-7.2 - (9.4620-1.2192) + 1, bound_side + 0.1, "Wall back", fontsize=10)
    axs[j].text(7.2 + 1, bound_side + 0.1, "Wall front", fontsize=10)
    axs[j].text(-7.2 + 7.2/2 + 1, bound_side + 9.4620-1.2192 + 0.1, "Wall left", fontsize=10)
    axs[j].text(-7.2 + 7.2/2 + 0.5, -bound_side - (9.4620-1.2192) - 1, "Wall right", fontsize=10)

    return

def plot_trajectories_data_frame(
        trajectory_curves_per_athelte_per_move,
        move,
        title_variable,
        output_filename,
):

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    i_subelite = 0
    i_elite = 0
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in subelite_names:
            if trajectory_curves_per_athelte_per_move[name][move] != []:
                axs[0].plot(trajectory_curves_per_athelte_per_move[name][move][0, :], trajectory_curves_per_athelte_per_move[name][move][1, :],
                            color=colors_subelites[i_subelite],
                            marker='.',
                            linestyle='None',
                            markersize=1,
                            label=name)
                # if significant_timings[move] is not None:
                #     for index in significant_timings[move].shape:
                #         index_this_time = np.arange(int(round(significant_timings[move][index, 0])),
                #                                     int(round(significant_timings[move][index, 1])))
                #         axs[0].plot(trajectory_curves_per_athelte_per_move[name][move][0, significant_timings[move][index_this_time]],
                #                     trajectory_curves_per_athelte_per_move[name][move][1, significant_timings[move][index_this_time]],
                #                     color=colors_subelites[i_subelite],
                #                     linewidth=3)
            i_subelite += 1
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in elite_names:
            if not isinstance(trajectory_curves_per_athelte_per_move[name][move], list):
                axs[1].plot(trajectory_curves_per_athelte_per_move[name][move][0, :], trajectory_curves_per_athelte_per_move[name][move][1, :],
                            color=colors_elites[i_elite],
                            marker='.',
                            linestyle='None',
                            markersize=1,
                            label=name)
                # if significant_timings[move] is not None:
                #     for index in significant_timings[move].shape:
                #         index_this_time = np.arange(int(round(significant_timings[move][index, 0])),
                #                                     int(round(significant_timings[move][index, 1])))
                #         axs[1].plot(trajectory_curves_per_athelte_per_move[name][move][0, significant_timings[move][index_this_time]],
                #                     trajectory_curves_per_athelte_per_move[name][move][1, significant_timings[move][index_this_time]],
                #                     color=colors_elites[i_elite],
                #                     linewidth=3)
            i_elite += 1

    for j in range(2):
        plot_gymnasium_unwrapped(axs, j)
        axs[j].axis('equal')

    plt.subplots_adjust(right=0.8)
    plt.suptitle(title_variable + move)
    axs[0].legend(bbox_to_anchor=(2.3, 1), loc='upper left', borderaxespad=0.)
    axs[1].legend(bbox_to_anchor=(1.1, 0.5), loc='upper left', borderaxespad=0.)
    plt.savefig(output_filename, dpi=300)
    # plt.show()
    return

def plot_mean_PGOS_per_athlete(name, move, interpolated_unwrapped_trajectory, home_path):
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i/np.shape(interpolated_unwrapped_trajectory)[2]) for i in range(np.shape(interpolated_unwrapped_trajectory)[2])]
    min_RMSE = 1e20
    i_min = 0
    for i in range(len(interpolated_unwrapped_trajectory[0, 0, :])):
        RMSE =  0
        for j in range(len(interpolated_unwrapped_trajectory[0, 0, :]) - 1):
            if i != j:
                nb_nan_elements = max(np.sum(np.isnan(interpolated_unwrapped_trajectory[0, :, i])),
                                        np.sum(np.isnan(interpolated_unwrapped_trajectory[0, :, j])))
                nb_non_nan_elements = 500 - nb_nan_elements
                # RMSE += np.sqrt(np.mean((interpolated_unwrapped_trajectory[0, :, i] - interpolated_unwrapped_trajectory[0, :, j]) ** 2)) / nb_non_nan_elements
                RMSE += np.nansum(np.abs(interpolated_unwrapped_trajectory[0, :, i] - interpolated_unwrapped_trajectory[0, :, j])) / nb_non_nan_elements
        if RMSE < min_RMSE:
            min_RMSE = RMSE
            i_min = i

    if GENERATE_EACH_ATHLETE_PGOS_GRAPH:
        fig, ax = plt.subplots()
        axs = [ax]
        plot_gymnasium_unwrapped(axs, 0)
        for i in range(interpolated_unwrapped_trajectory.shape[2]):
            plt.plot(interpolated_unwrapped_trajectory[0, :, i], interpolated_unwrapped_trajectory[1, :, i], color=colors[i], marker='.', markersize=0.5, linestyle='None')
        # plt.plot(mean_trajectory[0, :], mean_trajectory[1, :], '-k')
        plt.plot(interpolated_unwrapped_trajectory[0, :, i_min], interpolated_unwrapped_trajectory[1, :, i_min], '-k', alpha=0.8, label='Representative\ntrajectory')
        plt.title(f"{name} {move}")
        plt.legend()
        plt.savefig(home_path + f"/disk/Eye-tracking/plots/PGOS/multiple_trials_{name}_{move}.png", dpi=300)
        # plt.show()

    # fig, axs = plt.subplots(2, 1)
    # for i in range(interpolated_unwrapped_trajectory.shape[2]):
    #     axs[0].plot(np.arange(500), interpolated_unwrapped_trajectory[0, :, i], '.b', label=f'{i}')
    #     axs[1].plot(np.arange(500), interpolated_unwrapped_trajectory[1, :, i], '.b', label=f'{i}')
    # axs[0].plot(np.arange(500), mean_trajectory[0, :], '-r')
    # axs[1].plot(np.arange(500), mean_trajectory[1, :], '-r')
    # axs[0].plot(np.arange(500), interpolated_unwrapped_trajectory[0, :, i_min], '-m', label=f'{i}')
    # axs[1].plot(np.arange(500), interpolated_unwrapped_trajectory[1, :, i_min], '-m', label=f'{i}')
    # plt.suptitle(f"{name} {move}")
    # plt.show()
    return interpolated_unwrapped_trajectory[:, :, i_min]

def plot_projection_of_PGOS(name, move, original_trajectory, projected_interpolated_trajectory_curve, home_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    axs = [ax]
    plot_gymnasium_unwrapped(axs, 0, FLAG_3D=True)
    plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], '.r')
    plt.plot(projected_interpolated_trajectory_curve[0, :], projected_interpolated_trajectory_curve[1, :], np.zeros((500, )), '.b')
    plt.title(f"{name} {move}")
    plt.savefig(home_path + f"/disk/Eye-tracking/plots/PGOS//plots/PGOS/projection_{name}_{move}.png", dpi=300)
    # plt.show()
    return

def unwrap_and_plot_gaze_position(gaze_position, wall_index, bound_side):

    gaze_position_x_y = np.zeros((2, np.shape(wall_index)[0]))
    gaze_position_x_y[:, :] = np.nan
    for i in range(1, len(wall_index)):
        if wall_index[i] == 0:  # trampoline
            gaze_position_x_y[:, i] = gaze_position[i][:2]
        if wall_index[i] == 1:  # wall front
            gaze_position_x_y[:, i] = [gaze_position[i][2] + 7.2, gaze_position[i][1]]
        elif wall_index[i] == 2:  # ceiling
            gaze_position_x_y[:, i] = [-7.2 - (9.4620-1.2192) - 7.2 - gaze_position[i][0], gaze_position[i][1]]
        elif wall_index[i] == 3:  # wall back
            gaze_position_x_y[:, i] = [-7.2 - gaze_position[i][2], gaze_position[i][1]]
        elif wall_index[i] == 4:  # bound right
            gaze_position_x_y[:, i] = [gaze_position[i][0], -bound_side - gaze_position[i][2]]
        elif wall_index[i] == 5:  # bound left
            gaze_position_x_y[:, i] = [gaze_position[i][0], bound_side + gaze_position[i][2]]


    return gaze_position_x_y


def nearest_interp(xi, x, y):
    out = np.zeros((len(xi),))
    for i in range(len(xi)):
        index_closest = np.argmin(np.abs(xi[i]-x))
        out[i] = y[index_closest]
    return out

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
            trajectory_curves = np.zeros((2, nb_interp_points, 1))
            trajectory_curves_3D = np.zeros((3, nb_interp_points, 1))
            for k in index_this_time:
                x_index_this_time = np.linspace(0, 1, len(trajectories_table[k][4]))
                trajectory_this_time_x = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 0])(xi_interp), (nb_interp_points, 1))
                trajectory_this_time_y = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 1])(xi_interp), (nb_interp_points, 1))
                trajectory_this_time_z = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 2])(xi_interp), (nb_interp_points, 1))
                wall_index_closest = nearest_interp(xi_interp, x_index_this_time, trajectories_table[k][6][:, 0])
                trajectory_this_time_3d = np.hstack((trajectory_this_time_x, trajectory_this_time_y, trajectory_this_time_z))
                unwrapped_trajectory_this_time = unwrap_and_plot_gaze_position(
                    trajectory_this_time_3d,
                    wall_index_closest,
                    bound_side)
                trajectory_curves = np.concatenate((trajectory_curves, np.reshape(unwrapped_trajectory_this_time, (2, nb_interp_points, 1))), axis=2)
                trajectory_curves_3D = np.concatenate((trajectory_curves_3D, np.reshape(trajectory_this_time_3d, (3, nb_interp_points, 1))), axis=2)

            trajectory_curves = trajectory_curves[:, :, 1:]
            trajectory_curves_3D = trajectory_curves[:, :, 1:]

            # plot_projection_of_PGOS(name, move, trajectories_table[k][4], trajectory_curves[:, :, -1], home_path)
            if len(index_this_time) > 0:
                representative_trajectory = plot_mean_PGOS_per_athlete(name, move, trajectory_curves, home_path)
                trajectory_curves_per_athelte_per_move[name][move] = representative_trajectory
                plt.close('all')

    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    subelites_trajectory_x = {'4-': np.zeros((len(xi_interp), )), '41': np.zeros((len(xi_interp), )),
                            '42': np.zeros((len(xi_interp), )), '43': np.zeros((len(xi_interp), ))}
    subelites_trajectory_y = {'4-': np.zeros((len(xi_interp), )), '41': np.zeros((len(xi_interp), )),
                            '42': np.zeros((len(xi_interp), )), '43': np.zeros((len(xi_interp), ))}
    elites_trajectory_x = {'4-': np.zeros((len(xi_interp), )), '41': np.zeros((len(xi_interp), )),
                         '42': np.zeros((len(xi_interp), )), '43': np.zeros((len(xi_interp), ))}
    elites_trajectory_y = {'4-': np.zeros((len(xi_interp), )), '41': np.zeros((len(xi_interp), )),
                         '42': np.zeros((len(xi_interp), )), '43': np.zeros((len(xi_interp), ))}
    for name in trajectory_curves_per_athelte_per_move.keys():
        for move in trajectory_curves_per_athelte_per_move[name].keys():
            if name in subelite_names:
                if len(trajectory_curves_per_athelte_per_move[name][move]) != 0:
                    subelites_trajectory_x[move] = np.vstack((subelites_trajectory_x[move], trajectory_curves_per_athelte_per_move[name][move][0, :]))
                    subelites_trajectory_y[move] = np.vstack((subelites_trajectory_y[move], trajectory_curves_per_athelte_per_move[name][move][1, :]))
            elif name in elite_names:
                if len(trajectory_curves_per_athelte_per_move[name][move]) != 0:
                    elites_trajectory_x[move] = np.vstack((elites_trajectory_x[move], trajectory_curves_per_athelte_per_move[name][move][0, :]))
                    elites_trajectory_y[move] = np.vstack((elites_trajectory_y[move], trajectory_curves_per_athelte_per_move[name][move][1, :]))
            else:
                print(f"Probleme de nom: {name} not recognised")

    for i, move in enumerate(['4-', '41', '42', '43']):
        subelites_trajectory_x[move] = subelites_trajectory_x[move][1:, :]
        subelites_trajectory_y[move] = subelites_trajectory_y[move][1:, :]
        elites_trajectory_x[move] = elites_trajectory_x[move][1:, :]
        elites_trajectory_y[move] = elites_trajectory_y[move][1:, :]

        plot_trajectories_data_frame(trajectory_curves_per_athelte_per_move, move, "Projected gaze trajectory symmetrized (PGOS) ",
                      home_path + '/disk/Eye-tracking/plots/' + f'PGOS_{move}.png')
        plt.close('all')

    admissible_timings_x, significant_timings_x = find_significant_timings(xi_interp, subelites_trajectory_x, elites_trajectory_x)
    admissible_timings_y, significant_timings_y = find_significant_timings(xi_interp, subelites_trajectory_y, elites_trajectory_y)

    significant_timings = np.logical_or(significant_timings_x, significant_timings_y)
    if significant_timings[move] is not None:
        print("Significant timings found for SPM1D on PGOS: ", significant_timings[move])

# ----------------------------------------- AOI data frame = Mixed ANOVA --------------------------------------------- #

if AOI_ANALYSIS_FLAG:
    AOI_proportions_data_frame = pd.DataFrame(AOI_proportions_table[1:], columns=AOI_proportions_table[0])
    AOI_proportions_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/AOI_proportions_data_frame.csv")
    AOI_proportions_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics', 'Trampoline bed',
                                                            'Trampoline', 'Wall back front', 'Ceiling', 'Wall sides',
                                                            'Athlete himself', 'Blink'])
    for i in range(len(AOI_proportions_data_frame)):
        # if primary_data_frame['Name'][i] != 'MaBo':
        # if i in list_move_ok_for_now:
        df = {'Name': [AOI_proportions_data_frame['Name'][i]],
        'Expertise': [AOI_proportions_data_frame['Expertise'][i]],
        'Acrobatics': [AOI_proportions_data_frame['Acrobatics'][i]],
        'Trampoline bed': [AOI_proportions_data_frame['Trampoline bed'][i]],
        'Trampoline': [AOI_proportions_data_frame['Trampoline'][i]],
        'Wall back front': [AOI_proportions_data_frame['Wall front'][i] + AOI_proportions_data_frame['Wall back'][i]],
        'Ceiling': [AOI_proportions_data_frame['Ceiling'][i]],
        'Wall sides': [AOI_proportions_data_frame['Wall sides'][i]],
        'Athlete himself': [AOI_proportions_data_frame['Athlete himself'][i]],
        'Blink': [AOI_proportions_data_frame['Blink'][i]]}
        AOI_proportions_table_temporary = pd.concat([AOI_proportions_table_temporary, pd.DataFrame(df)])

    AOI_proportions_data_frame = AOI_proportions_table_temporary

    print("Mixed ANOVA for Trampoline bed")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Trampoline bed', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Trampoline")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Trampoline', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Wall back front")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Wall back front', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Ceiling")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Ceiling', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Wall sides")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Wall sides', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Athlete himself")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Athlete himself', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Blink")
    out = pg.mixed_anova(data=AOI_proportions_data_frame, dv='Blink', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Trampoline bed")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Trampoline bed', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Trampoline")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Trampoline', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Wall back front")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Wall back front', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Ceiling")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Ceiling', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Wall sides")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Wall sides', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Athlete himself")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Athlete himself', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Blink")
    out = pg.pairwise_tukey(data=AOI_proportions_data_frame, dv='Blink', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ------------------------- Neck + eye proportion of movements data frame = Mixed ANOVA ----------------------------- #
if NECK_EYE_ANALYSIS_FLAG:
    neck_eye_movements_data_frame = pd.DataFrame(neck_eye_movements_table[1:], columns=neck_eye_movements_table[0])
    neck_eye_movements_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/neck_eye_movements_data_frame.csv")
    neck_eye_movements_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Anticipatory movements', 'Compensatory movements',
                                                               'Spotting movements', 'Movement detection', 'Blinks'])
    for i in range(len(neck_eye_movements_data_frame)):
        # if i in list_move_ok_for_now:
        df = {'Name': [neck_eye_movements_data_frame['Name'][i]],
        'Expertise': [neck_eye_movements_data_frame['Expertise'][i]],
        'Acrobatics': [neck_eye_movements_data_frame['Acrobatics'][i]],
        'Anticipatory movements': [neck_eye_movements_data_frame['Anticipatory movements'][i]],
        'Compensatory movements': [neck_eye_movements_data_frame['Compensatory movements'][i]],
        'Spotting movements': [neck_eye_movements_data_frame['Spotting movements'][i]],
        'Movement detection': [neck_eye_movements_data_frame['Movement detection'][i]],
        'Blinks': [neck_eye_movements_data_frame['Blinks'][i]]}
        neck_eye_movements_table_temporary = pd.concat([neck_eye_movements_table_temporary, pd.DataFrame(df)])

    neck_eye_movements_data_frame = neck_eye_movements_table_temporary


    print("Mixed ANOVA for Anticipatory movements")
    out = pg.mixed_anova(data=neck_eye_movements_data_frame, dv='Anticipatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Compensatory movements")
    out = pg.mixed_anova(data=neck_eye_movements_data_frame, dv='Compensatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Spotting movements")
    out = pg.mixed_anova(data=neck_eye_movements_data_frame, dv='Spotting movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Movement detection")
    out = pg.mixed_anova(data=neck_eye_movements_data_frame, dv='Movement detection', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("Mixed ANOVA for Blinks")
    out = pg.mixed_anova(data=neck_eye_movements_data_frame, dv='Blinks', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Anticipatory movements")
    out = pg.pairwise_tukey(data=neck_eye_movements_data_frame, dv='Anticipatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Compensatory movements")
    out = pg.pairwise_tukey(data=neck_eye_movements_data_frame, dv='Compensatory movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Spotting movements")
    out = pg.pairwise_tukey(data=neck_eye_movements_data_frame, dv='Spotting movements', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Movement detection")
    out = pg.pairwise_tukey(data=neck_eye_movements_data_frame, dv='Movement detection', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')
    print("pairwise t-test for Blinks")
    out = pg.pairwise_tukey(data=neck_eye_movements_data_frame, dv='Blinks', within='Acrobatics', between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ----------------------------------- Heatmap spreading data frame = Mixed ANOVA -------------------------------------- #

if SPREADING_HEATMAP_FLAG:
    heatmaps_spreading_data_frame = pd.DataFrame(heatmaps_spreading_table[1:], columns=heatmaps_spreading_table[0])
    heatmaps_spreading_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/heatmaps_spreading_data_frame.csv")
    heatmaps_spreading_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Distance from the center of each point of the heatmap',
                                                               'Heat map 90th percentile'])
    for i in range(len(heatmaps_spreading_data_frame)):
        # if i in list_move_ok_for_now:
        df = {'Name': [heatmaps_spreading_data_frame['Name'][i]],
              'Expertise': [heatmaps_spreading_data_frame['Expertise'][i]],
              'Acrobatics': [heatmaps_spreading_data_frame['Acrobatics'][i]],
              'Distance from the center of each point of the heatmap': [heatmaps_spreading_data_frame['Distance from the center of each point of the heatmap'][i]],
              'Heat map 90th percentile': [heatmaps_spreading_data_frame['Heat map 90th percentile'][i]]}
        heatmaps_spreading_table_temporary = pd.concat([heatmaps_spreading_table_temporary, pd.DataFrame(df)])

    heatmaps_spreading_data_frame = heatmaps_spreading_table_temporary

    print("Mixed ANOVA for Heat map 90th percentile")
    out = pg.mixed_anova(data=heatmaps_spreading_data_frame, dv='Heat map 90th percentile', within='Acrobatics',
                         between='Expertise', subject='Name')
    print(f'{out}\n\n')

    print("pairwise t-test for Heat map 90th percentile")
    out = pg.pairwise_tukey(data=heatmaps_spreading_data_frame, dv='Heat map 90th percentile', within='Acrobatics',
                            between='Expertise', subject='Name')
    print(f'{out}\n\n')


# ---------------------------------------- Qualitative data frame = SPM1D -------------------------------------------- #

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

def plot_presence_all_at_the_same_time(presence_curves_per_athelte, move, xi_interp, output_file_name):

    colors = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 1, 5)]

    variable_names = ["Anticipatory movements", "Compensatory movements", "Spotting", "Movement detection", "Blink"]

    presence_curves_subelites = {variable_names[index_variable]: np.zeros((len(xi_interp, ))) for index_variable in range(len(variable_names))}
    presence_curves_elites = {variable_names[index_variable]: np.zeros((len(xi_interp, ))) for index_variable in range(len(variable_names))}
    for index_variable, variable_name in enumerate(variable_names):
        for i, name in enumerate(presence_curves_per_athelte.keys()):
            if name in subelite_names:
                if presence_curves_per_athelte[name][move] != []:
                    presence_curves_subelites[variable_names[index_variable]] = np.vstack((
                        presence_curves_subelites[variable_names[index_variable]],
                        presence_curves_per_athelte[name][move][index_variable]))
        for i, name in enumerate(presence_curves_per_athelte.keys()):
            if name in elite_names:
                if presence_curves_per_athelte[name][move] != []:
                    presence_curves_elites[variable_names[index_variable]] = np.vstack((
                        presence_curves_elites[variable_names[index_variable]],
                        presence_curves_per_athelte[name][move][index_variable]))

    presence_curves_subelites = {variable_names[index_variable]: presence_curves_subelites[variable_names[index_variable]][1:, :] for index_variable in range(len(variable_names))}
    presence_curves_elites = {variable_names[index_variable]: presence_curves_elites[variable_names[index_variable]][1:, :] for index_variable in range(len(variable_names))}

    mean_presence_curves_subelites = {variable_names[index_variable]:
                                          np.mean(presence_curves_subelites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    std_presence_curves_subelites = {variable_names[index_variable]:
                                          np.std(presence_curves_subelites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    mean_presence_curves_elites = {variable_names[index_variable]:
                                          np.mean(presence_curves_elites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    std_presence_curves_elites = {variable_names[index_variable]:
                                          np.std(presence_curves_elites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}

    fig, axs = plt.subplots(2, 1, figsize=(9, 6))
    for index_variable in range(len(variable_names)):
        axs[0].fill_between(xi_interp * 100,
                            mean_presence_curves_subelites[variable_names[index_variable]] - std_presence_curves_subelites[variable_names[index_variable]],
                            mean_presence_curves_subelites[variable_names[index_variable]] + std_presence_curves_subelites[variable_names[index_variable]],
                            alpha=0.2, facecolor=colors[index_variable])
        axs[0].plot(xi_interp * 100, mean_presence_curves_subelites[variable_names[index_variable]], color=colors[index_variable], label=variable_names[index_variable])
        axs[1].fill_between(xi_interp * 100,
                            mean_presence_curves_elites[variable_names[index_variable]] - std_presence_curves_elites[variable_names[index_variable]],
                            mean_presence_curves_elites[variable_names[index_variable]] + std_presence_curves_elites[variable_names[index_variable]],
                            alpha=0.2, facecolor=colors[index_variable])
        axs[1].plot(xi_interp * 100, mean_presence_curves_elites[variable_names[index_variable]], color=colors[index_variable], label=variable_names[index_variable])
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=3)
    axs[1].set_xlabel("Normalized time [%]")
    axs[0].set_ylabel("Subelites", fontsize=16)
    axs[1].set_ylabel("Elites", fontsize=16)
    plt.subplots_adjust(top=0.8)
    plt.suptitle(move)
    plt.savefig(output_file_name, dpi=300)
    # plt.show()
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

        plot_presence_all_at_the_same_time(presence_curves_per_athelte, move, xi_interp,
                          home_path + '/disk/Eye-tracking/plots/' + f'presence_all_{move}.png')







