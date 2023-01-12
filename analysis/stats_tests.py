
import numpy as np
import pingouin as pg
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import spm1d

##########################################################################################
# run --- python stats_test.py > stats_output.txt ---  to save the output to a file
##########################################################################################


PRIMARY_ANALYSIS_FLAG = False
AOI_ANALYSIS_FLAG = False
NECK_EYE_ANALYSIS_FLAG = True
SPREADING_HEATMAP_FLAG = True
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

print("Faire un table du nombre d'essais de chaque par athlete")
subelite_names = []
elite_names = []
for i in range(len(primary_table)):
    if primary_table[i][1] == 'SubElite':
        if primary_table[i][0] not in subelite_names:
            subelite_names.append(primary_table[i][0])
    if primary_table[i][1] == 'Elite':
        if primary_table[i][0] not in elite_names:
            elite_names.append(primary_table[i][0])

# ------------------------------------ Primary data frame = Mixed Anova ---------------------------------------- #
if PRIMARY_ANALYSIS_FLAG:
    primary_data_frame = pd.DataFrame(primary_table[1:], columns=primary_table[0])
    primary_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/AllAnalysedMovesGrid.csv")

    # # Equal number of movements per group, unequal number of participants OK!
    # list_move_ok_for_now = [29, 30, 31, 32, 33,
    #                         39, 40, 41, 42, 43,
    #                         58, 59, 60, 61, 62,
    #                         68, 69, 70, 71, 72,
    #                         78, 79, 80, 81, 82,
    #                         83, 84, 85, 86, 87,
    #                         93, 94, 95, 96, 97,
    #                         98, 99, 100, 101, 102,
    #                         114, 115, 116, 117, 118,
    #                         122, 123, 124, 125, 126,
    #                         146, 147, 148, 149, 150,
    #                         156, 157, 158, 159, 160,
    #                         185, 186, 187, 188, 189,
    #                         193, 194, 195, 196, 197,
    #                         222, 223, 224, 225, 226,
    #                         228, 229, 230, 231, 232,
    #                         244, 245, 246, 247, 248,
    #                         251, 252, 253, 254, 255]

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
# trajectories_table
# add meanplots



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

if QUALITATIVE_ANALYSIS_FLAG:

    nb_interp_points = 500
    xi_interp = np.linspace(0, 1, nb_interp_points)
    trial_per_athlete_index = {}
    for i in range(1, len(qualitative_table)):
        if qualitative_table[i][0] not in trial_per_athlete_index.keys():
            trial_per_athlete_index[qualitative_table[i][0]] = [i]
        else:
            trial_per_athlete_index[qualitative_table[i][0]] += [i]

    presence_curves_per_athelte = {}
    for j, name in enumerate(trial_per_athlete_index.keys()):
        index_this_time = trial_per_athlete_index[name]
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

        presence_curves_per_athelte[name] = [anticipatory_curve/len(index_this_time),
                                             compensatory_curve/len(index_this_time),
                                             spotting_curve/len(index_this_time),
                                             movement_detection_curve/len(index_this_time),
                                             blink_curve/len(index_this_time)]


    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    colors = []
    subelites_anticipatory = []
    elites_anticipatory = []
    subelites_compensatory = []
    elites_compensatory = []
    subelites_spotting = []
    elites_spotting = []
    subelites_movement_detection = []
    elites_movement_detection = []
    subelites_blink = []
    elites_blink = []
    i_elites = 0
    i_subelites = 0
    for name in presence_curves_per_athelte.keys():
        if name in subelite_names:
            colors += [colors_subelites[i_subelites]]
            i_subelites += 1
            subelites_anticipatory += [presence_curves_per_athelte[name][0]]
            subelites_compensatory = [presence_curves_per_athelte[name][1]]
            subelites_spotting = [presence_curves_per_athelte[name][2]]
            subelites_movement_detection = [presence_curves_per_athelte[name][3]]
            subelites_blink = [presence_curves_per_athelte[name][4]]
        elif name in elite_names:
            colors += [colors_elites[i_elites]]
            i_elites += 1
            elites_anticipatory = [presence_curves_per_athelte[name][0]]
            elites_compensatory = [presence_curves_per_athelte[name][1]]
            elites_spotting = [presence_curves_per_athelte[name][2]]
            elites_movement_detection = [presence_curves_per_athelte[name][3]]
            elites_blink = [presence_curves_per_athelte[name][4]]
        else:
            print(f"Probleme de nom: {name} not recognised")


    fig, axs = plt.subplots(2, 1)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            axs[0].plot(xi_interp*100, presence_curves_per_athelte[name][0], color=colors[i], label=name)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            axs[1].plot(xi_interp*100, presence_curves_per_athelte[name][0], color=colors[i], label=name)

    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Normalized time [%]")
    plt.suptitle("Anticipatory movements")
    plt.savefig(home_path + '/disk/Eye-tracking/plots/' + 'anticiaptory_presence.png', dpi=300)

    fig, axs = plt.subplots(2, 1)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            axs[0].plot(xi_interp*100, presence_curves_per_athelte[name][1], color=colors[i], label=name)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            axs[1].plot(xi_interp*100, presence_curves_per_athelte[name][1], color=colors[i], label=name)

    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Normalized time [%]")
    plt.suptitle("Compensatory movements")
    plt.savefig(home_path + '/disk/Eye-tracking/plots/' + 'compensatory_presence.png', dpi=300)

    fig, axs = plt.subplots(2, 1)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            axs[0].plot(xi_interp*100, presence_curves_per_athelte[name][2], color=colors[i], label=name)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            axs[1].plot(xi_interp*100, presence_curves_per_athelte[name][2], color=colors[i], label=name)

    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Normalized time [%]")
    plt.suptitle("Spotting")
    plt.savefig(home_path + '/disk/Eye-tracking/plots/' + 'spotting_presence.png', dpi=300)

    fig, axs = plt.subplots(2, 1)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            axs[0].plot(xi_interp*100, presence_curves_per_athelte[name][3], color=colors[i], label=name)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            axs[1].plot(xi_interp*100, presence_curves_per_athelte[name][3], color=colors[i], label=name)

    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Normalized time [%]")
    plt.suptitle("Movement detection")
    plt.savefig(home_path + '/disk/Eye-tracking/plots/' + 'movement_detection_presence.png', dpi=300)

    fig, axs = plt.subplots(2, 1)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            axs[0].plot(xi_interp*100, presence_curves_per_athelte[name][4], color=colors[i], label=name)
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            axs[1].plot(xi_interp*100, presence_curves_per_athelte[name][4], color=colors[i], label=name)

    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel("Normalized time [%]")
    plt.suptitle("Blinks")
    plt.savefig(home_path + '/disk/Eye-tracking/plots/' + 'blink_presence.png', dpi=300)
    plt.show()

    print("Il faut exclure les moments ou il n'y a pas de variance...\nChanger subelites_anticipatory pour np.array et faire np.std()")

    # t = spm1d.stats.ttest2(subelites_anticipatory, elites_anticipatory, equal_var=False)
    # ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    # ti.plot()







