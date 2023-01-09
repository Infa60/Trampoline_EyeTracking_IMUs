
import numpy as np
import pingouin as pg
import pickle
import os
import pandas as pd

##########################################################################################
# run --- python stats_test.py > stats_output.txt ---  to save the output to a file
##########################################################################################


GENRATE_DATA_FRAME_FLAG = True

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


# ------------------------------------ Primary data frame = Mixed Anova ---------------------------------------- #
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



primary_data_frame_temporary = pd.DataFrame(columns=primary_table[0])
num_rows = 0
for i in range(len(primary_data_frame)):
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






















