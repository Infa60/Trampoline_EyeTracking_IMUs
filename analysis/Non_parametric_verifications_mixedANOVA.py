
import numpy as np
import pingouin as pg
import pickle
import pandas as pd


def test_non_normality_impact_on_results(primary_data_frame, move_list):

    # Normality of the data
    for i_move, move in enumerate(move_list):
        for expertise in ['SubElite', 'Elite']:
            print("Shapiro-Wilk normality test for Fixations duration relative " + move + " for " + expertise + " athletes")
            out_fixation_duration = pg.normality(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == expertise), 'Fixations duration relative'])
            print(f'{out_fixation_duration}\n\n')
            print("Shapiro-Wilk normality test for Number of fixations")
            out_fixation_number = pg.normality(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == expertise), 'Number of fixations'])
            print(f'{out_fixation_number}\n\n')
            print("Shapiro-Wilk normality test for Quiet eye duration relative")
            out_QE_duration = pg.normality(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == expertise), 'Quiet eye duration relative'])
            print(f'{out_QE_duration}\n\n')
            print("Shapiro-Wilk normality test for Eye amplitude")
            out_eye_amplitude = pg.normality(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == expertise), 'Eye amplitude'])
            print(f'{out_eye_amplitude}\n\n')
            print("Shapiro-Wilk normality test for Neck amplitude")
            out_neck_amplitude = pg.normality(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == expertise), 'Neck amplitude'])
            print(f'{out_neck_amplitude}\n\n')

            print("Summary for " + move + ' ' + expertise + ' : ')
            print("Fixations duration relative : ", np.array(out_fixation_duration.normal.array))
            print("Number of fixations : ", np.array(out_fixation_number.normal.array))
            print("Quiet eye duration relative : ", np.array(out_QE_duration.normal.array))
            print("Eye amplitude : ", np.array(out_eye_amplitude.normal.array))
            print("Neck amplitude : ", np.array(out_neck_amplitude.normal.array))


    # Non parametric tests - Acrobatics factor was fixed
    for i_move, move in enumerate(move_list):
        print("Mann-Whitney test for Fixations duration relative " + move)
        out = pg.mwu(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'SubElite'), 'Fixations duration relative'],
                     primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'Elite'), 'Fixations duration relative'])
        print(f'{out}\n\n')
        print("Mann-Whitney test for Number of fixations " + move)
        out = pg.mwu(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'SubElite'), 'Number of fixations'],
                        primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'Elite'), 'Number of fixations'])
        print(f'{out}\n\n')
        print("Mann-Whitney test for Quiet eye duration relative " + move)
        out = pg.mwu(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'SubElite'), 'Quiet eye duration relative'],
                        primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'Elite'), 'Quiet eye duration relative'])
        print(f'{out}\n\n')
        print("Mann-Whitney test for Eye amplitude " + move)
        out = pg.mwu(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'SubElite'), 'Eye amplitude'],
                        primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'Elite'), 'Eye amplitude'])
        print(f'{out}\n\n')
        print("Mann-Whitney test for Neck amplitude " + move)
        out = pg.mwu(primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'SubElite'), 'Neck amplitude'],
                        primary_data_frame.loc[(primary_data_frame['Acrobatics'] == move) & (primary_data_frame['Expertise'] == 'Elite'), 'Neck amplitude'])
        print(f'{out}\n\n')


    # Non parametric tests - Expertise factor was fixed
    for expertise in ['SubElite', 'Elite']:
        print("Freedman test for Fixations duration relative " + expertise)
        out = pg.friedman(data=primary_data_frame.loc[primary_data_frame['Expertise'] == expertise], dv='Fixations duration relative', within='Acrobatics', subject='Name')
        print(f'{out}\n\n')
        print("Freedman test for Number of fixations " + expertise)
        out = pg.friedman(data=primary_data_frame.loc[primary_data_frame['Expertise'] == expertise], dv='Number of fixations', within='Acrobatics', subject='Name')
        print(f'{out}\n\n')
        print("Freedman test for Quiet eye duration relative " + expertise)
        out = pg.friedman(data=primary_data_frame.loc[primary_data_frame['Expertise'] == expertise], dv='Quiet eye duration relative', within='Acrobatics', subject='Name')
        print(f'{out}\n\n')
        print("Freedman test for Eye amplitude " + expertise)
        out = pg.friedman(data=primary_data_frame.loc[primary_data_frame['Expertise'] == expertise], dv='Eye amplitude', within='Acrobatics', subject='Name')
        print(f'{out}\n\n')
        print("Freedman test for Neck amplitude " + expertise)
        out = pg.friedman(data=primary_data_frame.loc[primary_data_frame['Expertise'] == expertise], dv='Neck amplitude', within='Acrobatics', subject='Name')
        print(f'{out}\n\n')

    return

