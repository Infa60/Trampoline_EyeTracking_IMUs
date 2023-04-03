
import numpy as np
import pingouin as pg


def test_non_normality_impact_on_results(data_frame, move_list):

    # Normality of the data
    for i_move, move in enumerate(move_list):
        for expertise in ['SubElite', 'Elite']:
            print("\nShapiro-Wilk normality test for " + move + " for " + expertise + " athletes")
            print("Summary for " + move + ' ' + expertise + ' : ')
            for metrics in data_frame.keys():
                if metrics not in ['Name', 'Expertise', 'Acrobatics', 'Fixations duration absolute',
                                   'Quiet eye duration absolute', 'Maximum eye amplitude', 'Maximum neck amplitude',
                                   'Distance from the center of each point of the heatmap', 'Heat map 90th percentile']:
                    out = pg.normality(data_frame.loc[(data_frame['Acrobatics'] == move) & (data_frame['Expertise'] == expertise), metrics])
                    print(metrics + " : ", np.array(out.normal.array))
    print()

    # Non parametric tests - Acrobatics factor was fixed
    for i_move, move in enumerate(move_list):
        for metrics in data_frame.keys():
            if metrics not in ['Name', 'Expertise', 'Acrobatics', 'Fixations duration absolute',
                               'Quiet eye duration absolute', 'Maximum eye amplitude', 'Maximum neck amplitude',
                               'Distance from the center of each point of the heatmap', 'Heat map 90th percentile']:
                print("Mann-Whitney test for " + metrics + ' ' + move)
                out = pg.mwu(data_frame.loc[(data_frame['Acrobatics'] == move) & (data_frame['Expertise'] == 'SubElite'), metrics],
                             data_frame.loc[(data_frame['Acrobatics'] == move) & (data_frame['Expertise'] == 'Elite'), metrics])
                print(f'{out}\n\n')
    print()

    # Non parametric tests - Expertise factor was fixed
    for expertise in ['SubElite', 'Elite']:
        for metrics in data_frame.keys():
            if metrics not in ['Name', 'Expertise', 'Acrobatics', 'Fixations duration absolute',
                               'Quiet eye duration absolute', 'Maximum eye amplitude', 'Maximum neck amplitude',
                               'Distance from the center of each point of the heatmap', 'Heat map 90th percentile']:
                print("Freedman test for " + metrics + ' ' + expertise)
                out = pg.friedman(data=data_frame.loc[data_frame['Expertise'] == expertise], dv=metrics, within='Acrobatics', subject='Name')
                print(f'{out}\n\n')
    print()

    return

