# Wrapper to load in the ACS data

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from sklearn.model_selection import train_test_split


def get_data(test_size, acs_task, acs_year, acs_states, acs_horizon='1-Year', acs_survey='person', row_start=0,
             row_end=-1, col_start=0, col_end=-1):
    # test_size: percentage of data to be used by test dataset (e.g. 0.2 or 0.5)
    # acs_task options: employment, income, public_coverage, mobility, and travel_time. only employment is tested code.
    # acs_year: >= 2014. upper limit unclear.
    # acs_states: any list of state abbreviations e.g. ['NY']
    # acs_horizon: '1-Year' or '5-Year'
    # acs_survey: 'person' or 'household'

    global columns, group, label, features
    data_source = ACSDataSource(survey_year=acs_year, horizon=acs_horizon, survey=acs_survey)

    # this pulls in the raw data
    acs_data = data_source.get_data(states=acs_states, download=True)

    # this block pulls out the relevant data columns to one of the 5 following prediction tasks.
    # each one reads out a features vector, a label vector (what you're trying to learn in that particular task),
    # and a 'group' vector.
    # the 'group' vector is always race; each x is labeled 1-9 depending on the racial category listed in appendix
    # B of the paper https://arxiv.org/pdf/2108.04884.pdf

    # for some reason, they stripped all the column names from the data, so it's hard to see what they are.
    # I've added these back in, which is what the 'columns' variable is.
    # so far, I've only re-added it for the employment data though.
    if acs_task == 'employment':
        # label is True if adult is employed

        # columns of the feature vector
        columns = [
            'AGEP',
            'SCHL',
            'MAR',
            'RELP',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'SEX',
            'RAC1P',
        ]
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
    elif acs_task == 'income':
        # label is True if US working adults’ yearly income is above $50,000

        # columns of the feature vector
        columns = [
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            'SEX',
            'RAC1P',
        ]
        features, label, group = ACSIncome.df_to_numpy(acs_data)
    elif acs_task == 'public_coverage':
        # label True if low-income individual, not eligible for Medicare, has coverage from public health insurance.

        # coluns of the feature vector 
        columns = [
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'PINCP',
            'ESR',
            'ST',
            'FER',
            'RAC1P',
        ]
        features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    elif acs_task == 'mobility':

        columns = [
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'CIT',
            'MIL',
            'ANC',
            'NATIVITY',
            'RELP',
            'DEAR',
            'DEYE',
            'DREM',
            'RAC1P',
            'GCL',
            'COW',
            'ESR',
            'WKHP',
            'JWMNP',
            'PINCP',
        ]
        # label True if a young adult moved addresses in the last year.
        features, label, group = ACSMobility.df_to_numpy(acs_data)
    elif acs_task == 'travel_time':

        columns = [
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'MIG',
            'RELP',
            'RAC1P',
            'PUMA',
            'ST',
            'CIT',
            'OCCP',
            'JWTR',
            'POWPUMA',
            'POVPIP',
        ]
        # label True if a working adult has a travel time to work of greater than 20 minutes
        features, label, group = ACSTravelTime.df_to_numpy(acs_data)
    else:
        print("Invalid task")

    # the next bit pulls out just some subset of the total rows of data, as defined by your row_start and row_end,
    # column_start, and column_end variables.
    # currently, the number of rows it pulls is defined by rows_used, which can be input from the jupyter notebook
    # into this function.

    features = features[row_start:row_end, col_start:col_end]
    label = label[row_start:row_end]
    # label = label.astype(int)        #If label is boolean and you want to make it binary

    # the group variable is race; labeled 1-8 according to categories in Appendix B of paper
    # https://arxiv.org/pdf/2108.04884.pdf
    group = group[row_start:row_end]

    # next, pull out all the test/train data.
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group,
                                                                                 test_size=test_size, random_state=0)

    # for our purposes, we want the race groups to be included in the training data so here we shove it back onto
    # the data
    X_train = np.hstack((X_train, group_train[:, np.newaxis]))
    X_test = np.hstack((X_test, group_test[:, np.newaxis]))

    # making the training data into an actual pandas dataframe so that we can actually read it and see what things mean.
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    # Building the group functions ###

    # first building the race indicators
    group_functions = [lambda x, group_id=group_id: 1 if x['RAC1P'] == group_id else 0 for group_id in [1, 2, 6, 8, 9]]
    group_functions.append(
        lambda x: 1 if x['RAC1P'] == 3 or x['RAC1P'] == 4 or x['RAC1P'] == 5 or x['RAC1P'] == 7 else 0)
    group_indicators = ['White', 'Black or African American', 'Asian',
                        'Native Hawaiian, Native American, Native Alaskan, or Pacific Islander', 'Some Other Race',
                        'Two or More Races']

    # next, the sex indicators

    group_functions += [lambda x: 1 if x['SEX'] == 1 else 0, lambda x: 1 if x['SEX'] == 2 else 0]
    group_indicators += ['Male', 'Female']
    # finally, the age indicators

    min_age = 30
    mid_age = 50

    group_functions += [lambda x: 1 if x['AGEP'] < min_age else 0, lambda x: 1 if min_age <= x['AGEP'] < mid_age else 0,
                        lambda x: 1 if x['AGEP'] >= mid_age else 0]
    group_indicators += ['Young', 'Middle', 'Old']
    return [X_train, y_train, X_test, y_test, group_functions, group_indicators, min_age, mid_age]
