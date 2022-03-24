import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from sklearn.model_selection import train_test_split
import groupSettings
import sys
sys.path.append("..")
sys.path.append(".")

#### DO NOT LOOK AT THIS FILE PLEASE :) FOR RESEARCHY REASONS WE NEED TO PRETEND YOU DON'T HAVE ACCESS TO IT BUT
#### IRA COULDN'T FIGURE OUT HOW TO ENCRYPT IT IN A WAY THAT WOULD WORK ON EVERYONE'S SYSTEMS

def get_data():
    acs_task = groupSettings.acs_task
    acs_states = groupSettings.acs_states
    test_size = 0.3
    acs_year = 2018
    acs_horizon = '1-Year'
    acs_survey = 'person'
    row_start = 0
    row_end = -1
    col_start = 0
    col_end = -1
    data_source = ACSDataSource(survey_year=acs_year, horizon=acs_horizon, survey=acs_survey)
    columns, features, label, group = [],[],[],[]
    # this pulls in the raw data
    acs_data = data_source.get_data(states=acs_states, download=True)
    # columns of the feature vector
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

    features = features[row_start:row_end, col_start:col_end]
    label = label[row_start:row_end]
    group = group[row_start:row_end]
    div = 2
    X_train, all_test_X, y_train, all_test_y, group_train, group_test = train_test_split(features, label, group,
                                                                                 test_size=test_size, random_state=10)
    X_train = np.hstack((X_train, group_train[:, np.newaxis]))
    all_test_X = np.hstack((all_test_X, group_test[:, np.newaxis]))

    # making the training data into an actual pandas dataframe so that we can actually read it and see what things mean.
    X_train = pd.DataFrame(X_train, columns=columns)
    all_test_X = pd.DataFrame(all_test_X, columns=columns)

    validation_x = all_test_X.iloc[:len(all_test_X)//div]
    validation_y = all_test_y[:len(all_test_y)//div]

    return [X_train, y_train, validation_x, validation_y]