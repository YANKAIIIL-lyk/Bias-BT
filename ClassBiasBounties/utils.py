import sys
from sklearn.tree import DecisionTreeClassifier

sys.path.append('dontlook')
from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
import uuid
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

V = True
def print(*args,**kwargs):
    if(V):
        print(*args,**kwargs)

def _build_model_g(x_conflict, y_h_win, dt_depth):
    print("building g*")
    # learn the indices first, since this is an inefficient operation
    # indices = x.apply(group_function, axis=1) == 1

    # then pull the particular rows from the dataframe
    training_xs = x_conflict
    training_ys = y_h_win

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    dt.fit(training_xs, training_ys)
    print("finished building g*")
    return dt


def build_model_h(x, y, group_function, dt_depth):
    print("building h*")
    # learn the indices first, since this is an inefficient operation
    indices = x.apply(group_function, axis=1) == 1

    # then pull the particular rows from the dataframe
    training_xs = x[indices]
    training_ys = y[indices]

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    dt.fit(training_xs, training_ys)
    print("finished building h*")
    return dt


def build_model_g(f, h_temp_model, train_x, train_y):
    # prepare data
    y_hat_h_temp = h_temp_model.predict(train_x)
    y_hat_f = train_x.apply(f.predict, axis=1)
    conflict_fh_idx = y_hat_f != y_hat_h_temp
    h_temp_correct_idx = train_y == y_hat_h_temp

    train_x_conflict_fh = train_x[conflict_fh_idx]
    train_y_conflict_fh = np.zeros(train_y.shape[0])
    train_y_conflict_fh[conflict_fh_idx & h_temp_correct_idx] = 1
    train_y_conflict_fh = train_y_conflict_fh[conflict_fh_idx]

    g_temp_model = _build_model_g(train_x_conflict_fh, train_y_conflict_fh, dt_depth=10)
    return g_temp_model




# %% impementation of 4.2.2 Algorithm 6 @Tao
# return (g*, h*)
def argmax_gh_local(f, train_x, train_y, validation_x, validation_y, init_g, epsilon=0.001):
    # f is the current PDL

    g_temp_func = init_g
    h_temp_model = None

    # g_star_idx = train_x.apply(g_temp_func, axis=1)
    current_improvement = -1  # pass 1st improvement check
    round = 0
    round += 1
    print("round %d started" % round)
    h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
    g_temp_model = build_model_g(f, h_temp_model, train_x, train_y)
    g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1, -1))[0]  # given srs of features x


    while (round < 1 / epsilon):

        # validate improvement
        g_weight = validation_x.apply(g_temp_func, axis=1).sum() / validation_x.shape[0]

        h_temp_pdl = bountyHuntWrapper.build_initial_pdl(h_temp_model, train_x, train_y, validation_x, validation_y)

        new_improvement = g_weight * (
                bountyHuntWrapper.measure_group_error(f, g_temp_func, validation_x, validation_y)
                - bountyHuntWrapper.measure_group_error(h_temp_pdl, g_temp_func, validation_x, validation_y))
        print("weighted error rate %.3f -> %.3f" % (current_improvement, new_improvement))
        print("improvement on weighted error rate %.4f" % (new_improvement - current_improvement))

        # validate using validation dataset
        if new_improvement > current_improvement + epsilon:

            print("round %d accepted" % round)
            current_improvement = new_improvement
            round += 1
            print("round %d started" % round)
            h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
            g_temp_model = build_model_g(f, h_temp_model, train_x, train_y)
            g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1, -1))[0]  # given srs of features x


        else:
            print("round %d failed" % round)

            break

    return str(uuid.uuid4()), g_temp_func, h_temp_model



# %% find max gh pairs among many local max gh pairs @Tao
def argmax_gh_global(gh_pairs, f, train_x, train_y, validation_x, validation_y):
    # assert(type(gh_pairs) == list)
    # initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
    # initial_model.fit(train_x, train_y)
    i = 0
    max_name = None
    max_delta = -1
    for k, gh in gh_pairs.items():
        g_func = gh[0]
        h_model = gh[1]
        g_weight = validation_x.apply(g_func, axis=1).sum() / validation_x.shape[0]

        h_temp_pdl = bountyHuntWrapper.build_initial_pdl(h_model, train_x, train_y, validation_x, validation_y)

        new_improvement = g_weight * (
                bountyHuntWrapper.measure_group_error(f, g_func, validation_x, validation_y)
                - bountyHuntWrapper.measure_group_error(h_temp_pdl, g_func, validation_x, validation_y))
        if new_improvement > max_delta:
            max_name = k
            max_delta = new_improvement
    print("argmax gh pair name %s,\n maximum improvement over all gh pairs: %.3f " % (max_name, max_delta))
    return gh_pairs[max_name]


# %% @Yi This is based on preprocessed dataset, where every column except for AGEP would be 1-hot encoded
# groups = []


# %% @Yi Generates all one dimensional groups
# groups is empty list
def g_gen(col):
    def g(x):
        if (x[col] == 1):
            return 1
        else:
            return 0
    return g

def generate_one_dim_groups(train_data,groups):
    for col in train_data.columns:
        if col != 'AGEP':
            g = g_gen( col)

            # def g_neg(x):
            #     return 1 - g(x)

            groups[col] = g
            # groups.append(g_neg)
    _generate_one_dim_group_age(groups)  # generate age groups separately because they don't follow one-hot pattern
    return groups

# %% @Yi Since AGEP does not use one-hot pattern, we need a separate logic to generate its one-dim groups
def _generate_one_dim_group_age(groups):
    def g1(x):
        if ((x['AGEP'] >= 0) and (x['AGEP'] < 30)):
            return 1
        else:
            return 0

    def g2(x):
        if ((x['AGEP'] >= 30) and (x['AGEP'] < 60)):
            return 1
        else:
            return 0

    def g3(x):
        if ((x['AGEP'] >= 60) and (x['AGEP'] < 100)):
            return 1
        else:
            return 0

    groups["AGEP_L"] = g1
    groups["AGEP_M"] = g2
    groups["AGEP_H"] = g3


# %% @Yi Finds the N initial groups with the highest FP/FN rate, after applying initial_model to them
# Returns: dictionary groupname: F*R "F*R" g()
def find_top_N_initial_g(N, dataset_x, dataset_y, initial_model, groups):
    records = []
    # i = 0
    for g_name, g in groups.items():
        indices = dataset_x.apply(g, axis=1) == 1
        validate_x_g = dataset_x[indices]
        validate_y_g = dataset_y[indices]
        predicted = initial_model.predict(validate_x_g)
        cm = confusion_matrix(validate_y_g, predicted)
        if(len(cm)!=1):
            TN = cm[0][0]
            FN = cm[1][0]
            TP = cm[1][1]
            FP = cm[0][1]
            FPR = FP / (FP + TN)
            FNR = FN / (TP + FN)
        else: # fix bug when true-only or false-only
            FPR = 0
            FNR = 0


        #         print("TP for group " + str(i) + " equals " + str(TP))
        #         print("TN for group " + str(i) + " equals " + str(TN))
        #         print("FP for group " + str(i) + " equals " + str(FP))
        #         print("FN for group " + str(i) + " equals " + str(FN))
        print("FPR for group %s equals %.6f" % ( g_name,  FPR ) )
        print("FNR for group %s equals %.6f" % (g_name, FNR))

        records.append([g_name, max(FPR, FNR), "FPR" if FPR>FNR else "FNR"])
        # i += 1

    records.sort(key=lambda x: x[1])
    print(records)
    max_N = records[-N:]  # The keys corresponding to the groups with the max N FN/FP rates (these
    # keys are the index of a group in groups)
    # print(max_N)

    result_g = dict()
    for tup in max_N:
        #         print("appending group " + str(idx))
        # result_g.append(groups[tup[0]])
        result_g[tup[0]] = (tup[1], tup[2], groups[tup[0]]) # groupname: F*R "F*R " g()
    return result_g
# groups = []
# def fun_gen():
#     ll = ["Tao", "Luo", "Chen", "Zhao"]
#     for l in ll:
#         def g():
#             return l
#
#         groups.append(g)
#

'''
This method will preprocess the training set and validation set. Drop all the outliers first. For the categorical
features, do one-hot encoding
'''
def preprocess(train_x, validation_x):
    # AGEP(age) - Numerical data, no preprocessing is needed
    # SCHL(Educational attainment) - Categorical data, but it's okay to preserve the natural order here

    # MAR(Marital status) - Categorical data, 85% of data is 1 or 5.
    # TODO: Need one hot encoding
    def preprocess_SEX(x):
        if (x == 1):
            return 1
        return 0

    train_x['SEX'] = train_x['SEX'].apply(preprocess_SEX)
    validation_x['SEX'] = validation_x['SEX'].apply(preprocess_SEX)

    # DIS(Disability recode) - binary data. Return 1 if with a disability, otherwise return 0.
    def preprocess_DIS(x):
        if (x == 1):
            return 1
        return 0

    train_x['DIS'] = train_x['DIS'].apply(preprocess_DIS)
    validation_x['DIS'] = validation_x['DIS'].apply(preprocess_DIS)

    # ESP(Employment status of parents) - 82% is 0, 8.3% is 1, all the other data ranges from 2 - 8
    # TODO: Need one hot encoding
    def preprocess_ESP(x):
        if (x != 0 and x != 1):
            return 2
        return x

    train_x['ESP'] = train_x['ESP'].apply(preprocess_ESP)
    validation_x['ESP'] = validation_x['ESP'].apply(preprocess_ESP)

    # MIG(Mobility status) - 88.5% is 1, 9.9% is 3. Since the other data only makes up to 1% of whole data set,
    # merge it into category 3. Then this feature becomes a binary feature.
    # Return 1 if original data is 1, otherwise return 0
    def preprocess_MIG(x):
        if (x == 1):
            return 1
        return 0

    train_x['MIG'] = train_x['MIG'].apply(preprocess_MIG)
    validation_x['MIG'] = validation_x['MIG'].apply(preprocess_MIG)

    # CIT(Citizenship status) - 78% data is 1, 12.2% data is 4. Other data will be merged.
    # return 1 -> 0, 4 -> 1, others -> 2
    # TODO: Need one hot encoding
    def preprocess_CIT(x):
        if (x == 1):
            return 0
        elif (x == 4):
            return 1
        return 2

    train_x['CIT'] = train_x['CIT'].apply(preprocess_CIT)
    validation_x['CIT'] = validation_x['CIT'].apply(preprocess_CIT)

    # MIL(Military service) - 76.7% data is 4, 17.8% data is 0. Other data will be merged.
    # return 0 -> 0, 4 -> 1, others -> 2
    # TODO: Need one hot encoding
    def preprocess_MIL(x):
        if (x == 0):
            return 0
        elif (x == 4):
            return 1
        return 2

    train_x['MIL'] = train_x['MIL'].apply(preprocess_MIL)
    validation_x['MIL'] = validation_x['MIL'].apply(preprocess_MIL)

    # ANC(Ancestry recode) - 1.6% data is 3, will be merged into 4
    # TODO: Need one hot encoding
    def preprocess_ANC(x):
        if (x == 3):
            return 4
        return x

    train_x['ANC'] = train_x['ANC'].apply(preprocess_ANC)
    validation_x['ANC'] = validation_x['ANC'].apply(preprocess_ANC)

    # NATIVITY(Nativity) - binary data. Return 1 if Native
    def preprocess_NATIVITY(x):
        if (x == 1):
            return 0
        return 1

    train_x['NATIVITY'] = train_x['NATIVITY'].apply(preprocess_NATIVITY)
    validation_x['NATIVITY'] = validation_x['NATIVITY'].apply(preprocess_NATIVITY)

    # RELP(Relationship) - 38.3% is 0, 24.5% is 2, 18.5% is 1, all the other values are below 3.5%. Merge them into
    # category 3
    # TODO: Need one hot encoding
    def preprocess_RELP(x):
        if (x not in [0, 1, 2]):
            return 3
        return x

    train_x['RELP'] = train_x['RELP'].apply(preprocess_RELP)
    validation_x['RELP'] = validation_x['RELP'].apply(preprocess_RELP)

    # DEAR, DEYE - binary data. Return 1 if Yes(1), otherwise 0.
    def preprocess_DEAR(x):
        if (x == 1):
            return 1
        return 0

    def preprocess_DEYE(x):
        return preprocess_DEAR(x)

    train_x['DEAR'] = train_x['DEAR'].apply(preprocess_DEAR)
    train_x['DEYE'] = train_x['DEYE'].apply(preprocess_DEYE)
    validation_x['DEAR'] = validation_x['DEAR'].apply(preprocess_DEAR)
    validation_x['DEYE'] = validation_x['DEYE'].apply(preprocess_DEYE)

    # DREM(Cognitive difficulty) - No preprocess needed
    # TODO: Need one hot encoding

    # RAC1P(Recoded detailed race code) - category 1, 2, 6, 8 includes 95% of the data. So all the other data will be
    # categorized into one category
    # TODO: Need one hot encoding
    def preprocess_RAC1P(x):
        if (x in [9, 3, 5, 7, 4]):
            return 0
        return x

    train_x['RAC1P'] = train_x['RAC1P'].apply(preprocess_RAC1P)
    validation_x['RAC1P'] = validation_x['RAC1P'].apply(preprocess_RAC1P)

    # make sure all the data are int
    train_x = pd.DataFrame(train_x, dtype=int)
    validation_x = pd.DataFrame(validation_x, dtype=int)

    # one-hot encode the training
    features_to_encode = ['MAR', 'ESP', 'CIT', 'MIL', 'ANC', 'RELP', 'DREM', 'RAC1P']
    encoder = OneHotEncoder(handle_unknown='ignore')
    train_x_encoded = pd.DataFrame(encoder.fit_transform(train_x[features_to_encode]).toarray(), \
                                   columns=encoder.get_feature_names_out(features_to_encode))
    train_x = train_x.drop(columns=features_to_encode, axis=1)
    train_x = train_x.join(train_x_encoded)
    train_x = pd.DataFrame(train_x, dtype=int)

    # one-hot encode the validation
    encoder = OneHotEncoder(handle_unknown='ignore')
    validation_x_encoded = pd.DataFrame(encoder.fit_transform(validation_x[features_to_encode]).toarray(), \
                                        columns=encoder.get_feature_names_out(features_to_encode))
    validation_x = validation_x.drop(columns=features_to_encode, axis=1)
    validation_x = validation_x.join(validation_x_encoded)
    validation_x = pd.DataFrame(validation_x, dtype=int)

    return train_x, validation_x

if __name__ == '__main__':
    # test
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn

    from pprint import pprint
    TOP_N = 3
    [train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()
    train_x, validation_x = preprocess(train_x, validation_x)

    initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
    initial_model.fit(train_x, train_y)
    groups_fcns = dict() # name -> grp_fcn
    generate_one_dim_groups(train_x,groups_fcns)
    result_g = find_top_N_initial_g(TOP_N, train_x, train_y, initial_model, groups_fcns)
    pprint(result_g)



    f = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)

    local_maxs = dict()
    for col,v in result_g.items():
        name_uuid, g, h = argmax_gh_local(f, train_x, train_y, validation_x, validation_y,v[-1])
        local_opt_id = "%s-%.3f%s-%s" % (col, v[0],v[1], name_uuid)
        print("find local max gh from %s" % local_opt_id)

        local_maxs[local_opt_id] = (g, h)


    g, h = argmax_gh_global(local_maxs, f, train_x, train_y, validation_x, validation_y)



    def updater(g, h, group_name="g"):
        # do not alter this code
        if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
            print("Running Update")
            bountyHuntWrapper.run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)

    updater(g, h.predict, group_name="g")
    preds = train_x.apply(f.predict, axis=1)
    # 2. Getting the zero-one loss of a model restricted to a group you have defined.
    # g = lambda x: 1  # here we define a group that just is all the data, replace as you see fit.
    bountyHuntWrapper.measure_group_error(f, g, train_x, train_y)
    # %%
    # 3. You can view the training data by calling `train_x`. If you want to only view the data for a single group defined by your group function, you can run the following:

    # replace g with whatever your group is
    indices = train_x.apply(g, axis=1) == 1
    xs = train_x[indices]
    ys = train_y[indices]
    # %%
    # 4. Inspecting the existing PDL: The PDL is stored as an object, and
    # tracks its training errors, validation set errors, and the group
    # functions that are used in lists where the ith element is the group
    # errors of all groups discovered so far on the ith node in the PDL.
    # If you are more curious about the implementation, you can look at the
    # model.py file in the codebase, which doesn't contain anything you can
    # use to adaptively modify your code. (But lives in the same folder as the
    # rest of the codebase just to make importing things easier)

    # f is the current model
    print(f.train_errors)  # group errors on training set.
    print(f.train_errors[
              0])  # this is the group error of each group on the initial PDL. The ith element of f.train_errors is the group error of each group on the ith version of the PDL.
    print(f.test_errors)  # group errors on validation set
    print(f.predicates)  # all of the group functions that have been appended so far
    print(f.leaves)  # all of the h functions appended so far
    print(
        f.pred_names)  # the names you passed in for each of the group functions, to more easily understand which are which.

    # %%
    # 5. Looking at the group error of the ith group over each round of updates:
    # Say you found a group at round 5 and you want to know how its group error
    # looked at previous or subsequent rounds. To do so, you can pull
    # `f.train_errors` or `f.test_errors` and look at the ith element of each
    # list as follows:

    target_group = 0  # this sets the group whose error you want to look at at each round to the initial model. If I wanted to look at the 1st group introduced, would change to a 1, e.g.

    group_errs = [e[target_group] for e in f.train_errors]
    print(group_errs)
