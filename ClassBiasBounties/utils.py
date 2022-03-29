import sys
from sklearn.tree import DecisionTreeClassifier

sys.path.append('dontlook')
from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
import uuid
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd



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

    h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
    g_temp_model = build_model_g(f, h_temp_model, train_x, train_y)
    g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1, -1))[0]  # given srs of features x
    round += 1
    print("round %d started" % round)

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

            h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
            g_temp_model = build_model_g(f, h_temp_model, train_x, train_y)
            g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1, -1))[0]  # given srs of features x
            round += 1
            print("round %d started" % round)

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
# Returns: a list containing the N initial groups with the highest FP/FN rate
def find_top_N_initial_g(N, dataset_x, dataset_y, initial_model, groups):
    records = []
    # i = 0
    for g_name, g in groups.items():
        indices = dataset_x.apply(g, axis=1) == 1
        validate_x_g = dataset_x[indices]
        validate_y_g = dataset_y[indices]
        predicted = initial_model.predict(validate_x_g)
        cm = confusion_matrix(validate_y_g, predicted)

        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)

        #         print("TP for group " + str(i) + " equals " + str(TP))
        #         print("TN for group " + str(i) + " equals " + str(TN))
        #         print("FP for group " + str(i) + " equals " + str(FP))
        #         print("FN for group " + str(i) + " equals " + str(FN))
        print("FPR for group " + g_name + " equals " + str(FPR))
        print("FNR for group " + g_name + " equals " + str(FNR))

        records.append([g_name, max(FPR, FNR)])
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
        result_g[tup[0]] = (tup[1], groups[tup[0]])
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

if __name__ == '__main__':
    # test
    from pprint import pprint
    [train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()

    initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
    initial_model.fit(train_x, train_y)
    groups_fcns = dict() # name -> grp_fcn
    generate_one_dim_groups(train_x,groups_fcns)
    result_g = find_top_N_initial_g(4, train_x, train_y, initial_model, groups_fcns)
    pprint(result_g)
