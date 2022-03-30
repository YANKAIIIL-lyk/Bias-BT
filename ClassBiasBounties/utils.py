import sys
from sklearn.tree import DecisionTreeClassifier

sys.path.append('dontlook')
from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
import uuid
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

V = 1  #


# 1,



def _build_model_g(x_conflict, y_h_win, dt_depth):
    vvprint("building g*")
    # learn the indices first, since this is an inefficient operation
    # indices = x.apply(group_function, axis=1) == 1

    # then pull the particular rows from the dataframe
    training_xs = x_conflict
    training_ys = y_h_win

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    dt.fit(training_xs, training_ys)
    vvprint("finished building g*")
    return dt


def build_model_h(x, y, group_function, dt_depth):
    vvprint("building h*")
    # learn the indices first, since this is an inefficient operation
    indices = x.apply(group_function, axis=1) == 1

    # then pull the particular rows from the dataframe
    training_xs = x[indices]
    training_ys = y[indices]

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    dt.fit(training_xs, training_ys)
    vvprint("finished building h*")
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
def argmax_gh_local(f, train_x, train_y, validation_x, validation_y, init_g, name_pfx, epsilon=0.001):
    # f is the current PDL

    g_temp_func = init_g
    h_temp_model = None

    # g_star_idx = train_x.apply(g_temp_func, axis=1)
    current_improvement = 0  # pass 1st improvement check
    round = 0
    round += 1
    vprint("round %d started" % round)
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
        vprint(
            "change of weighted error rate reduction: %.4f -> %.4f, error rate reduce by additional %.4f(hopefully positive)" % (
                current_improvement, new_improvement, (new_improvement - current_improvement)))
        # vprint("Reduction on weighted error rate %.4f" % (new_improvement - current_improvement))

        # validate using validation dataset
        if new_improvement > current_improvement + epsilon:

            vprint("round %d accepted" % round)
            current_improvement = new_improvement
            round += 1
            vprint("round %d started" % round)
            h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
            g_temp_model = build_model_g(f, h_temp_model, train_x, train_y)
            g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1, -1))[0]  # given srs of features x


        else:
            vprint("round %d failed" % round)

            break
    if round != 1:
        return name_pfx + "-" + str(uuid.uuid4()), g_temp_func, h_temp_model
    else:
        return None, None, None


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
    vprint("argmax gh pair name %s,\n maximum improvement over all gh pairs: %.3f " % (max_name, max_delta))
    return max_name, gh_pairs[max_name]


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


def generate_one_dim_groups(train_data, groups):
    for col in train_data.columns:
        if col != 'AGEP':
            g = g_gen(col)

            # def g_neg(x):
            #     return 1 - g(x)

            groups[col] = g
            # groups.append(g_neg)
    _generate_one_dim_group_age(groups)  # generate age groups separately because they don't follow one-hot pattern
    return groups


# %% @Yi Since AGEP does not use one-hot pattern, we need a separate logic to generate its one-dim groups
def _generate_one_dim_group_age(groups):
    def g0(x):
        if ((x['AGEP'] >= 0) and (x['AGEP'] < 13)):  # child
            return 1
        else:
            return 0

    def g1(x):
        if ((x['AGEP'] >= 13) and (x['AGEP'] < 20)):  # teen
            return 1
        else:
            return 0

    def g2(x):
        if ((x['AGEP'] >= 20) and (x['AGEP'] < 60)):  # adult
            return 1
        else:
            return 0

    def g3(x):
        if ((x['AGEP'] >= 60) and (x['AGEP'] < 100)):  # Senior
            return 1
        else:
            return 0

    groups["AGEP_L"] = g1
    groups["AGEP_M"] = g2
    groups["AGEP_H"] = g3


# %% @Yi Finds the N initial groups with the highest FP/FN rate, after applying initial_model to them
# Returns: dictionary groupname: F*R "F*R" g()
def find_top_N_initial_g(N, dataset_x, dataset_y, predict_func, groups, M=10, orderby=None):
    records = []
    # i = 0
    for g_name, g in groups.items():
        indices = dataset_x.apply(g, axis=1) == 1
        x_g = dataset_x[indices]
        y_g = dataset_y[indices]
        predicted = predict_func(x_g)
        cm = confusion_matrix(y_g, predicted)
        if orderby == "F*R":
            if (len(cm) != 1):
                TN = cm[0][0]
                FN = cm[1][0]
                TP = cm[1][1]
                FP = cm[0][1]
                FPR = FP / (FP + TN)
                FNR = FN / (TP + FN)
            else:  # fix bug when true-only or false-only
                FPR = 0
                FNR = 0

            vvprint("FPR for group %s equals %.6f" % (g_name, FPR))
            vvprint("FNR for group %s equals %.6f" % (g_name, FNR))

            records.append([g_name, max(FPR, FNR), "FPR" if FPR > FNR else "FNR"])
            # i += 1
        else:
            if (len(cm) != 1):
                TN = cm[0][0]
                FN = cm[1][0]
                TP = cm[1][1]
                FP = cm[0][1]
                ERR = (FP + FN) / (FP + TN + TP + FN)
                # FNR = FN / (TP + FN)
            else:  # fix edge case when true-only or false-only
                ERR = 0
                # FPR = 0
                # FNR = 0
            records.append([g_name, max(FPR, FNR), "ERR" if FPR > FNR else "FNR"])

    records.sort(key=lambda x: x[1])
    vprint(records)
    max_M = records[-M:].copy()  # The keys corresponding to the groups with the max N FN/FP rates (these
    random.shuffle(max_M)
    max_N = max_M[:N]

    # keys are the index of a group in groups)
    # vprint(max_N)

    result_g = dict()
    for tup in max_N:
        #         vprint("appending group " + str(idx))
        # result_g.append(groups[tup[0]])
        result_g[tup[0]] = (tup[1], tup[2], groups[tup[0]])  # groupname: F*R "F*R " g()
    return result_g


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

    # train_x['SEX'] = train_x['SEX'].apply(preprocess_SEX)
    # validation_x['SEX'] = validation_x['SEX'].apply(preprocess_SEX)

    # DIS(Disability recode) - binary data. Return 1 if with a disability, otherwise return 0.
    def preprocess_DIS(x):
        if (x == 1):
            return 1
        return 0

    # train_x['DIS'] = train_x['DIS'].apply(preprocess_DIS)
    # validation_x['DIS'] = validation_x['DIS'].apply(preprocess_DIS)

    # ESP(Employment status of parents) - 82% is 0, 8.3% is 1, all the other data ranges from 2 - 8
    # TODO: Need one hot encoding
    def preprocess_ESP(x):
        if (x != 0 and x != 1):
            return 2
        return x

    # train_x['ESP'] = train_x['ESP'].apply(preprocess_ESP)
    # validation_x['ESP'] = validation_x['ESP'].apply(preprocess_ESP)

    # MIG(Mobility status) - 88.5% is 1, 9.9% is 3. Since the other data only makes up to 1% of whole data set,
    # merge it into category 3. Then this feature becomes a binary feature.
    # Return 1 if original data is 1, otherwise return 0
    def preprocess_MIG(x):
        if (x == 1):
            return 1
        return 0

    # train_x['MIG'] = train_x['MIG'].apply(preprocess_MIG)
    # validation_x['MIG'] = validation_x['MIG'].apply(preprocess_MIG)

    # CIT(Citizenship status) - 78% data is 1, 12.2% data is 4. Other data will be merged.
    # return 1 -> 0, 4 -> 1, others -> 2
    # TODO: Need one hot encoding
    def preprocess_CIT(x):
        if (x == 1):
            return 0
        elif (x == 4):
            return 1
        return 2

    # train_x['CIT'] = train_x['CIT'].apply(preprocess_CIT)
    # validation_x['CIT'] = validation_x['CIT'].apply(preprocess_CIT)

    # MIL(Military service) - 76.7% data is 4, 17.8% data is 0. Other data will be merged.
    # return 0 -> 0, 4 -> 1, others -> 2
    # TODO: Need one hot encoding
    def preprocess_MIL(x):
        if (x == 0):
            return 0
        elif (x == 4):
            return 1
        return 2

    # train_x['MIL'] = train_x['MIL'].apply(preprocess_MIL)
    # validation_x['MIL'] = validation_x['MIL'].apply(preprocess_MIL)

    # ANC(Ancestry recode) - 1.6% data is 3, will be merged into 4
    # TODO: Need one hot encoding
    def preprocess_ANC(x):
        if (x == 3):
            return 4
        return x

    # train_x['ANC'] = train_x['ANC'].apply(preprocess_ANC)
    # validation_x['ANC'] = validation_x['ANC'].apply(preprocess_ANC)

    # NATIVITY(Nativity) - binary data. Return 1 if Native
    def preprocess_NATIVITY(x):
        if (x == 1):
            return 0
        return 1

    # train_x['NATIVITY'] = train_x['NATIVITY'].apply(preprocess_NATIVITY)
    # validation_x['NATIVITY'] = validation_x['NATIVITY'].apply(preprocess_NATIVITY)

    # RELP(Relationship) - 38.3% is 0, 24.5% is 2, 18.5% is 1, all the other values are below 3.5%. Merge them into
    # category 3
    # TODO: Need one hot encoding
    def preprocess_RELP(x):
        if (x not in [0, 1, 2]):
            return 3
        return x

    # train_x['RELP'] = train_x['RELP'].apply(preprocess_RELP)
    # validation_x['RELP'] = validation_x['RELP'].apply(preprocess_RELP)

    # DEAR, DEYE - binary data. Return 1 if Yes(1), otherwise 0.
    def preprocess_DEAR(x):
        if (x == 1):
            return 1
        return 0

    def preprocess_DEYE(x):
        return preprocess_DEAR(x)

    # train_x['DEAR'] = train_x['DEAR'].apply(preprocess_DEAR)
    # train_x['DEYE'] = train_x['DEYE'].apply(preprocess_DEYE)
    # validation_x['DEAR'] = validation_x['DEAR'].apply(preprocess_DEAR)
    # validation_x['DEYE'] = validation_x['DEYE'].apply(preprocess_DEYE)

    # DREM(Cognitive difficulty) - No preprocess needed
    # TODO: Need one hot encoding

    # RAC1P(Recoded detailed race code) - category 1, 2, 6, 8 includes 95% of the data. So all the other data will be
    # categorized into one category
    # TODO: Need one hot encoding
    def preprocess_RAC1P(x):
        if (x in [9, 3, 5, 7, 4]):
            return 0
        return x

    # train_x['RAC1P'] = train_x['RAC1P'].apply(preprocess_RAC1P)
    # validation_x['RAC1P'] = validation_x['RAC1P'].apply(preprocess_RAC1P)

    # make sure all the data are int
    train_x = pd.DataFrame(train_x, dtype=int)
    validation_x = pd.DataFrame(validation_x, dtype=int)

    # one-hot encode the training
    # features_to_encode = ['MAR', 'ESP', 'CIT', 'MIL', 'ANC', 'RELP', 'DREM', 'RAC1P']
    features_to_encode = ['MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'CIT', 'MIL', 'ANC', 'NATIVITY', 'RELP', 'DREM', 'RAC1P']
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


def simple_updater(f, g, group_name="g"):
    # if you want to change how h is trained, you can edit the below line.
    h = bountyHuntWrapper.build_model(train_x, train_y, g, dt_depth=10)
    # do not change anything beyond this point.
    if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)
        return True
    return False


def evaluation(current_pdl, train_x, train_y):
    vprint("per-group training errors of lastest version PDL:")
    vprint(current_pdl.train_errors[
               -1])  # this is the group error of each group on the initial PDL.
    # The ith element of f.train_errors is the group error of each group on
    # the ith version of the PDL.
    vprint("per-group testing errors of lastest version PDL:")
    vprint(current_pdl.test_errors[-1])  # group errors on validation set
    # vprint(f.predicates)  # all of the group functions that have been appended so far
    # vprint(f.leaves)  # all of the h functions appended so far
    vprint(
        current_pdl.pred_names)  # the names you passed in for each of the group functions, to more easily understand which are which.
    # %%
    # 5. Looking at the group error of the ith group over each round of updates:
    # Say you found a group at round 5 and you want to know how its group error
    # looked at previous or subsequent rounds. To do so, you can pull
    # `f.train_errors` or `f.test_errors` and look at the ith element of each
    # list as follows:
    target_group = -1  # this sets the group whose error you want to look at at each round to the initial model. If I wanted to look at the 1st group introduced, would change to a 1, e.g.
    vprint("latest group's training errors on all preivous PDLs:")
    vprint([e[-1] for e in current_pdl.train_errors])
    vprint("latest group's testing errors on all preivous PDLs:")
    vprint([e[-1] for e in current_pdl.test_errors])
    g_all = lambda x: 1  # here we define a group that just is all the data, replace as you see fit.
    bountyHuntWrapper.measure_group_error(current_pdl, g_all, train_x, train_y)

    pred_ys = validation_x.apply(current_pdl.predict, axis=1).to_numpy()
    errors = metrics.zero_one_loss(validation_y, pred_ys)
    vprint("latest PDL's overall testing  errors %.10f" % errors)


if __name__ == '__main__':
    # test
    def warn(*args, **kwargs):
        pass


    import warnings

    # from multiprocessing.pool import ThreadPool

    warnings.warn = warn

    TOP_N = 4
    epsilon = 0.001  # 0.001
    [train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()
    train_x, validation_x = preprocess(train_x, validation_x)

    initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
    initial_model.fit(train_x, train_y)
    groups_fcns = dict()  # name -> grp_fcn
    generate_one_dim_groups(train_x, groups_fcns)

    f = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)


    def g1(x):
        """
        Given an x, g should return either 0 or 1.
        :param x: input vector
        :return: 0 or 1; 1 if x belongs to group, 0 otherwise
        """

        # here is how to make a function g that returns 1 for all African American individuals
        if x['SCHL'] <= 15:
            return 1
        else:
            return 0


    simple_updater(f, g1, group_name="g1")


    def g2(x):
        """
        Given an x, g should return either 0 or 1.
        :param x: input vector
        :return: 0 or 1; 1 if x belongs to group, 0 otherwise
        """

        # here is how to make a function g that returns 1 for all African American individuals
        if x['AGEP'] < 16 or x['AGEP'] >= 60:
            return 1
        else:
            return 0


    simple_updater(f, g2, group_name="g2")


    def visitEachList(df):
        columns = list(df.columns)[2:]
        count = 1
        for column in columns:
            print(column)

            def aggr(column):
                def g(x):
                    if (x[column] == 1):
                        return 1
                    else:
                        return 0

                return g

            g = aggr(column)
            ret = simple_updater(f, g, "group" + str(count))
            if ret:
                print("Updated")
                evaluation(f, train_x, train_y)


    visitEachList(train_x)
    evaluation(f, train_x, train_y)

    f_predict_xN = lambda xN: xN.apply(f.predict, axis=1)
    retries_nr = 0
    updates_count = 0
    itr_count = 0
    # for i in range(int(1 / epsilon)):

    result_g = find_top_N_initial_g(TOP_N, train_x, train_y, f_predict_xN, groups_fcns, M=TOP_N, orderby="F*R")
    print("initial groups:")
    print(result_g)
    local_maxs = dict()
    process_list = []

    for col, v in result_g.items():
        print("%d-th iteration, finding %d-th gh pair" % (itr_count + 1, updates_count + 1))
        itr_count += 1
        init_g_err = "%s-%.4f%s" % (col, v[0], v[1])
        vprint("init_g_err %s" % init_g_err)
        name, g, h = argmax_gh_local(f, train_x, train_y, validation_x, validation_y, v[-1], init_g_err, epsilon)
        if (name is not None):
            vprint("find local max gh from %s" % name)
            # local_maxs[name] = (g, h)
            if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h.predict, train_x=train_x,
                                            train_y=train_y):
                updates_count += 1
                vprint("Running Update %d" % updates_count)
                bountyHuntWrapper.run_updates(f, g, h.predict, train_x, train_y, validation_x, validation_y,
                                              group_name=name)


    evaluation(f, train_x, train_y)
