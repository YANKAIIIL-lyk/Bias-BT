#%%
import sys
from sklearn.tree import DecisionTreeClassifier
sys.path.append('dontlook')
from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
import uuid


import numpy as np
import pandas as pd
#%%
[train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()


# columns of the feature vector
columns = [
#  age (AGEP), education (SCHL), marital status (MAR), relationship (RELP),
    'AGEP', # 0-99 num
    'SCHL', # 1-24 na  num?
    'MAR', # 1-5 cate
    'RELP', # 0-17  cate

# disability status (DIS), parent employee status (ESP), citizen status (CIT),
    'DIS', # 1,2 bin
    'ESP', # 1-8 na cate
    'CIT', # 1-5 cate

# mobility status (MIG), military service (MIL), ancestry record (ANC), nation of origin (NATIVITY), hearing difficulty (DEAR),
    'MIG', # cate
    'MIL', # na 1-4 cate
    'ANC', # cate
    'NATIVITY', # 1,2 bin
    'DEAR', # 1,2 bin

# visual dificulty (DEYE), learning disability (DREM), sex (SEX), and race (RAC1P).
    'DEYE', # 1,2 bin
    'DREM', # na 1,2 bin?
    'SEX', # 1, 2 bin
    'RAC1P', # 1-9 cate
]

#%%
initial_model = DecisionTreeClassifier(max_depth = 1, random_state=0)
initial_model.fit(train_x, train_y)
my_pdl = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)

#%%
# group only depends on x
# African American group
def g1(x):
    """
    Given an x, g should return either 0 or 1.
    :param x: input vector
    :return: 0 or 1; 1 if x belongs to group, 0 otherwise
    """
    # print(x)
    # here is how to make a function g that returns 1 for all African American individuals
    if x['RAC1P'] == 2:
        return 1
    else:
        return 0

def g1_neg(x):
    """
    Given an x, g should return either 0 or 1.
    :param x: input vector
    :return: 0 or 1; 1 if x belongs to group, 0 otherwise
    """
    # print(x)
    # here is how to make a function g that returns 1 for all African American individuals
    return 1 - g1(x)

#%% skip examples
# a group based on f
def g_(f,train_x, train_y):
    # f is the current PDL
    preds = train_x.apply(f.predict, axis=1)
    xs = train_x[preds == 1]
    ys = train_y[preds == 1]
    dt = DecisionTreeClassifier(max_depth = 1, random_state=0)
    dt.fit(xs, ys)
    def g(x):
        # g should take as input a SINGLE x and return 0 or 1 for it.
        # if we call dt.predict on x it will break because the dimensions of x are wrong, so we have to reshape it and reshape the output.
        # this is not particularly efficient, so if you have better ways of doing this go for it. :)
        y = dt.predict(np.array(x).reshape(1,-1))
        return y[0]
    return g

# if you wanted to build a particular g using the above, you could use the following line.
g = g_(my_pdl, train_x, train_y)



#%% based on 4.2.2 Algorithm 6
# helper func
def build_model_g(x_conflict, y_h_win,  dt_depth):
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


def opt_g(f, h_temp_model, train_x, train_y):
    y_hat_h_temp = h_temp_model.predict(train_x)
    # y = h_temp_model.predict(np.array(train_x).reshape(1, -1)) # todo
    y_hat_f = train_x.apply(f.predict, axis=1)
    conflict_fh_idx = y_hat_f != y_hat_h_temp
    h_temp_correct_idx = train_y == y_hat_h_temp
    # f_win_idx = y_hat_f == y_hat_h_temp
    train_x_conflict_fh = train_x[conflict_fh_idx]
    train_y_conflict_fh = np.zeros(train_y.shape[0])
    train_y_conflict_fh[conflict_fh_idx & h_temp_correct_idx] = 1
    train_y_conflict_fh = train_y_conflict_fh[conflict_fh_idx]
    g_temp_model = build_model_g(train_x_conflict_fh, train_y_conflict_fh, dt_depth=10)
    return g_temp_model


def argmax_gh_global(gh_pairs, f, validation_x, validation_y):
    # assert(type(gh_pairs) == list)
    # initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
    # initial_model.fit(train_x, train_y)
    i = 0
    max_name = None;
    max_delta = -1;
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
    print("argmax gh pair name %s,\n maximum improvement over all gh pairs: %.3f " % (max_name, max_delta) )
    return gh_pairs[max_name]

#%%
# return (g*, h*)
def argmax_gh_local(f, train_x, train_y, init_g):

    # f is the current PDL
    epsilon = 0.001
    g_temp_func = init_g
    h_temp_model = None

    # g_star_idx = train_x.apply(g_temp_func, axis=1)
    current_improvement = -1 # pass 1st improvement check
    round = 0
    while (round < 1/ epsilon):
        if h_temp_model is None:
            h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
            g_temp_model = opt_g(f, h_temp_model, train_x, train_y)
            g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1,-1))[0] # given srs of features x


            round+= 1
            print("round %d" % round)
        # validate improvement
        g_weight = validation_x.apply(g_temp_func,axis=1).sum() / validation_x.shape[0]
        # initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
        # initial_model.fit(train_x, train_y)
        h_temp_pdl = bountyHuntWrapper.build_initial_pdl(h_temp_model, train_x, train_y, validation_x, validation_y)


        new_improvement = g_weight * (
                    bountyHuntWrapper.measure_group_error(f, g_temp_func, validation_x, validation_y)
                    - bountyHuntWrapper.measure_group_error(h_temp_pdl, g_temp_func, validation_x, validation_y))
        print("weighted error rate %.3f -> %.3f" %  (current_improvement, new_improvement ) )
        print("improvement on weighted error rate %.4f" % (new_improvement - current_improvement) )

        # validate using validation dataset
        if new_improvement > current_improvement + epsilon:
            round += 1
            print("improve round %d" % round)

            current_improvement = new_improvement

            h_temp_model = build_model_h(train_x, train_y, g_temp_func, dt_depth=10)
            g_temp_model = opt_g(f, h_temp_model, train_x, train_y)
            g_temp_func = lambda srs: g_temp_model.predict(srs.values.reshape(1,-1))[0] # given srs of features x


        else:
            break


    # def g(x):
    #     # g should take as input a SINGLE x and return 0 or 1 for it.
    #     # if we call dt.predict on x it will break because the dimensions of x are wrong, so we have to reshape it and reshape the output.
    #     # this is not particularly efficient, so if you have better ways of doing this go for it. :)
    #     y = g_temp_func(x)
    #     return y
    return str(uuid.uuid4()), g_temp_func, h_temp_model



#%%
# if you wanted to build a particular g using the above, you could use the following line.
local_maxs = dict()
name, g, h = argmax_gh_local(my_pdl, train_x, train_y, g1)
local_maxs[name]  = (g, h)

#%%
name_neg, g_neg, h_neg = argmax_gh_local(my_pdl, train_x, train_y, g1_neg)
local_maxs[name_neg]  = (g_neg, h_neg)


#%%

g, h = argmax_gh_global(local_maxs, my_pdl, validation_x, validation_y)


#%%
# not used
def simple_updater(g,group_name = "g"):

    # if you want to change how h is trained, you can edit the below line.
    h = bountyHuntWrapper.build_model(train_x, train_y, g, dt_depth=10)
    # do not change anything beyond this point.
    if bountyHuntWrapper.run_checks(my_pdl, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(my_pdl, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)

def updater(g, h, group_name="g"):
    # do not alter this code
    if bountyHuntWrapper.run_checks(my_pdl, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(my_pdl, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)

#%%
# simple_updater(g,group_name="g1")
updater(g, h.predict, group_name="g")

## Saving Your Model

# import dill as pickle # you will probably need to install dill, which you do w pip install dill in your command line
# with open('pdl.pkl', 'wb') as pickle_file:
#     pickle.dump(f, pickle_file)
#
# with open('pdl.pkl', 'rb') as pickle_file:
#     content = pickle.load(pickle_file)
#

#%%
# Data Exploration

preds = train_x.apply(my_pdl.predict, axis=1)

#%%
# 2. Getting the zero-one loss of a model restricted to a group you have defined.
g = lambda x: 1 #here we define a group that just is all the data, replace as you see fit.
bountyHuntWrapper.measure_group_error(my_pdl, g, train_x, train_y)
#%%
# 3. You can view the training data by calling `train_x`. If you want to only view the data for a single group defined by your group function, you can run the following:

# replace g with whatever your group is
indices = train_x.apply(g, axis=1) == 1
xs = train_x[indices]
ys = train_y[indices]
#%%
# 4. Inspecting the existing PDL: The PDL is stored as an object, and
# tracks its training errors, validation set errors, and the group
# functions that are used in lists where the ith element is the group
# errors of all groups discovered so far on the ith node in the PDL.
# If you are more curious about the implementation, you can look at the
# model.py file in the codebase, which doesn't contain anything you can
# use to adaptively modify your code. (But lives in the same folder as the
# rest of the codebase just to make importing things easier)

# f is the current model
print(my_pdl.train_errors) # group errors on training set.
print(my_pdl.train_errors[0]) # this is the group error of each group on the initial PDL. The ith element of f.train_errors is the group error of each group on the ith version of the PDL.
print(my_pdl.test_errors) # group errors on validation set
print(my_pdl.predicates) # all of the group functions that have been appended so far
print(my_pdl.leaves) # all of the h functions appended so far
print(my_pdl.pred_names) # the names you passed in for each of the group functions, to more easily understand which are which.

#%%
# 5. Looking at the group error of the ith group over each round of updates:
# Say you found a group at round 5 and you want to know how its group error
# looked at previous or subsequent rounds. To do so, you can pull
# `f.train_errors` or `f.test_errors` and look at the ith element of each
# list as follows:

target_group = 0 # this sets the group whose error you want to look at at each round to the initial model. If I wanted to look at the 1st group introduced, would change to a 1, e.g.

group_errs = [e[target_group] for e in my_pdl.train_errors]
print(group_errs)







