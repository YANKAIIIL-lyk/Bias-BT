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


acs_task = 'employment'
acs_states = ['NY']

TOP_N = 4
epsilon = 0.001  # 0.001
[train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()
train_x, validation_x = preprocess(train_x, validation_x)

initial_model = DecisionTreeClassifier(max_depth=1, random_state=0)
initial_model.fit(train_x, train_y)
groups_fcns = dict()  # name -> grp_fcn
generate_one_dim_groups(train_x, groups_fcns)

# f is the initial PDL
f = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)

# Here, define all gs and hs that you developed in your notebook.
def g1(x):
    # here is how to make a function g that returns 1 for all African American individuals
    if x['SCHL'] <= 15:
        return 1
    else:
        return 0
# Update f with g1
simple_updater(f, g1, group_name="g1")


def g2(x):
    # here is how to make a function g that returns 1 for all African American individuals
    if x['AGEP'] < 16 or x['AGEP'] >= 60:
        return 1
    else:
        return 0
# Update f with g2
simple_updater(f, g2, group_name="g2")

# Visit each feature, try to update the PDL
def visitEachList(df):
    columns = list(df.columns)[2:]
    count = 1
    for column in columns:
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
            evaluation(f, train_x, train_y)
visitEachList(train_x)

# Implemented algorithm 6 in the given paper to find g
f_predict_xN = lambda xN: xN.apply(f.predict, axis=1)
retries_nr = 0
updates_count = 0
itr_count = 0

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

# Evalutate the final PDL. The final overall testing error is around 0.1716
evaluation(f, train_x, train_y)