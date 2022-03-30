#!/usr/bin/env python
# coding: utf-8

# # Bounty Hunting Project Notebook

# < Insert group member names here and change the name of the file so it has your group ID in it.>

# ## Set Up
# 
# First, we load in some helper files.

# In[1]:


import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
sys.path.append('dontlook')
from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper
from utils import preprocess
import pandas as pd
import numpy as np


# Next, we load in the data. You should use `train_x` and `train_y` to train your models. The second set of data (`validation_x` and `validation_y`) is for testing your models, to ensure that you aren't overfitting. It is also what will be passed to the updater in order to determine if a proposed update should be accepted and if repairs are needed. Since you have access to this data, you could overfit to it and get a bunch of updates accepted. However, a) we'll be able to tell you did this and b) your updates will fail on the holdout set that only we have access to, so doing this is not in your best interest.

# In[2]:


[train_x, train_y, validation_x, validation_y] = bountyHuntData.get_data()


# # Preprocessing
# Some features contain nominal data and the underlying distribution is highly skewed. Preprocess those features before move forward.

# In[3]:


train_x, validation_x = preprocess(train_x, validation_x)


# The model that you'll be building off of is a decision stump, i.e. a very stupid decision list with only one node. **Warning: do not rerun the next code block unless you want to completely restart building your PDL, as it will re-initialize it to just the decision stump!**

# In[5]:


import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


# In[6]:


from sklearn import metrics
def evaluation(current_pdl):
    print("per-group training errors of lastest version PDL:")
    print(current_pdl.train_errors[
               -1])  # this is the group error of each group on the initial PDL.
    # The ith element of f.train_errors is the group error of each group on
    # the ith version of the PDL.
    print("per-group testing errors of lastest version PDL:")
    print(current_pdl.test_errors[-1])  # group errors on validation set
    # vprint(f.predicates)  # all of the group functions that have been appended so far
    # vprint(f.leaves)  # all of the h functions appended so far
    print(
        current_pdl.pred_names)  # the names you passed in for each of the group functions, to more easily understand which are which.
    # %%
    # 5. Looking at the group error of the ith group over each round of updates:
    # Say you found a group at round 5 and you want to know how its group error
    # looked at previous or subsequent rounds. To do so, you can pull
    # `f.train_errors` or `f.test_errors` and look at the ith element of each
    # list as follows:
    target_group = -1  # this sets the group whose error you want to look at at each round to the initial model. If I wanted to look at the 1st group introduced, would change to a 1, e.g.
    print("latest group's training errors on all preivous PDLs:")
    print([e[-1] for e in current_pdl.train_errors])
    print("latest group's testing errors on all preivous PDLs:")
    print([e[-1] for e in current_pdl.test_errors])
    g_all = lambda x: 1  # here we define a group that just is all the data, replace as you see fit.
    bountyHuntWrapper.measure_group_error(f, g_all, train_x, train_y)

    pred_ys = validation_x.apply(f.predict, axis=1).to_numpy()
    errors = metrics.zero_one_loss(validation_y, pred_ys)
    print("latest PDL's overall testing  errors %.10f" % errors)


# In[7]:


initial_model = DecisionTreeClassifier(max_depth = 1, random_state=0)
initial_model.fit(train_x, train_y)
f = bountyHuntWrapper.build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y)


# In[10]:

def simple_updater(g,group_name = "g"):
    # if you want to change how h is trained, you can edit the below line.
    h = bountyHuntWrapper.build_model(train_x, train_y, g, dt_depth=10)
    # do not change anything beyond this point.
    if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)
        return True
    return False
        
def rf_updater(g, group_name = "g", dt_depth = 5):
    print("building h")
    # Get indices first
    indices = train_x.apply(g, axis=1) == 1
    # Find training set
    training_xs = train_x[indices]
    training_ys = train_y[indices]

    clf = RandomForestClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    clf.fit(training_xs, training_ys)
    print("finished building h")
    h = clf.predict
    # do not change anything beyond this point.
    if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)

def updater(g, h, group_name="g"):
    # do not alter this code
    if bountyHuntWrapper.run_checks(f, validation_x, validation_y, g, h, train_x=train_x, train_y=train_y):
        print("Running Update")
        bountyHuntWrapper.run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name=group_name)


def visitEachList(df):
    columns = list(df.columns)[2:]
    count = 1
    for column in columns:
        print (column)
        def aggr(column):
            def g(x):
                if(x[column] == 1):
                    return 1
                else:
                    return 0
            return g
        g = aggr(column)
        ret = simple_updater(g, "group" + str(count))
        if ret:
            print ("Updated")
            # evaluation(f)

visitEachList(train_x)
evaluation(f)