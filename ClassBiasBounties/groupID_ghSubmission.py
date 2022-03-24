from dontlook import bountyHuntData
from dontlook import bountyHuntWrapper

acs_task = 'employment'
acs_states = ['NY']

[train_x, train_y, _, _] = bountyHuntData.get_data()

# Here, define all gs and hs that you developed in your notebook.


def g1(x):  # change this to be your first g
    return 1


h1 = bountyHuntWrapper.build_model(train_x, train_y, g1, dt_depth=10)  # change this to be your first h


# insert subsequent gs and hs you found here, including comments when it isn't obvious what g and h are.

# if you found more complex g's and h's that require interaction with the current PDL f, instead include a constructor
# for such a g or h, e.g.:
def g_(f):
    # do something clever with f here
    def g(x):
        return 1
    return g
