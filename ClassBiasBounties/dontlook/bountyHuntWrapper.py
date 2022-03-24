from sklearn.tree import DecisionTreeClassifier
import model
import verifier
import cscUpdater
from sklearn import metrics


def build_model(x, y, group_function, dt_depth):
    print("building h")
    # learn the indices first, since this is an inefficient operation
    indices = x.apply(group_function, axis=1) == 1

    # then pull the particular rows from the dataframe
    training_xs = x[indices]
    training_ys = y[indices]

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=0)  # setting random state for replicability
    dt.fit(training_xs, training_ys)
    print("finished building h")
    return dt.predict


def build_initial_pdl(initial_model, train_x, train_y, validation_x, validation_y):
    f = model.PointerDecisionList(initial_model.predict)
    # manually stick in the train and test errors because I'm dumb and have codependencies in the files
    f.test_errors.append(cscUpdater.measure_group_errors(f, validation_x, validation_y))
    f.train_errors.append(cscUpdater.measure_group_errors(f, train_x, train_y))
    return f


def verify_size(x, group):
    # helper function that checks that the discovered group isn't too small to run on
    indices = x.apply(group, axis=1) == 1
    xs = x[indices]
    if len(xs) == 0:
        return False
    else:
        return True


def run_checks(f, validation_x, validation_y, g, h, train_x, train_y):
    size_check = verify_size(validation_x, g)
    if not size_check:  # Remove before deployment:
        print("Group has 0 weight in test set")
        indices = train_x.apply(g, axis=1) == 1
        xs = train_x[indices]
        ys = train_y[indices]

        # get predicted ys from current model and proposed h
        curr_model_preds = xs.apply(f.predict, axis=1)
        h_preds = h(xs)

        # measure the error of current model and proposed h
        curr_model_error = metrics.zero_one_loss(ys, curr_model_preds)
        h_error = metrics.zero_one_loss(ys, h_preds)

        print("Training Error of current model on proposed group: %s" % curr_model_error)
        print("Training Error of h trained on proposed group: %s" % h_error)
        print("Group size in training set: %s" % len(xs))
        print("Group weight in training set: %s" % (len(xs)/len(train_x)))
    if size_check:
        improvement_check = verifier.is_proposed_group_good(f, validation_x, validation_y, h, g,
                                                            train_x, train_y)
        if improvement_check:
            print("Passed checks.")
            return True
        else:
            print("Failed improvement check.")
            return False
    else:
        print("Failed group size check.")
        return False


def measure_group_error(model, group, X, y):
    """
    Function to measure group errors of a specific group

    NOTE THIS WILL BREAK IF YOU PASS IN AN EMPTY GROUP
    """

    indices = X.apply(group, axis=1) == 1
    xs = X[indices]
    ys = y[indices]
    pred_ys = xs.apply(model.predict, axis=1).to_numpy()
    group_errors = metrics.zero_one_loss(ys, pred_ys)

    return group_errors


def run_updates(f, g, h, train_x, train_y, validation_x, validation_y, group_name="g"):
    cscUpdater.iterative_update(f, h, g, train_x, train_y, validation_x, validation_y, group_name)
