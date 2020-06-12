import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics

from cleveland import preproc
from ml.models import NetBin

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src/<package>
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEC = 3

LABEL = 'disease'
DROP_FIRST = False  # for assigning dummy variables using pandas method
FILE_PATH = "processed.cleveland.data.csv"
EPSILON = 1e-9


def standardise(df):
    means = df.mean()
    stds = df.std()
    standardised = (df - means).div(stds + EPSILON)  # avoid zero div problem
    return standardised, means, stds


def performance(y_true, y_pred, dataset=np.nan):
    result = {}
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    result['dataset'] = dataset
    result['acc.'] = (tn + tp) / (tn + fp + fn + tp)
    result['sens.'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    result['spec.'] = tn / (tn + fp) if (tn + fp) != 0 else np.nan
    result['roc_auc'] = metrics.roc_auc_score(y_true, y_pred)
    result.update(dict(zip(['tn', 'fp', 'fn', 'tp'], [tn, fp, fn, tp])))
    return result


# TODO could change order of args to match sklearn convention
def experiment(X, y, X_test, y_test, architecture=[], weight_scale=0.01, alpha=1, n_iter=1e2, reg_param=0,
               print_freq=0.1):
    """Instantiate, train and evaluate a neural network model"""
    architecture_str = "_".join([str(units) for units in architecture + [1]])
    model_info = {'arch.': architecture_str, 'init': weight_scale, 'alpha': alpha, 'n_iter': n_iter, 'reg': reg_param}
    model = NetBin(X.shape[0], architecture, w_init_scale=weight_scale)
    cost = model.fit(X.values, y.values, alpha, int(n_iter), reg_param=reg_param, print_frequency=print_freq)
    train_pred = pd.Series(model.predict(X.values), index=X.columns, name='predict').to_frame().T
    train_performance = performance(y.values.squeeze(), train_pred.values.squeeze(), dataset="train")
    test_pred = pd.Series(model.predict(X_test.values), index=X_test.columns, name='predict').to_frame().T
    test_performance = performance(y_test.values.squeeze(), test_pred.values.squeeze(), dataset="val/test")
    return model, cost, {**model_info, **train_performance}, {**model_info, **test_performance}


def tmp_kfolds_example():
    X = np.array([[.1, .2], [.3, .4], [.1, .2], [.3, .4]])
    y = np.array([1, 2, 3, 4])
    kf = KFold(n_splits=2, shuffle=True, random_state=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def split_folds(X, y, standardise_data=True, **kwargs):
    # NB this assumes rows are cases, columns are features
    kf = KFold(**kwargs)
    folds_X_train, folds_X_val, folds_y_train, folds_y_val = [], [], [], []
    for train_index, val_index in kf.split(X):
        folds_X_train.append(X.iloc[train_index])
        folds_X_val.append(X.iloc[val_index])
        folds_y_train.append(y.iloc[train_index])
        folds_y_val.append(y.iloc[val_index])

    if standardise_data:
        for i in range(kf.get_n_splits()):
            X_train_scaled, X_train_means, X_train_stds = standardise(folds_X_train[i])
            X_val_scaled = (folds_X_val[i] - X_train_means).div(X_train_stds + EPSILON)  # avoid zero div problem
            folds_X_train[i] = X_train_scaled
            folds_X_val[i] = X_val_scaled
    return folds_X_train, folds_X_val, folds_y_train, folds_y_val


def reshape_folds(folds_X_train, folds_X_val, folds_y_train, folds_y_val):
    assert len(folds_X_train) == len(folds_X_val) == len(folds_y_train) == len(folds_y_val), "Unequal fold numbers"
    for i in range(len(folds_X_train)):
        folds_X_train[i] = folds_X_train[i].T
        folds_X_val[i] = folds_X_val[i].T
        folds_y_train[i] = folds_y_train[i].to_frame().T
        folds_y_val[i] = folds_y_val[i].to_frame().T
    return folds_X_train, folds_X_val, folds_y_train, folds_y_val


def hp_search():
    data = preproc.from_file_with_dummies(os.path.join(DATA_DIR, FILE_PATH), DROP_FIRST)
    labels = data[LABEL]
    measurements = data.drop(LABEL, axis=1)
    np.random.seed(10)
    # Split into CV and test sets
    # TODO move this outside of this function?
    X_train_val, X_test, y_train_val, y_test = train_test_split(measurements, labels, test_size=0.2)

    n_folds = 4
    folds_X_train, folds_X_val, folds_y_train, folds_y_val = split_folds(X_train_val, y_train_val,
                                                                         standardise_data=True, n_splits=n_folds,
                                                                         random_state=None, shuffle=True)

    # for experiment in experiments, the set of hyperparameters to try
        # for fold in folds:
            # train model using those hyperparameters, save performance measures on val set
        # take average performance measures across folds
    folds_X_train, folds_X_val, folds_y_train, folds_y_val = reshape_folds(folds_X_train, folds_X_val,
                                                                           folds_y_train, folds_y_val)
    for i in range(n_folds):
        print(folds_y_train[i].values.shape)
    # model, cost, train_perf, val_perf = experiment(folds_X_train[0], folds_y_train[0],
    #                                                folds_X_val[0], folds_y_val[0], architecture=[2])

    shared_cols = ['exp_id', 'arch.', 'init', 'alpha', 'n_iter', 'reg', 'dataset', 'roc_auc', 'sens.', 'spec.', 'acc.',
                   'tn', 'fp', 'fn', 'tp']

    train_results = pd.DataFrame(columns=shared_cols+['cost'])
    val_results = pd.DataFrame(columns=shared_cols)

    n_prints = 5
    exp_id = 0
    for architecture in [[], [2], [4], [8]]:
        for reg_param in [0, 1, 2]:
            for w_init in [0.01]:  #  [0.001, 0.01, 0.1, 1, 10]:
                for n_iter in [1e4]:
                    for alpha in [0.01, 0.1, 1]:

                        settings = dict(zip(['architecture', 'reg_param', 'weight_scale', 'alpha', 'n_iter',
                                             'print_freq'],
                                        [architecture, reg_param, w_init, alpha, n_iter, n_prints/n_iter]))
                        # TODO for each fold:
                        model, cost, train_perf, val_perf = experiment(folds_X_train[0], folds_y_train[0],
                                                                       folds_X_val[0], folds_y_val[0], **settings)
                        train_perf['cost'] = cost
                        train_perf['exp_id'], val_perf['exp_id'] = exp_id, exp_id
                        train_results = train_results.append(train_perf, ignore_index=True)
                        val_results = val_results.append(val_perf, ignore_index=True)
                        exp_id += 1

    print(train_results)
    print()
    print()
    print(val_results)

    exit()
    n_iter = int(1 * 1e4)
    alpha = 0.15
    n_print_statements = 5
    print_freq = n_print_statements / n_iter

    # # find an example where it fails to learn, so we can debug it specifically.
    # # ie need to find what the initial parameters are
    # for architecture in [[2, 2]]:
    #     for reg_param in [2]:
    #         for w_init in [0.01]:  #  [0.001, 0.01, 0.1, 1, 10]:
    #             for alpha in [0.1]:
    #                 np.random.seed(10)
    #                 for i in range(3):
    #                     model, cost, train_result, test_result = experiment(X_train_val, y_train_val, X_test, y_test,
    #                                                                         architecture=architecture,
    #                                                                         weight_scale=w_init, alpha=alpha,
    #                                                                         n_iter=n_iter, reg_param=reg_param,
    #                                                                         print_freq=print_freq)
    #                     results = results.append(train_result, ignore_index=True)
    #                     results = results.append(test_result, ignore_index=True)
    #                     print(f"i={i}; reg_param={reg_param}; w_init={w_init}; alpha={alpha}; cost={cost}")
    #                     print()
    #
    # print(results.sort_values(['dataset', 'init']))
    # exit()

    print("------------")
    np.random.seed(10)  # this one learns. it has a higher weight scale
    model1, cost1, train_result1, test_result1 = experiment(X_train_val, y_train_val, X_test, y_test, architecture=[2, 2],
                                                 weight_scale=0.1, alpha=0.1,
                                                 n_iter=int(1e4), reg_param=0,
                                                 print_freq=print_freq)
    print(cost1)
    print(model1.weights)
    print()
    print()
    print()
    print()

    np.random.seed(10)  # this one does not learn
    model2, cost2, train_result2, test_result2 = experiment(X, y, X_test, y_test, architecture=[2, 2],
                                                 weight_scale=0.01, alpha=0.1,
                                                 n_iter=int(1e4), reg_param=0,
                                                 print_freq=print_freq)
    print(cost2)
    print(model2.weights)

    # TODO class balancing? implement other gradient descent algorithms?


if __name__ == '__main__':
    # tmp_kfolds_example()
    hp_search()
