import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from cleveland import preproc
from ml.models import NetBin

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src/<package>
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "out")
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


# def tmp_kfolds_example():
#     X = np.array([[.1, .2], [.3, .4], [.1, .2], [.3, .4]])
#     y = np.array([1, 2, 3, 4])
#     kf = KFold(n_splits=2, shuffle=True, random_state=10)
#     kf.get_n_splits(X)
#     for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]


def split_folds(X, y, standardise_data=True, **kwargs):
    # NB this assumes rows are cases, columns are features
    kf = StratifiedKFold(**kwargs)
    folds_X_train, folds_X_val, folds_y_train, folds_y_val = [], [], [], []
    for train_index, val_index in kf.split(X, y):
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
    # TODO move this outside of this function, ie could supply only the train/val dataset for hp search
    data = preproc.from_file_with_dummies(os.path.join(DATA_DIR, FILE_PATH), DROP_FIRST)
    labels = data[LABEL]
    measurements = data.drop(LABEL, axis=1)
    np.random.seed(10)
    # Split into CV and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(measurements, labels, test_size=0.2)

    n_folds = 4
    folds_X_train, folds_X_val, folds_y_train, folds_y_val = split_folds(X_train_val, y_train_val,
                                                                         standardise_data=True, n_splits=n_folds,
                                                                         random_state=None, shuffle=True)

    folds_X_train, folds_X_val, folds_y_train, folds_y_val = reshape_folds(folds_X_train, folds_X_val,
                                                                           folds_y_train, folds_y_val)

    shared_cols = ['fold', 'dataset', 'arch.', 'init', 'alpha', 'n_iter', 'reg',
                   'roc_auc', 'sens.', 'spec.', 'acc.', 'tn', 'fp', 'fn', 'tp']

    # pd.DataFrame(columns=shared_cols + ['cost']).to_csv(os.path.join(OUT_DIR, "train_results.csv"), index=False)
    # pd.DataFrame(columns=shared_cols).to_csv(os.path.join(OUT_DIR, "val_results.csv"), index=False)
    n_prints = 5

    # Coarse search:  # TODO could make a grid search object to house these? sci kit learn may have something
    # architectures = [[1], [2], [4], [8], [16], [2, 2], [2, 4], [2, 8], [2, 16], [4, 2], [4, 4], [4, 8],
    #                  [4, 16], [8, 2], [8, 4], [8, 8], [8, 16], [16, 2], [16, 4], [16, 8], [16, 16, 1]]
    # reg_params = [0.0, 0.5, 1.0, 1.5]
    # w_inits = [0.01]
    # n_iters = [1e4]
    # alphas = [0.01, 0.05, 0.1, 0.5, 1]

    # Finer search:
    architectures = [[2], [2, 2]]
    reg_params = [0, 0.1, 0.2, 0.3]
    w_inits = [0.01]
    n_iters = [1e4]
    alphas = [0.8, 0.9, 1, 1.1, 1.2]
    train_file = os.path.join(OUT_DIR, "fine.train_results.csv")
    val_file = os.path.join(OUT_DIR, "fine.val_results.csv")
    for architecture in architectures:
        for reg_param in reg_params:
            for w_init in w_inits:
                for n_iter in n_iters:
                    for alpha in alphas:
                        train_results = pd.DataFrame(columns=shared_cols + ['cost'])
                        val_results = pd.DataFrame(columns=shared_cols)
                        settings = dict(zip(['architecture', 'reg_param', 'weight_scale', 'alpha', 'n_iter',
                                             'print_freq'],
                                        [architecture, reg_param, w_init, alpha, n_iter, n_prints/n_iter]))
                        print(settings)

                        for i_fold in range(n_folds):
                            np.random.seed(10)  # same initial weights for each fold, variance attributable to fold only
                            model, cost, train_perf, val_perf = experiment(folds_X_train[i_fold], folds_y_train[i_fold],
                                                                           folds_X_val[i_fold], folds_y_val[i_fold],
                                                                           **settings)

                            train_perf['cost'] = cost
                            train_perf['fold'], val_perf['fold'] = i_fold, i_fold
                            train_results = train_results.append(train_perf, ignore_index=True)
                            val_results = val_results.append(val_perf, ignore_index=True)
                        train_results.to_csv(train_file, mode='a', header=False, index=False)
                        val_results.to_csv(val_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    # tmp_kfolds_example()
    hp_search()
