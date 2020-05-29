import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

import preproc_cleveland
from models import LogReg
from models import NetBin

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # assuming this file in <proj_root>/src/<package>
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEC = 3


def standardise(df):
    means = df.mean()
    stds = df.std()
    standardised = (df - means).div(stds)
    return standardised, means, stds


def performance(y_true, y_pred, model=np.nan, dataset=np.nan):
    result = {}
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    result['model'] = model
    result['dataset'] = dataset
    result['acc.'] = (tn + tp) / (tn + fp + fn + tp)
    result['sens.'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    result['spec.'] = tn / (tn + fp) if (tn + fp) != 0 else np.nan
    result['roc_auc'] = metrics.roc_auc_score(y_true, y_pred)
    result.update(dict(zip(['tn', 'fp', 'fn', 'tp'], [tn, fp, fn, tp])))
    return result


def sk_logreg(X, y, X_test, y_test):
    """ Using sci kit learn implenentation of logistic regression """
    lr2 = LogisticRegression(penalty='none')  # NB not using regularisation yet
    lr2.fit(X.T, y)
    lr_y_pred_train = pd.Series(lr2.predict(X.T), index=X.T.index, name='predict')
    train_performance = performance(y, lr_y_pred_train, model="skl_lr", dataset="train")

    lr_y_pred_test = pd.Series(lr2.predict(X_test.T), index=X_test.T.index, name='predict')
    test_performance = performance(y_test, lr_y_pred_test, model="skl_lr", dataset="test")
    return train_performance, test_performance


def sk_logreg_rfecv(X, y, X_test, y_test):
    """Feature selection using RFECV"""
    selector_lr = LogisticRegression(penalty='none')
    selector = RFECV(selector_lr, step=1, verbose=0)
    selector.fit(X.T, y)
    print(f"Features found by RFECV with sklearn implementation ({selector.n_features_}):",
          X.T.loc[:, selector.support_].columns.to_numpy(), "", sep="\n")

    lr2 = LogisticRegression(penalty='none')
    lr2.fit(X.T.loc[:, selector.support_], y)
    lr_y_pred_train = pd.Series(lr2.predict(X.T.loc[:, selector.support_]), index=X.T.index, name='predict')
    train_performance = performance(y, lr_y_pred_train, model="skl_lr_rfecv", dataset="train")

    lr_y_pred_test = pd.Series(lr2.predict(X_test.T.loc[:, selector.support_]), index=X_test.T.index, name='predict')
    test_performance = performance(y_test, lr_y_pred_test, model="skl_lr_rfecv", dataset="test")
    return train_performance, test_performance


def logreg(X, y, X_test, y_test, n_features, alpha, n_iterations, print_frequency):
    """My implementation of logistic regression:"""
    model = LogReg(num_features=n_features)
    final_cost = model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency=print_frequency)
    y_pred_train_lr = pd.Series(model.predict(X), index=X.columns, name='predict')
    train_performance = performance(y, y_pred_train_lr, model="logreg", dataset="train")
    y_pred_test = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    test_performance = performance(y_test, y_pred_test, model="logreg", dataset="test")
    return train_performance, test_performance


def heart_disease():
    np.random.seed(10)
    LABEL = 'disease'
    DROP_FIRST = False
    signal_catagorical = ['sex', 'cp', 'exang', 'slope', 'thal']
    signal_quantitative = ['age', 'thalach', 'oldpeak', 'ca']
    signal_features = signal_catagorical + signal_quantitative

    data = preproc_cleveland.from_file(os.path.join(DATA_DIR, "processed.cleveland.data.csv"), DROP_FIRST)
    features_to_use = [col for col in data.columns for feature in signal_features if
     col == feature or col.startswith(feature + preproc_cleveland.DUMMY_SEPARATOR)]
    # data = data[[LABEL] + features_to_use]
    labels = data[LABEL]
    measurements = data.drop(LABEL, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(measurements, labels, test_size=0.33)

    X_train_scaled, X_train_means, X_train_stds = standardise(X_train)
    X_test_scaled = (X_test - X_train_means).div(X_train_stds)

    X = X_train_scaled.T  # rows are features, columns are training cases
    X_test = X_test_scaled.T
    y = y_train

    n_features, m = X.shape
    print(f"m={m}, n_features={n_features}", "", sep="\n")
    results = pd.DataFrame(columns=['model', 'dataset', 'roc_auc', 'sens.', 'spec.', 'acc.', 'tn', 'fp',
                                    'fn', 'tp'])

    # sk_logreg_train_perform, sk_logreg_test_perform = sk_logreg(X, y, X_test, y_test)
    # results = results.append(sk_logreg_train_perform, ignore_index=True)
    # results = results.append(sk_logreg_test_perform, ignore_index=True)
    #
    # sk_logreg_rfecv_train_perform, sk_logreg_rfecv_test_perform = sk_logreg_rfecv(X, y, X_test, y_test)
    # results = results.append(sk_logreg_rfecv_train_perform, ignore_index=True)
    # results = results.append(sk_logreg_rfecv_test_perform, ignore_index=True)

    n_iter = int(1 * 1e4)
    alpha = 0.1
    n_print_statements = 10
    print_freq = n_print_statements / n_iter

    # logreg_train_perform, logreg_test_perform = logreg(X, y, X_test, y_test, n_features, alpha, n_iter, print_freq)
    # results = results.append(logreg_train_perform, ignore_index=True)
    # results = results.append(logreg_test_perform, ignore_index=True)

    architecture, reg_param = [], 0
    name = "1:" + str(reg_param)
    print(f"Running model {name}")
    net_logreg = NetBin(X.shape[0], architecture, w_init_scale=0)  # tried scaling this to 0 to make the same as logreg
    net_logreg_cost = net_logreg.fit(X, np.expand_dims(y, 0), alpha, n_iter, print_frequency=print_freq)
    y_pred_train_net_logreg = pd.Series(net_logreg.predict(X), index=X.columns, name='predict')
    results = results.append(performance(y, y_pred_train_net_logreg, model=name, dataset="train"), ignore_index=True)
    net_y_pred_test = pd.Series(net_logreg.predict(X_test), index=X_test.columns, name='predict')
    results = results.append(performance(y_test, net_y_pred_test, model=name, dataset="test"), ignore_index=True)

    architecture, reg_param = [2], 0
    name = "_".join([str(units) for units in architecture+[1]]) + ":" + str(reg_param)
    print(f"Running model {name}")
    model = NetBin(X.shape[0], architecture, w_init_scale=0.01)
    cost = model.fit(X, np.expand_dims(y, 0), alpha, n_iter, print_frequency=print_freq)
    train_pred = pd.Series(model.predict(X), index=X.columns, name='predict')
    results = results.append(performance(y, train_pred, model=name, dataset="train"), ignore_index=True)
    test_pred = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    results = results.append(performance(y_test, test_pred, model=name, dataset="test"), ignore_index=True)

    architecture, reg_param = [2], 1
    name = "_".join([str(units) for units in architecture + [1]]) + ":" + str(reg_param)
    print(f"Running model {name}")
    model = NetBin(X.shape[0], architecture, w_init_scale=0.01)
    cost = model.fit(X, np.expand_dims(y, 0), alpha, n_iter, print_frequency=print_freq)
    train_pred = pd.Series(model.predict(X), index=X.columns, name='predict')
    results = results.append(performance(y, train_pred, model=name, dataset="train"), ignore_index=True)
    test_pred = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    results = results.append(performance(y_test, test_pred, model=name, dataset="test"), ignore_index=True)

    print()
    print(results.sort_values(["dataset", "model"]).round(DEC))

    # TODO class balancing? implement other gradient descent algorithms?


if __name__ == '__main__':
    heart_disease()
