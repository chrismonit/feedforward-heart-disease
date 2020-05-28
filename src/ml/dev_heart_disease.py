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
    result['accuracy'] = (tn + tp) / (tn + fp + fn + tp)
    result['sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    result['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else np.nan
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
    results = pd.DataFrame(columns=['model', 'dataset', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'tn', 'fp',
                                    'fn', 'tp'])

    sk_logreg_train_perform, sk_logreg_test_perform = sk_logreg(X, y, X_test, y_test)
    results = results.append(sk_logreg_train_perform, ignore_index=True)
    results = results.append(sk_logreg_test_perform, ignore_index=True)

    sk_logreg_rfecv_train_perform, sk_logreg_rfecv_test_perform = sk_logreg_rfecv(X, y, X_test, y_test)
    results = results.append(sk_logreg_rfecv_train_perform, ignore_index=True)
    results = results.append(sk_logreg_rfecv_test_perform, ignore_index=True)

    print("My models implementations:")
    n_iterations = int(1 * 10)
    print_frequency = 0.25
    alpha = 0.001
    model = LogReg(num_features=n_features)

    # print(f"LogReg initialised weights", f"{model.w}", "", sep="\n")
    final_cost = model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency=print_frequency)
    print(f"LogReg final cost", f"{final_cost}", "", sep="\n")

    y_pred_train_lr = pd.Series(model.predict(X), index=X.columns, name='predict')
    results = results.append(performance(y, y_pred_train_lr, model="my_lr", dataset="train"), ignore_index=True)

    y_pred_test = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    results = results.append(performance(y_test, y_pred_test, model="my_lr", dataset="test"), ignore_index=True)

    my_net_logreg = NetBin(X.shape[0], [], w_init_scale=0)  # tried scaling this to 0 to make the same as logreg
    # print(f"net logreg initialised weights", f"{my_net_logreg.weights}", "", sep="\n")
    # print(f"net logreg initialised biases", f"{my_net_logreg.biases}", "", sep="\n")
    my_net_logreg_cost = my_net_logreg.fit(X, np.expand_dims(y, 0), alpha, n_iterations,
                                           print_frequency=print_frequency)
    print(f"net_lr final cost", f"{my_net_logreg_cost}", "", sep="\n")
    y_pred_train_net_logreg = pd.Series(my_net_logreg.predict(X), index=X.columns, name='predict')
    results = results.append(performance(y, y_pred_train_net_logreg, model="net_logreg", dataset="train"),
                             ignore_index=True)
    net_y_pred_test = pd.Series(my_net_logreg.predict(X_test), index=X_test.columns, name='predict')
    results = results.append(performance(y_test, net_y_pred_test, model="net_logreg", dataset="test"),
                             ignore_index=True)

    print()
    print(results.sort_values("dataset"))


if __name__ == '__main__':
    print(f"Development of heart disease classification problem", f"", "", sep="\n")
    heart_disease()
