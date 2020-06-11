import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from cleveland.dev_heart_disease import performance
from ml.models import LogReg

# TODO untested in this context


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