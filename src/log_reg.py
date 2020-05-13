import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import src.preproc as preproc
import src.preproc_cleveland as preproc_cleveland
import matplotlib.pyplot as plt

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEC = 3


def sigmoid(z):
    return 1./(1 + np.exp(-z))


def standardise(df):
    means = df.mean()
    stds = df.std()
    standardised = (df - means).div(stds)
    return standardised, means, stds


# TODO regularisation? Class balancing?
# TODO could check if this satisfies sci kit learn's api as a classifier
class Log_reg():
    def __init__(self, num_features, w_init=None, b_init=None, predict_thresh=0.5):
        self.w = w_init if w_init else np.zeros((num_features, 1))
        self.b = b_init if b_init else 0
        self.predict_thresh = predict_thresh

    def _forward(self, X, y):
        Z = (np.dot(self.w.T, X) + self.b)
        A = self.sigmoid(Z)
        cost = self.cross_entropy_cost(A, y)
        return A, cost

    @staticmethod
    def _backward(A, X, y):
        dz = A - y.values
        dw = np.dot(X, dz.T) / len(y)  # dz = A - Y
        db = np.sum(dz) / len(y)
        return dw, db

    def _update(self, dw, db, learn_rate):
        self.w = self.w - learn_rate * dw
        self.b = self.b - learn_rate * db

    # TODO could return cost values during fitting.
    def fit(self, X, y, learn_rate, num_iterations, print_frequency=0.1):
        for i in range(num_iterations):
            A, cost = self._forward(X, y)
            dw, db = self._backward(A, X, y)
            self._update(dw, db, learn_rate)
            if i % int(1. / print_frequency) == 0:
                print(f"LogReg: iteration={i}, cost={cost}")
        return cost

    @staticmethod
    def cross_entropy_cost(y_pred, y_true):
        cost = np.sum(y_true.values * np.log(y_pred) + (1. - y_true.values) * np.log(1. - y_pred)) / -len(y_true)
        return cost.squeeze()

    @staticmethod
    def sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def predict(self, X):
        Z = np.dot(self.w.T, X) + self.b
        A = sigmoid(Z)
        y_pred = [int(a > self.predict_thresh) for a in A.squeeze()]
        return y_pred


def seq_classifier():
    np.random.seed(10)
    data = preproc.preproc_file(os.path.join(DATA_DIR, "short_names_capsid_master.fasta"), 'fasta')
    classes = {'A1': 1, 'A2': 0}
    data = data[data["label"].isin(classes.keys())]
    data["label"] = data["label"].map(classes)
    print(f"All data value counts", data['label'].value_counts(), "", sep="\n")
    sample_1 = data[data["label"] == 1].iloc[:, :]
    sample_2 = data[data["label"] == 0].iloc[:, :]
    sample = pd.concat([sample_1, sample_2], axis=0)

    labels = sample['label']
    data = sample.drop(columns=['label'])
    # expects rows to be training examples. NB returned X are dfs and returned y are series
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

    X = X_train.T  # rows are features, columns are training cases
    X_test = X_test.T
    y = y_train

    n_features, m = X.shape
    n_iterations = 500
    alpha = 0.01
    print(f"m={m}, n_features={n_features}", f"", "", sep="\n")

    model = Log_reg(num_features=n_features)
    model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency=0.1)

    y_pred_train = pd.Series(model.predict(X), index=X.columns, name='predict')
    train_results = pd.merge(y, y_pred_train, left_index=True, right_index=True)
    print(pd.crosstab(train_results['label'], train_results['predict']))
    print(f"Model accuracy on train data: {metrics.accuracy_score(y, y_pred_train, normalize=True)}")

    y_pred_test = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    test_results = pd.merge(y_test, y_pred_test, left_index=True, right_index=True)
    print(pd.crosstab(test_results['label'], test_results['predict']))
    print(f"Model accuracy on test data: {metrics.accuracy_score(y_test, y_pred_test, normalize=True)}")
    print()


def performance(y_true, y_pred):
    result = {}
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    result['accuracy'] = (tn + tp) / (tn + fp + fn + tp)
    result['sensitivity'] = tp / (tp + fn)
    result['specificity '] = tn / (tn + fp)
    result['roc_auc'] = metrics.roc_auc_score(y_true, y_pred)
    result.update(dict(zip(['tn', 'fp', 'fn', 'tp'], [tn, fp, fn, tp])))
    return result


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
    n_iterations = int(1 * 1e1)
    alpha = 0.001
    print(f"m={m}, n_features={n_features}", f"", "", sep="\n")

    model = Log_reg(num_features=n_features)
    final_cost = model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency=0.0001)
    print(f"Final cost: {final_cost}", "", sep="\n")

    y_pred_train_lr = pd.Series(model.predict(X), index=X.columns, name='predict')
    train_performance = performance(y, y_pred_train_lr)
    print("Performance on training data:")
    for k in train_performance.keys():
        print(k, np.round(train_performance[k], DEC))
    print()
    # print(f"TP", train_confusion.loc[(train_confusion['predict'] == 1) & train_confusion[LABEL] == 1], "", sep="\n")

    print("Performance on test data:")
    y_pred_test = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    test_performance = performance(y_test, y_pred_test)
    for k in test_performance.keys():
        print(k, np.round(test_performance[k], DEC))
    print()

    print("Testing sklearn implementation of logistinc regression")

    lr2 = LogisticRegression(penalty='none')
    lr2.fit(X.T, y)
    lr_y_pred_train = pd.Series(lr2.predict(X.T), index=X.T.index, name='predict')
    lr_train_performance = performance(y, lr_y_pred_train)
    for k in lr_train_performance.keys():
        print(k, np.round(lr_train_performance[k], DEC))
    print()

    lr_y_pred_test = pd.Series(lr2.predict(X_test.T), index=X_test.T.index, name='predict')
    lr_test_performance = performance(y_test, lr_y_pred_test)
    for k in lr_test_performance.keys():
        print(k, np.round(lr_test_performance[k], DEC))
    print()

    print(f"Feature selection using RFECV", f"", "", sep="\n")
    selector_lr = LogisticRegression(penalty='none')
    selector = RFECV(selector_lr, step=1, verbose=0)
    selector.fit(X.T, y)
    print(f"Number of features chosen by RFECV={selector.n_features_}")
    print(f"Features found to ", X.T.loc[:, selector.support_].columns)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()

    lr2 = LogisticRegression(penalty='none')
    lr2.fit(X.T.loc[:, selector.support_], y)
    lr_y_pred_train = pd.Series(lr2.predict(X.T.loc[:, selector.support_]), index=X.T.index, name='predict')
    lr_train_performance = performance(y, lr_y_pred_train)
    for k in lr_train_performance.keys():
        print(k, np.round(lr_train_performance[k], DEC))
    print()

    lr_y_pred_test = pd.Series(lr2.predict(X_test.T.loc[:, selector.support_]), index=X_test.T.index, name='predict')
    lr_test_performance = performance(y_test, lr_y_pred_test)
    for k in lr_test_performance.keys():
        print(k, np.round(lr_test_performance[k], DEC))
    print()


if __name__ == '__main__':
    heart_disease()
