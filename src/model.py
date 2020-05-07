import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import src.preproc as preproc
import src.preproc_cleveland as preproc_cleveland

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
    train_results = pd.merge(y_true, y_pred, left_index=True, right_index=True)
    result['confusion'] = pd.crosstab(train_results.iloc[:, 0], train_results.iloc[:, 1])
    result['accuracy'] = metrics.accuracy_score(y_true, y_pred, normalize=True)
    result['sensitivity'] = result['confusion'].loc[1, 1] / result['confusion'].loc[1, :].sum()
    result['specificity'] = result['confusion'].loc[0, 0] / result['confusion'].loc[0, :].sum()
    return result


def heart_disease():
    # TODO should we not drop the nth category in the dummy fields?
    np.random.seed(10)

    signal_catagorical = ['sex', 'cp', 'exang', 'slope', 'thal']
    signal_quantitative = ['age', 'thalach', 'oldpeak', 'ca']
    signal_features = signal_catagorical + signal_quantitative

    data = preproc_cleveland.from_file(os.path.join(DATA_DIR, "processed.cleveland.data.csv"))
    features_to_use = [col for col in data.columns for feature in signal_features if
     col == feature or col.startswith(feature + preproc_cleveland.DUMMY_SEPARATOR)]
    data = data[['disease'] + features_to_use]
    LABEL = 'disease'
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
    model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency=0.0001)

    y_pred_train = pd.Series(model.predict(X), index=X.columns, name='predict')
    train_performance = performance(y, y_pred_train)
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


if __name__ == '__main__':
    heart_disease()
