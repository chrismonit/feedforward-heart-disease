import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import src.preproc as preproc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")


def sigmoid(z):
    return 1./(1 + np.exp(-z))


def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


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

    # TODO could return cost values during fitting. Best as a generator??
    def fit(self, X, y, learn_rate, num_iterations, print_frequency_fraction=0.1):
        for i in range(num_iterations):
            A, cost = self._forward(X, y)
            dw, db = self._backward(A, X, y)
            self._update(dw, db, learn_rate)
            if i % int(1./print_frequency_fraction) == 0:
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
        # print("accuracy:", metrics.accuracy_score(y_test, y_pred))


def main():
    np.random.seed(10)
    data = preproc.preproc_file(os.path.join(DATA_DIR, "short_names_capsid_master.fasta"), 'fasta')
    data = data[data["label"].isin(["B", "C"])]
    data["label"] = data["label"].map({'B': 1, 'C': 0})
    b_sample = data[data["label"] == 1].iloc[:500, :]
    c_sample = data[data["label"] == 0].iloc[:500, :]
    sample = pd.concat([b_sample, c_sample], axis=0)

    labels = sample['label']
    # print(labels)
    # a = np.reshape(np.array([1]*len(labels)), (1, 20))
    # print(a)
    # print(a + labels.to_frame().T)
    data = sample.drop(columns=['label'])
    print(f"labels", labels, f"", "", sep="\n")

    # expects rows to be training examples. NB returned X are dfs and returned y are series
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

    X = X_train.T  # rows are features, columns are training cases
    X_test = X_test.T
    y = y_train

    n_features, m = X.shape
    n_iterations = 100
    alpha = 0.01
    print(f"m={m}, n_features={n_features}", f"", "", sep="\n")

    model = Log_reg(num_features=n_features)
    # A, cost = model._forward(X, y)
    # print(f"From model, cost={cost}, A={A}")
    model.fit(X, y, alpha, num_iterations=n_iterations, print_frequency_fraction=0.1)
    model_y_pred = model.predict(X_test)
    print(f"Model accuracy: {metrics.accuracy_score(y_test, model_y_pred)}")
    print()
    # exit()

    # w_init_scale = 0.01
    # wT = np.random.randn(1, n_features) * w_init_scale
    w = np.zeros((n_features, 1))  # can initilise with 0 for LR
    b = 0.0
    # print("Input dimensions:"
    #       f"X={X.shape}",
    #       f"y={y.shape}",
    #       f"w={w.shape}",
    #       f"", "", sep="\n")
    for i_iter in range(n_iterations):
        Z = (np.dot(w.T, X) + b)
        A = sigmoid(Z)
        cost = np.sum(y_train.values * np.log(A) + (1. - y_train.values) * np.log(1. - A)) / -m
        cost = cost.squeeze()

        dz = A - y.values
        dw = np.dot(X, dz.T) / m  # dz = A - Y
        db = np.sum(dz) / m

        w = w - alpha * dw
        b = b - alpha * db
        # print(f"Z shape={Z.shape}, A.shape={A.shape}, J shape={J.shape}",
        #       f"dZ.shape={dZ.shape}", f"y_train.shape={y_train.shape}"
        #       f"", "", sep="\n")

        # print(f"i={i}", f"z={Z}", f"a={A}", f"y={y_train.values}", f"J={J}", "", sep="\n")
        if i_iter % 10 == 0:
            print(f"iteration={i_iter}, cost={cost}")

    THRESH = 0.5
    Z_test = np.dot(w.T, X_test) + b
    A_test = sigmoid(Z_test)
    y_pred = [int(a > THRESH) for a in A_test.squeeze()]
    # print(y_pred)
    # print(y_test)
    print("accuracy:", metrics.accuracy_score(y_test, y_pred))



if __name__ == '__main__':
    main()
