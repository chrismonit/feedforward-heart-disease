import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import src.preproc as preproc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")


def sigmoid(z):
    return 1./(1 + np.exp(-z))


def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


# def cost(y_pred, y_true):
#     return -1./len(y_true) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    np.random.seed(10)
    data = preproc.preproc_file(os.path.join(DATA_DIR, "short_names_capsid_master.fasta"), 'fasta')
    data = data[data["label"].isin(["B", "C"])]
    data["label"] = data["label"].map({'B': 1, 'C': 0})
    b_sample = data[data["label"] == 1].iloc[:10, :]
    c_sample = data[data["label"] == 0].iloc[:10, :]
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
    n_iterations = 500
    print(f"m={m}, n_features={n_features}", f"", "", sep="\n")

    # w_init_scale = 0.01
    # wT = np.random.randn(1, n_features) * w_init_scale
    w = np.zeros((n_features, 1))  # can initilise with 0 for LR
    b = 0.0
    print("Input dimensions:"
          f"X={X.shape}",
          f"y={y.shape}",
          f"w={w.shape}",
          f"", "", sep="\n")
    alpha = 0.01
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
    print(y_pred)
    print(y_test)



if __name__ == '__main__':
    main()
