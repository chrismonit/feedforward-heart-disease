import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics

from subtype import preproc
from models import LogReg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # assuming this file in <proj_root>/src
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEC = 3


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

    model = LogReg(num_features=n_features)
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


if __name__ == '__main__':
    print(f"Development of HIV subtype classification problem", f"", "", sep="\n")
    seq_classifier()
