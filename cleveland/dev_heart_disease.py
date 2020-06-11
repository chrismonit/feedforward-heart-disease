import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics

from cleveland import preproc
from ml.models import NetBin

pd.options.display.width = 0  # adjust according to terminal width

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # assuming this file in <proj_root>/src/<package>
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEC = 3


def standardise(df):
    means = df.mean()
    stds = df.std()
    standardised = (df - means).div(stds)
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


def experiment(X, y, X_test, y_test, architecture=[], weight_scale=0.01, alpha=1, n_iter=1e2, reg_param=0,
               print_freq=0.1):
    """Instantiate, train and evaluate a neural network model"""
    architecture_str = "_".join([str(units) for units in architecture + [1]])
    model_info = {'arch.': architecture_str, 'init': weight_scale, 'alpha': alpha, 'n_iter': n_iter, 'reg': reg_param}
    model = NetBin(X.shape[0], architecture, w_init_scale=weight_scale)
    cost = model.fit(X, np.expand_dims(y, 0), alpha, n_iter, reg_param=reg_param, print_frequency=print_freq)
    train_pred = pd.Series(model.predict(X), index=X.columns, name='predict')
    train_performance = performance(y, train_pred, dataset="train")
    test_pred = pd.Series(model.predict(X_test), index=X_test.columns, name='predict')
    test_performance = performance(y_test, test_pred, dataset="test")
    return model, cost, {**model_info, **train_performance}, {**model_info, **test_performance}


def heart_disease():
    LABEL = 'disease'
    DROP_FIRST = False  # for assigning dummy variables using pandas method
    data = preproc.from_file_with_dummies(os.path.join(DATA_DIR, "processed.cleveland.data.csv"), DROP_FIRST)
    # signal_catagorical = ['sex', 'cp', 'exang', 'slope', 'thal']  # features found to have significant differences
    # signal_quantitative = ['age', 'thalach', 'oldpeak', 'ca']
    # signal_features = signal_catagorical + signal_quantitative
    # features_to_use = [col for col in data.columns for feature in signal_features if
    #  col == feature or col.startswith(feature + preproc_cleveland.DUMMY_SEPARATOR)]
    # # data = data[[LABEL] + features_to_use]

    labels = data[LABEL]
    measurements = data.drop(LABEL, axis=1)
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(measurements, labels, test_size=0.25)

    X_train_scaled, X_train_means, X_train_stds = standardise(X_train)
    X_test_scaled = (X_test - X_train_means).div(X_train_stds)

    X = X_train_scaled.T  # rows are features, columns are training cases
    X_test = X_test_scaled.T
    y = y_train
    print(y)
    exit()

    n_features, m = X.shape
    print(f"m={m}, n_features={n_features}", "", sep="\n")
    results = pd.DataFrame(columns=['arch.', 'init', 'alpha', 'n_iter', 'reg', 'dataset',
                                    'roc_auc', 'sens.', 'spec.', 'acc.', 'tn', 'fp', 'fn', 'tp'])
    n_iter = int(1 * 1e4)
    alpha = 0.15
    n_print_statements = 5
    print_freq = n_print_statements / n_iter

    # find an example where it fails to learn, so we can debug it specifically.
    # ie need to find what the initial parameters are
    for architecture in [[2, 2]]:
        for reg_param in [2]:
            for w_init in [0.01]:  #  [0.001, 0.01, 0.1, 1, 10]:
                for alpha in [0.1]:
                    np.random.seed(10)
                    for i in range(3):
                        model, cost, train_result, test_result = experiment(X, y, X_test, y_test,
                                                                            architecture=architecture,
                                                                            weight_scale=w_init, alpha=alpha,
                                                                            n_iter=n_iter, reg_param=reg_param,
                                                                            print_freq=print_freq)
                        results = results.append(train_result, ignore_index=True)
                        results = results.append(test_result, ignore_index=True)
                        print(f"i={i}; reg_param={reg_param}; w_init={w_init}; alpha={alpha}; cost={cost}")
                        print()

    print(results.sort_values(['dataset', 'init']))
    exit()

    print("------------")
    np.random.seed(10)  # this one learns. it has a higher weight scale
    model1, cost1, train_result1, test_result1 = experiment(X, y, X_test, y_test, architecture=[2, 2],
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
    heart_disease()
