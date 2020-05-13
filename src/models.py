import numpy as np


# TODO regularisation? Class balancing?
# TODO could check if this satisfies sci kit learn's api as a classifier
class LogReg:
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
        A = LogReg.sigmoid(Z)
        y_pred = [int(a > self.predict_thresh) for a in A.squeeze()]
        return y_pred


# NB if hidden layers is empty, or has single value 0, should become logreg? something like that
class NetBin:
    def __init__(self, num_features, hidden_layers, w_init_scale=0.01):
        layers = [num_features] + hidden_layers + [1]
        self.weights = []
        self.biases = []
        for l in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[l], layers[l-1]) * w_init_scale)
            self.biases.append(np.zeros((layers[l], 1)))

        for l in range(len(self.weights)):
            print(f"Layer={l+1}", self.weights[l], self.weights[l].shape, self.biases[l], "", sep="\n")


if __name__ == '__main__':
    np.random.seed(10)
    mynet = NetBin(2, [3, 5, 4, 2])

