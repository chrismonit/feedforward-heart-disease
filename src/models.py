import numpy as np
import pandas as pd

EPS = 1e-20
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
        # print(f"LogReg, A={A}")
        cost = self.cross_entropy_cost(A, y)
        return A, cost

    @staticmethod
    def _backward(A, X, y):
        dz = A - y
        print(f"logreg dZ", f"{dz}", "", sep="\n")
        dw = np.dot(X, dz.T) / len(y)  # dz = A - Y
        # db = np.sum(dz) / len(y)
        db = np.sum(dz, axis=1, keepdims=True) / len(y)
        return dw, db

    def _update(self, dw, db, learn_rate):
        self.w = self.w - learn_rate * dw
        self.b = self.b - learn_rate * db

    # TODO could return cost values during fitting.
    def fit(self, X, y, learn_rate, num_iterations, print_frequency=0.1):
        y = np.expand_dims(y, 0)
        for i in range(num_iterations):
            A, cost = self._forward(X, y)
            dw, db = self._backward(A, X, y)
            # print(f"LogReg, dw={dw.T}", f"LogReg, db={db}", "", sep="\n")
            self._update(dw, db, learn_rate)
            if i % int(1. / print_frequency) == 0:
                print(f"LogReg: iteration={i}, cost={cost}")
        return cost

    @staticmethod
    def cross_entropy_cost(y_pred, y_true):  # TODO assumes y_true is a series, but better if more generic
        cost = np.sum(y_true * np.log(y_pred) + (1. - y_true) * np.log(1. - y_pred)) / -len(y_true)
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
    def __init__(self, num_features, hidden_layers, w_init_scale=0.01, predict_thresh=0.5):
        self.predict_thresh = predict_thresh
        self.layers = [num_features] + hidden_layers + [1]
        self.weights = [np.array([None])]
        self.biases = [np.array([None])]
        self.signals = [np.array([None])]  # array of Z
        self.activation_functions = [None] + [NetBin.tanh]*len(hidden_layers) + [NetBin.sigmoid]
        self.activation_function_derivatives = [None] + [NetBin.tanh_deriv]*len(hidden_layers) + [NetBin.sigmoid_deriv]
        for l in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[l], self.layers[l-1]) * w_init_scale)
            self.biases.append(np.zeros((self.layers[l], 1)))

    # NB will need to store the intermediate values
    def _forward(self, X, y):
        self.activations = []  # TODO may be more efficient to create single list in init and update its values
        # TODO do we need to save all the activations at all?
        self.activations.append(X)  # activations of layer 0
        for l in range(1, len(self.layers)):
            Z = np.dot(self.weights[l], self.activations[l-1]) + self.biases[l]
            A = self.activation_functions[l](Z)
            self.signals.append(Z)
            self.activations.append(A)
            # print(f"Net, layer={l}, A={A}")
        return NetBin.cost(self.activations[-1], y)

    @staticmethod
    def cost(y_pred, y_true):  # TODO regularisation
        # print(f"Net, y_pred={y_pred}", f"y_true={y_true}", "", sep="\n")
        cost = np.sum(y_true * np.log(y_pred+EPS) + ((1.-y_true)+EPS) * np.log((1.-y_pred)+EPS)) / -len(y_true)
        return cost.squeeze()

    @staticmethod
    def cost_deriv(A, y):
        return - (np.divide(y, A+EPS) - np.divide(1 - y, (1.-A)+EPS))

    @staticmethod
    def sigmoid(z):
        return 1. / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(Z):
        return LogReg.sigmoid(Z) * (1 - LogReg.sigmoid(Z))

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_deriv(Z):
        return 1 - np.power(np.tanh(Z), 2)

    # TODO need to test
    def _backward(self, y):
        next_dA = NetBin.cost_deriv(self.activations[-1], y)  # initilise as dAL
        weight_derivs = []
        bias_derivs = []
        for l in range(1, len(self.layers))[::-1]:
            g_prime_Z = self.activation_function_derivatives[l](self.signals[l])
            dZ = np.multiply(next_dA, g_prime_Z)
            print(f"Net dZ", f"{dZ}", "", sep="\n")
            dW = 1/len(y) * np.dot(dZ, self.activations[l-1].T)
            db = 1/len(y) * np.sum(dZ, axis=1, keepdims=True)
            weight_derivs.append(dW)
            bias_derivs.append(db)
            next_dA = np.dot(self.weights[l].T, dZ)
        weight_derivs.append(np.array([None]))  # for the 0th layer
        bias_derivs.append(np.array([None]))
        return weight_derivs[::-1], bias_derivs[::-1]

    def _update(self, weight_derivs, bias_derivs, learning_rate):
        for l in range(1, len(self.layers)):
            self.weights[l] -= learning_rate * weight_derivs[l]
            self.biases[l] -= learning_rate * bias_derivs[l]

    # TODO could return cost values during fitting? Better to write to log file, or checkpoint
    def fit(self, X, y, learn_rate, num_iterations, print_frequency=0.1):
        for i in range(num_iterations):
            cost = self._forward(X, y)
            weight_derivs, bias_derivs = self._backward(y)
            # print(f"Net, dw={weight_derivs[1]}", f"Net, db={bias_derivs[1]}", "", sep="\n")
            self._update(weight_derivs, bias_derivs, learn_rate)
            if i % int(1. / print_frequency) == 0:  # TODO save these values and report afterwards? also checkpoint
                print(f"NetBin: iteration={i}, cost={cost}")
        return cost

    def predict(self, X):  # TODO same logic as forward pass, can we combine these?
        A = X.copy()
        for l in range(1, len(self.layers)):
            Z = np.dot(self.weights[l], A) + self.biases[l]
            A = self.activation_functions[l](Z)
        y_pred = [int(a > self.predict_thresh) for a in A.squeeze()]
        return y_pred


if __name__ == '__main__':
    np.random.seed(10)
    num_features, m = 2, 5
    mynet = NetBin(num_features, [3, 5, 4, 2], w_init_scale=1)
    X = np.random.randn(num_features, m)
    y = np.expand_dims(np.array([1]*m), 0)

    print("Forward:")
    cost = mynet._forward(X, y)
    print(f"cost={cost}", f"", "", sep="\n")
    # print(NetBin.cost_deriv(np.expand_dims(np.array([0.5, 0.5, 0.5]), 1), np.expand_dims(np.array([0, 1, 1]), 1)))
    dW, db = mynet._backward(y)

    print("weights:")
    for w in mynet.weights:
        print(w.shape)
    print()
    print("weight derivatives")
    for dw in dW:
        print(dw.shape)
    print()
    print()
    print("biases:")
    for b in mynet.biases:
        print(b.shape)
    print()
    print("bias derivatives")
    for d_b in db:
        print(d_b.shape)
    print()
    print(f"Final cost:", mynet.fit(X, y, 0.1, 500, print_frequency=0.1), "", sep="\n")
    print(f"Predictions on training data", mynet.predict(X), "", sep="\n")
