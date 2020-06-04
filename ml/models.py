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
        cost = self.cross_entropy_cost(A, y)
        return A, cost

    @staticmethod
    def _backward(A, X, y):
        dz = A - y
        dw = np.dot(X, dz.T) / len(y)  # dz = A - Y  # TODO warning, may need to be using y.shape[1] instead
        # db = np.sum(dz) / len(y)
        db = np.sum(dz, axis=1, keepdims=True) / len(y)  # TODO warning, may need to be using y.shape[1] instead
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


class NetBin:
    def __init__(self, num_features, hidden_layers, w_init_scale=0.01, predict_thresh=0.5):
        self.predict_thresh = predict_thresh
        self.layers = [num_features] + hidden_layers + [1]
        self.weights = [np.array([None])]
        self.biases = [np.array([None])]
        self.activation_functions = [None] + [NetBin._tanh] * len(hidden_layers) + [NetBin._sigmoid]
        self.activation_function_derivatives = [None] + [NetBin._tanh_deriv] * len(hidden_layers) + [NetBin._sigmoid_deriv]
        for l in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[l], self.layers[l-1]) * w_init_scale)
            self.biases.append(np.zeros((self.layers[l], 1)))

    # def init_parameters(self, w_init_scale):  # TODO not sure this is necessary after all
    #     init_weights, init_biases = [], []
    #     for l in range(1, len(self.layers)):
    #         init_weights.append(np.random.randn(self.layers[l], self.layers[l - 1]) * w_init_scale)
    #         init_biases.append(np.zeros((self.layers[l], 1)))
    #     return init_weights, init_biases

    @staticmethod
    def _cost(y_pred, y_true):
        cost = np.sum(y_true * np.log(y_pred+EPS) + ((1.-y_true)+EPS) * np.log((1.-y_pred)+EPS)) / -len(y_true)  # TODO possible bug, need y.shape[1] instead
        return cost.squeeze()

    @staticmethod
    def _cost_deriv(A, y):
        return - (np.divide(y, A+EPS) - np.divide(1 - y, (1.-A)+EPS))

    @staticmethod
    def _sigmoid(z):
        return 1. / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(Z):
        return LogReg.sigmoid(Z) * (1 - LogReg.sigmoid(Z))

    @staticmethod
    def _tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def _tanh_deriv(Z):
        return 1 - np.power(np.tanh(Z), 2)

    def _regularisation(self, reg_param, n_cases):  # TODO should be a wrapper for other functions
        """Regularisation by weight decay, ie Frobenius norm of weights. Does nothing with biases"""
        # TODO this is memory inefficient, no need to have list of norm values
        sum_frob = np.sum([np.linalg.norm(self.weights[l], ord='fro') for l in range(1, len(self.layers))])
        return (reg_param / (2 * n_cases)) * sum_frob

    def _forward(self, X, y, reg_param):
        self.activations = [X]
        self.signals = [np.array([None])]  # array of Z
        for l in range(1, len(self.layers)):
            Z = np.dot(self.weights[l], self.activations[l - 1]) + self.biases[l]
            A = self.activation_functions[l](Z)
            self.signals.append(Z)
            self.activations.append(A)
        return NetBin._cost(self.activations[-1], y) + self._regularisation(reg_param, X.shape[1])

    def _backward(self, y, reg_param):
        next_dA = NetBin._cost_deriv(self.activations[-1], y)  # initilise as dAL
        weight_derivs = []
        bias_derivs = []
        for l in range(1, len(self.layers))[::-1]:
            g_prime_Z = self.activation_function_derivatives[l](self.signals[l])
            dZ = np.multiply(next_dA, g_prime_Z)
            reg_term = (reg_param / y.shape[1]) * self.weights[l]
            dW = 1/y.shape[1] * np.dot(dZ, self.activations[l-1].T) + reg_term
            db = 1/y.shape[1] * np.sum(dZ, axis=1, keepdims=True)  # TODO this has changed, need to test backprop
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
    #  or like a file buffer, which can be on disk or stdout
    def fit(self, X, y, learn_rate, num_iterations, reg_param=0, print_frequency=0.1):
        for i in range(num_iterations):
            cost = self._forward(X, y, reg_param)
            weight_derivs, bias_derivs = self._backward(y, reg_param)
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


def numerical_grad():
    num_features, m = 3, 1
    seed = 10
    np.random.seed(seed)
    X = np.random.randn(num_features, m)
    y = np.expand_dims(np.array([1]*m), 0)
    print(f"X={X}", f"y={y}", "", sep="\n")
    epsilon = 1e-7
    print(f"epsilon={epsilon}")
    hidden_layers = [2, 2, 2]
    layers = [num_features] + hidden_layers + [1]
    w_init_scale = 1
    reg_param = 0
    layer = 2
    unit = 1
    weight = 0

    # Testing bias derivatives:
    np.random.seed(seed)
    model_orig = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
    for layer_biases in model_orig.biases:
        print(layer_biases)

    np.random.seed(seed)
    model_minus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
    model_minus.biases[layer][unit] -= epsilon
    for layer_biases in model_minus.biases:
        print(layer_biases)
    minus_cost = model_minus._forward(X, y, reg_param=reg_param)
    print(f"minus cost={minus_cost}")

    np.random.seed(seed)
    model_plus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
    model_plus.biases[layer][unit] += epsilon
    for layer_biases in model_plus.biases:
        print(layer_biases)
    plus_cost = model_plus._forward(X, y, reg_param=reg_param)
    print(f"plus cost={plus_cost}")
    approx_grad = (plus_cost - minus_cost) / (2 * epsilon)
    print(f"approx_grad={approx_grad}")

    print("model2")
    np.random.seed(seed)
    model2 = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
    cost = model2._forward(X, y, reg_param=reg_param)
    weight_derivs, bias_derivs = model2._backward(y, reg_param)
    for layer_bias_derivs in bias_derivs:
        print(layer_bias_derivs)
    print(f"backgrop grad={bias_derivs[layer][unit]}")
    print("----------------------")
    abs_diff = np.abs(approx_grad - bias_derivs[layer][unit])
    print(f"abs_diff={abs_diff}")
    # abs_diffs.append(abs_diff)
    print()

    exit()

    # Testing weight derivatives:
    abs_diffs = []
    for layer in range(1, len(layers)):
        for unit in range(layers[layer]):
            for weight in range(layers[layer-1]):
                print(layer, unit, weight)
                np.random.seed(seed)
                model_orig = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                print("model_orig initial weights")
                for layer_weights in model_orig.weights:
                    print(layer_weights)
                print()
                orig_cost = model_orig._forward(X, y, reg_param=reg_param)
                print(f"model_orig cost={orig_cost}")
                print("----------------------")

                np.random.seed(seed)
                model_minus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                print("model_minus initial weights")
                for layer_weights in model_minus.weights:
                    print(layer_weights)
                print()
                model_minus.weights[layer][unit, weight] -= epsilon
                print("model_minus initial weights, after modification")
                for layer_weights in model_minus.weights:
                    print(layer_weights)
                print()
                minus_cost = model_minus._forward(X, y, reg_param=reg_param)
                print(f"model_minus cost={minus_cost}")
                print("----------------------")

                np.random.seed(seed)
                model_plus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                print("model_plus initial weights")
                for layer_weights in model_plus.weights:
                    print(layer_weights)
                print()
                model_plus.weights[layer][unit, weight] += epsilon
                print("model_plus initial weights, after modification")
                for layer_weights in model_plus.weights:
                    print(layer_weights)
                print()
                plus_cost = model_plus._forward(X, y, reg_param=reg_param)
                print(f"model_plus cost={plus_cost}")
                print("----------------------")
                approx_grad = (plus_cost - minus_cost) / (2 * epsilon)
                print(f"approx_grad={approx_grad}")
                print("----------------------")

                print("model2")
                np.random.seed(seed)
                model2 = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                cost = model2._forward(X, y, reg_param=reg_param)
                weight_derivs, bias_derivs = model2._backward(y, reg_param)
                for layer_weight_derivs in weight_derivs:
                    print(layer_weight_derivs)
                print("----------------------")
                abs_diff = np.abs(approx_grad - weight_derivs[layer][unit, weight])
                print(f"abs_diff={abs_diff}")
                abs_diffs.append(abs_diff)
                print("----------------------")
                print("----------------------")
    print()
    print(abs_diffs)
    print(np.max(abs_diffs))


def calculations():
    np.random.seed(10)
    num_features, m = 2, 5
    model = NetBin(num_features, [2, 2], w_init_scale=1)
    X = np.random.randn(num_features, m)
    y = np.expand_dims(np.array([1] * m), 0)
    print(f"X={X}", f"y={y}", "", sep="\n")

    print(f"model weights")
    for w in model.weights:
        print(w)
    print()

    cost_no_reg = model._forward(X, y, reg_param=0)
    print(f"cost without reg={cost_no_reg}", "", sep="\n")

    reg = 10
    print(f"reg term={model._regularisation(reg, m)}", "", sep="\n")
    cost_reg = model._forward(X, y, reg_param=reg)
    print(f"cost with reg ({reg}) = {cost_reg}", "", sep="\n")
    # dW, db = mynet._backward(y)

    print("backward, without reg")
    weight_derivs, bias_derivs = model._backward(y, reg_param=0)
    for dw in weight_derivs:
        print(dw)
    for db in bias_derivs:
        print(db)
    print()

    reg = 1
    print(f"backward, with reg={reg}")
    weight_derivs, bias_derivs = model._backward(y, reg_param=1)
    for dw in weight_derivs:
        print(dw)
    for db in bias_derivs:
        print(db)

    # print(f"Final cost:", mynet.fit(X, y, 0.1, 500, print_frequency=0.1), "", sep="\n")
    # print(f"Predictions on training data", mynet.predict(X), "", sep="\n")

if __name__ == '__main__':
    numerical_grad()
