import numpy as np

EPS = 1e-20
# TODO could check if this satisfies sci kit learn's api as a classifier


class NetBin:
    """Artificial neural network for binary classification"""
    # TODO make hidden layer activation functions soft coded
    def __init__(self, num_features, hidden_layers, w_init_scale=0.01, predict_thresh=0.5):
        self.predict_thresh = predict_thresh
        self.layers = [num_features] + hidden_layers + [1]
        self.weights = [np.array([None])]
        self.biases = [np.array([None])]
        self.activation_functions = [None] + [NetBin._tanh] * len(hidden_layers) + [NetBin._sigmoid]
        self.activation_function_derivs = [None] + [NetBin._tanh_deriv] * len(hidden_layers) + [NetBin._sigmoid_deriv]
        for l in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[l], self.layers[l-1]) * w_init_scale)
            self.biases.append(np.zeros((self.layers[l], 1)))

    @staticmethod
    def _cost(y_pred, y_true):
        cost = np.sum(y_true * np.log(y_pred+EPS) + ((1.-y_true)+EPS) * np.log((1.-y_pred)+EPS)) / -(y_true.shape[1])
        return cost.squeeze()

    @staticmethod
    def _cost_deriv(A, y):
        return - (np.divide(y, A+EPS) - np.divide(1 - y, (1.-A)+EPS))

    @staticmethod
    def _sigmoid(z):
        return 1. / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(Z):
        return NetBin.sigmoid(Z) * (1 - NetBin.sigmoid(Z))

    @staticmethod
    def _tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def _tanh_deriv(Z):
        return 1 - np.power(np.tanh(Z), 2)

    @staticmethod
    def _relu(Z):
        return np.maximum(Z, 0)  # NB there are faster ways to implement this

    @staticmethod
    def _relu_deriv(Z):
        return 1 * (Z > 0.0)

    def _regularisation(self, reg_param, n_cases):  # TODO should be a wrapper for other functions
        """Regularisation by weight decay, ie square Frobenius norm of weights. Does nothing with biases"""
        # TODO can make this much more efficient
        sum_frob = np.sum([np.power(np.linalg.norm(self.weights[l], ord='fro'), 2) for l in range(1, len(self.layers))])
        return (reg_param / (2 * n_cases)) * sum_frob

    def _forward(self, X, y, reg_param):  # TODO fail more gracefully if X and y dimensions don't work
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
            g_prime_Z = self.activation_function_derivs[l](self.signals[l])
            dZ = np.multiply(next_dA, g_prime_Z)
            reg_term = (reg_param / y.shape[1]) * self.weights[l]
            dW = (1/y.shape[1]) * np.dot(dZ, self.activations[l-1].T) + reg_term
            db = (1/y.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
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
            if i % int(1. / print_frequency) == 0:
                print(f"NetBin: iteration={i}, cost={cost}")
        return cost

    def predict_scores(self, X):
        A = X.copy()
        for l in range(1, len(self.layers)):
            Z = np.dot(self.weights[l], A) + self.biases[l]
            A = self.activation_functions[l](Z)
        return A

    def predict(self, X):  # TODO same logic as forward pass, can we combine these?
        A = self.predict_scores(X)
        y_pred = [int(a > self.predict_thresh) for a in A.squeeze()]
        return y_pred


def _test_calculations():
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
    _test_calculations()
