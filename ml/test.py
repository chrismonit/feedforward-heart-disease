import numpy as np
from ml.models import NetBin


def numerical_grad():
    """Comparing weight and bias derivatives calculated by backprob with gradients calculated numerically."""
    num_features, m = 3, 1
    seed = 10
    np.random.seed(seed)
    X = np.random.randn(num_features, m)  # generate fake dataset
    y = np.expand_dims(np.array([1]*m), 0)
    epsilon = 1e-7
    tolerance = 1e-6
    print(f"X={X}", f"y={y}", f"epsilon={epsilon}", "", sep="\n")
    hidden_layers = [2, 2]
    layers = [num_features] + hidden_layers + [1]
    w_init_scale = 0.01
    reg_param = 2

    # Testing bias derivatives:
    bias_abs_diffs = []
    for layer in range(1, len(layers)):
        for unit in range(layers[layer]):
            np.random.seed(seed)  # NB biases are not necessarily initialised stochastically
            model_minus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
            model_minus.biases[layer][unit] -= epsilon
            minus_cost = model_minus._forward(X, y, reg_param=reg_param)

            np.random.seed(seed)
            model_plus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
            model_plus.biases[layer][unit] += epsilon
            plus_cost = model_plus._forward(X, y, reg_param=reg_param)
            approx_grad = (plus_cost - minus_cost) / (2 * epsilon)

            np.random.seed(seed)
            model2 = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
            cost = model2._forward(X, y, reg_param=reg_param)
            weight_derivs, bias_derivs = model2._backward(y, reg_param)
            backprob_deriv = bias_derivs[layer][unit]
            bias_abs_diff = np.abs(approx_grad - backprob_deriv)
            bias_abs_diffs.append(bias_abs_diff)
    max_bias_abs_diff = np.max(bias_abs_diffs)
    assert max_bias_abs_diff < tolerance
    print(f"Passed bias gradient test when reg_param={reg_param}. Max abs difference={max_bias_abs_diff}")

    # Testing weight derivatives:
    weight_abs_diffs = []
    for layer in range(1, len(layers)):
        for unit in range(layers[layer]):
            for weight in range(layers[layer-1]):
                # np.random.seed(seed)  # not necessary, used to compare with others because of random initial weights
                # model_orig = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                # orig_cost = model_orig._forward(X, y, reg_param=reg_param)
                np.random.seed(seed)  # weights are initialised stochastically, so must reset seed every time
                model_minus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                model_minus.weights[layer][unit, weight] -= epsilon
                minus_cost = model_minus._forward(X, y, reg_param=reg_param)

                np.random.seed(seed)
                model_plus = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                model_plus.weights[layer][unit, weight] += epsilon
                plus_cost = model_plus._forward(X, y, reg_param=reg_param)
                approx_grad = (plus_cost - minus_cost) / (2 * epsilon)

                np.random.seed(seed)  # we effectively clone the models used above, ie has same initial random weights
                model2 = NetBin(num_features, hidden_layers, w_init_scale=w_init_scale)
                cost = model2._forward(X, y, reg_param=reg_param)
                weight_derivs, bias_derivs = model2._backward(y, reg_param)
                abs_diff = np.abs(approx_grad - weight_derivs[layer][unit, weight])
                weight_abs_diffs.append(abs_diff)
    max_weight_abs_diff = np.max(weight_abs_diffs)
    assert max_weight_abs_diff < tolerance, f"Max abs weight diff={max_weight_abs_diff}"
    print(f"Passed weight gradient test when reg_param={reg_param}. Max abs difference={max_weight_abs_diff}")


if __name__ == '__main__':
    numerical_grad()