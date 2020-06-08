import numpy as np
from ml.models import NetBin

# for np isclose, it returns: absolute(a - b) <= (atol + rtol * absolute(b))
ABS_TOL = 1e-8
REL_TOL = 0


def numerical_grad():
    """Comparing weight and bias derivatives calculated by backprob with gradients calculated numerically."""
    num_features, m = 3, 1
    seed = 10
    np.random.seed(seed)
    X = np.random.randn(num_features, m)  # generate fake dataset
    y = np.expand_dims(np.array([1] * m), 0)
    epsilon = 1e-7
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
    assert max_bias_abs_diff < ABS_TOL
    print(f"Passed bias gradient test when reg_param={reg_param}. Max abs difference={max_bias_abs_diff}")

    # Testing weight derivatives:
    weight_abs_diffs = []
    for layer in range(1, len(layers)):
        for unit in range(layers[layer]):
            for weight in range(layers[layer - 1]):
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
    assert max_weight_abs_diff < ABS_TOL, f"Max abs weight diff={max_weight_abs_diff}"
    print(f"Passed weight gradient test when reg_param={reg_param}. Max abs difference={max_weight_abs_diff}")


def forward_network():
    # NB code below counts on these values and dimensions being fixed
    X = np.array([
        [1/2, 1/5],
        [1/3, 1/7]])
    y = np.array([[1, 0]])
    W1 = np.array([
        [1/11, 1/13],
        [1/17, 1/19]])
    b1 = np.array([[1/23],
                   [1/29]])
    W2 = np.array([
        [1/31, 1/37],
        [1/41, 1/43]])
    b2 = np.array([
        [1/47],
        [1/53]])
    W3 = np.array([[1/59, 1/61]])
    b3 = np.array([[1/67]])
    regularisation_parameter = 10

    print("Input data and model parameters:")
    print(f"X", f"{X}", f"y", f"{y}", f"W1", f"{W1}", f"b1", f"{b1}", f"W2", f"{W2}", f"b2", f"{b2}",
          f"lambda", f"{regularisation_parameter}", "", sep="\n")

    Z1 = np.array([
        [(1/11)*(1/2)+(1/13)*(1/3) + (1/23), (1/11)*(1/5)+(1/13)*(1/7) + (1/23)],
        [(1/17)*(1/2)+(1/19)*(1/3) + (1/29), (1/17)*(1/5)+(1/19)*(1/7) + (1/29)]
    ])
    A1 = np.tanh(Z1)

    Z2 = np.array([
        [(1/31)*A1[0, 0]+(1/37)*A1[1, 0] + (1/47), (1/31)*A1[0, 1]+(1/37)*A1[1, 1] + (1/47)],
        [(1/41)*A1[0, 0]+(1/43)*A1[1, 0] + (1/53), (1/41)*A1[0, 1]+(1/43)*A1[1, 1] + (1/53)]
    ])
    A2 = np.tanh(Z2)

    Z3 = np.array([
        [(1/59)*A2[0, 0]+(1/61)*A2[1, 0] + (1/67), (1/59)*A2[0, 1]+(1/61)*A2[1, 1] + (1/67)]
    ])
    A3 = 1/(1 + np.exp(-Z3))

    expected_cost = -(1/2) * (
            1 * np.log(0.50394274 + 1e-20) + (1-1) * np.log(1-0.50394274+1e-20) +
            0 * np.log(0.50392714 + 1e-20) + (1-0) * np.log(1-0.50392714+1e-20))

    expected_reg_term = (regularisation_parameter / (2 * 2)) * ((1/11)**2 + (1/13)**2 + (1/17)**2 + (1/19)**2 + (1/31)**2 +
                                                        (1/37)**2 + (1/41)**2 + (1/43)**2 + (1/59)**2 + (1/61)**2)
    expected_final_cost = expected_cost + expected_reg_term
    print("Expected calculated values:")
    print(f"Z1", f"{Z1}", f"A1", f"{A1}", f"Z2", f"{Z2}", f"A2", f"{A2}", f"Z3", f"{Z3}", f"A3", f"{A3}",
          f"raw cost", f"{expected_cost}", f"reg_term", f"{expected_reg_term}", f"final cost", f"{expected_final_cost}",
          sep="\n")
    print("---------------------")

    # NB random seed not necessary because we hard code the initial values
    model = NetBin(X.shape[1], [2, 2], w_init_scale=0, predict_thresh=0.5)
    model.weights[1] = W1  # overriding initial params
    model.weights[2] = W2
    model.weights[3] = W3
    model.biases[1] = b1
    model.biases[2] = b2
    model.biases[3] = b3

    print(f"Initial weights:")  # printing these to confirm that overriding the intial values worked
    for w in model.weights:
        print(w)
    print()
    print(f"Initial biases:")
    for b in model.biases:
        print(b)
    print()
    print("---------------------")

    cost = model._forward(X, y, reg_param=regularisation_parameter)
    print("Intermediate values as calculated by model class:")
    print("Signals (linear activations, {Z}):")
    for Z in model.signals:
        print(Z)
    print()
    print("Activations {A}:")  # NB the 0th layer activations are the input data, X
    for A in model.activations:
        print(A)
    print()
    print(f"cost calculated by model, with regularisation={cost}")
    print()
    assert model.signals[1].shape == Z1.shape, "Failed network signals dimension"
    assert np.allclose(model.signals[1].squeeze(), Z1.squeeze(), rtol=REL_TOL, atol=ABS_TOL)
    assert model.activations[1].shape == A1.shape, "Failed network activations dimension"
    assert np.allclose(model.activations[1].squeeze(), A1.squeeze(), rtol=REL_TOL, atol=ABS_TOL)

    assert model.signals[2].shape == Z2.shape, "Failed network signals dimension"
    assert np.allclose(model.signals[2].squeeze(), Z2.squeeze(), rtol=REL_TOL, atol=ABS_TOL)
    assert model.activations[2].shape == A2.shape, "Failed network activations dimension"
    assert np.allclose(model.activations[2].squeeze(), A2.squeeze(), rtol=REL_TOL, atol=ABS_TOL)

    assert model.signals[3].shape == Z3.shape, "Failed network signals dimension"
    assert np.allclose(model.signals[3].squeeze(), Z3.squeeze(), rtol=REL_TOL, atol=ABS_TOL)
    assert model.activations[3].shape == A3.shape, "Failed network activations dimension"
    assert np.allclose(model.activations[3].squeeze(), A3.squeeze(), rtol=REL_TOL, atol=ABS_TOL)

    assert np.isclose(cost, expected_final_cost, rtol=REL_TOL, atol=ABS_TOL), "Failed cost equality"
    print("Unit tests completed succesfully.")


if __name__ == '__main__':
    # forward_network()
    numerical_grad()
