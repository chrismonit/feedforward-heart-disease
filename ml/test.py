import numpy as np
from ml.models import NetBin


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


if __name__ == '__main__':
    numerical_grad()