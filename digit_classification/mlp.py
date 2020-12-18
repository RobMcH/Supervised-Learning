import numpy as np
import numba
import itertools
import tqdm
from data import read_data, random_split_indices


def cross_entropy_loss(y, y_hat):
    return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))


@numba.njit()
def softmax(x_):
    exp = np.exp(x_ - np.max(x_))
    return exp / np.sum(exp)


@numba.njit()
def sigma(x_):
    """
    Implements the logistic function.
    """
    return 1 / (1 + np.exp(-x_))


@numba.njit()
def sigma_prime(x_):
    """
    Implements the derivative of the logistic function.
    """
    s = sigma(x_)
    return s * (1 - s)


@numba.njit()
def relu(x_):
    return np.maximum(x_, 0)


@numba.njit()
def relu_prime(x_):
    return np.minimum(np.maximum(x_, 0), 1)


@numba.njit()
def predict(x_, w_, b_):
    """
    Calculates the predictions y_hat = w_ @ x_ + b.
    """
    return x_ @ w_ + b_


def forward_pass(xs, weights, return_prediction=False, activation=sigma):
    """
    Calculates one forward pass of the model. If return_predictions=True only the predictions of the model will be
    returned.
    """
    forward_dict = {}
    prev_output = xs
    for i in range(len(weights)):
        weight, bias = weights[i]
        forward_dict[f"h{i + 1}"] = predict(prev_output, weight, bias)
        if i == len(weights) - 1:
            prev_output = forward_dict[f"h{i + 1}_sigma"] = np.array(
                [softmax(forward_dict[f"h{i + 1}"][j]) for j in range(forward_dict[f"h{i + 1}"].shape[0])])
        else:
            prev_output = forward_dict[f"h{i + 1}_sigma"] = activation(forward_dict[f"h{i + 1}"])
    if return_prediction:
        return prev_output
    return forward_dict


def backward_pass(xs, ys, weights, forward_dict, activation_derivative=sigma_prime):
    """
    Calculates one backward pass of the model.
    """
    num_layers = len(forward_dict) // 2
    gradients = []
    y_hat = forward_dict[f"h{num_layers}_sigma"]
    ys = np.eye(y_hat.shape[1])[ys]
    # Delta for last layer.
    delta = (1.0 / ys.shape[0]) * (y_hat - ys)
    for i in range(num_layers - 1, -1, -1):
        # Loop through the layers and calculate the gradients for each weight/bias vector.
        prev_output = forward_dict[f"h{i}_sigma"] if i >= 1 else xs
        weight_gradient = prev_output.T @ delta
        bias_gradient = np.sum(delta, axis=0)
        gradients.append((weight_gradient, bias_gradient))
        if i != 0:
            delta = (delta @ weights[i][0].T) * activation_derivative(forward_dict[f"h{i}"])
    return gradients


def initialise_weights(layers):
    """
    Randomly initialise the weights specified by layers.
    """
    weights = []
    for i in range(len(layers)):
        input_dim, output_dim = layers[i]
        weight = np.random.randn(input_dim, output_dim) / 100
        bias = np.random.randn(output_dim) / 100
        weights.append((weight, bias))
    return weights


def analytical_gradients(xs, ys, weights, activation=sigma, activation_derivative=sigma_prime):
    """
    Calculate the analytical gradients of the model.
    """
    forward_dict = forward_pass(xs, weights, activation=activation)
    gradients = backward_pass(xs, ys, weights, forward_dict, activation_derivative=activation_derivative)
    return gradients, forward_dict


def calculate_error_loss(xs, weights, true):
    """
    Calculate the accuracy as well as the loss for a given dataset (xs, true) and weights.
    """
    predictions = forward_pass(xs, weights, return_prediction=True)
    ys = np.eye(predictions.shape[1])[true]
    return cross_entropy_loss(ys, predictions) / predictions.shape[0], (
            predictions.argmax(axis=1) != true).sum() / true.size * 100.0


def update_weights(weights, gradients, learning_rate, layer_count, momentum, prev_gradients):
    """
    Updates the weights of all of the layers for given gradients.
    :param weights: The weights to be updated.
    :param gradients: The gradients w.r.t. the error.
    :param learning_rate: The learning rate.
    :param layer_count: The number of layers in the model.
    :param momentum: The momentum coefficient. Set to 0 to disable momentum.
    :param prev_gradients: The previous gradients (used for momentum).
    :return: The updated weights.
    """
    for i in range(len(gradients)):
        weight_gradient, bias_gradient = gradients[i]
        prev_weight_gradient, prev_bias_gradient = prev_gradients[i]
        weight, bias = weights[layer_count - i - 1]
        weight -= learning_rate * weight_gradient + momentum * prev_weight_gradient
        bias -= learning_rate * bias_gradient + momentum * prev_bias_gradient
        weights[layer_count - i - 1] = (weight, bias)
    return weights


def train_mlp(xs, ys, epochs, learning_rate, layers, optimizer=update_weights, batching="Full", batch_size=0,
              momentum=0.0, return_metrics=False, return_best_weights=False, print_metrics=True, test_xs=None,
              test_ys=None):
    """
    Trains a multi-layer perceptron. The training is performed by gradient descent and backpropagation. The training
    supports different batch modes ('Full', 'Mini', and 'SGD'). The parameter batch_size specifies the size of a single
    batch for the batch mode 'Mini' (otherwise the parameter is ignored). The momentum parameter specifies the momentum
    coefficient. To disable momentum the parameter can be set to 0.0 (default).
    :param xs: Training data.
    :param ys: Training targets.
    :param epochs: The number of epochs the training should be run for.
    :param learning_rate: The learning rate for the weight updates.
    :param layers: The specification of the layers. An iterable containing tuples of (output_dimension, input_dimension)
    are expected. Each tuple specifies a single layer in the network.
    :param optimizer: Function to perform the weight update.
    :param batching: Batch mode. Either 'Full' (default), 'Mini', or SGD.
    :param batch_size: The size of a single batch. Only relevant for batch mode 'Mini'.
    :param momentum: The momentum coefficient. Set to 0 to disable momentum (default).
    :param return_metrics: Returns a dictionary containing all the training/validation accuracies and losses per epoch.
    :param return_best_weights: Return the best weights (defined by the highest validation accuracy).
    :param print_metrics: Print the metrics during training.
    :return: weights [list], (metrics [dict])
    """
    weights = initialise_weights(layers)
    best_weights, best_dev_acc = None, 0.0
    y_hat_alias, layer_count = f"h{len(layers)}_sigma", len(layers)
    metrics = {"train_loss": np.zeros(epochs), "train_err": np.zeros(epochs), "test_loss": np.zeros(epochs),
               "test_err": np.zeros(epochs)}
    prev_gradients = [(np.zeros_like(weight[0]), np.zeros_like(weight[1])) for weight in weights]
    # Reverse order of initial prev_gradients (based on weight shapes) as the gradients start from the last layer.
    prev_gradients.reverse()
    # RNG
    rng = np.random.default_rng()
    # Split the training data into full/mini/SGD batches.
    if batching != "Full":
        if batching == "Mini" and batch_size > 0:
            # Split into batches of size batch_size with the remainder being omitted.
            x_batches = np.array_split(xs, np.arange(batch_size, xs.shape[0], batch_size))
            y_batches = np.array_split(ys, np.arange(batch_size, ys.shape[0], batch_size))
        elif batching == "SGD":
            # Split into batches of size 1. This allows reusing the same training code for mini-batches.
            x_batches = np.hsplit(xs, xs.shape[1])
            y_batches = np.hsplit(ys, ys.shape[1])
        else:
            raise ValueError("Parameter batching must be one either 'Full', 'Mini' or 'SGD'.")
    # Training
    for epoch in range(epochs):
        if batching != "Full":
            if batching == "SGD":
                # Shuffle data for SGD. xs and ys are shuffled in the exact same way.
                rng_state = rng.__getstate__()
                rng.shuffle(x_batches)
                rng.__setstate__(rng_state)
                rng.shuffle(y_batches)
            for i in range(len(x_batches)):
                # Train on mini-batches. prev_gradients is only used if momentum > 0.0.
                gradients, forward_dict = analytical_gradients(x_batches[i], y_batches[i], weights)
                weights = optimizer(weights, gradients, learning_rate, layer_count, momentum, prev_gradients)
                prev_gradients = gradients
        else:
            # Train on full batch. prev_gradients is only used if momentum > 0.0.
            gradients, forward_dict = analytical_gradients(xs, ys, weights)
            weights = optimizer(weights, gradients, learning_rate, layer_count, momentum, prev_gradients)
            prev_gradients = gradients
        # Calculate accuracy and loss on the training and validation sets.
        train_loss, train_error = calculate_error_loss(xs, weights, ys)
        metrics["train_err"][epoch] = train_error
        metrics["train_loss"][epoch] = train_loss
        if test_xs is not None and test_ys is not None:
            test_loss, test_error = calculate_error_loss(test_xs, weights, test_ys)
            metrics["test_err"][epoch] = test_error
            metrics["test_loss"][epoch] = test_loss
            if print_metrics:
                print(
                    f"Epoch {epoch} - Training loss {train_loss} - Training error {train_error} - Test loss {test_loss}"
                    f" - Test error {test_error}")
    if return_best_weights:
        weights = best_weights
    if return_metrics:
        return weights, metrics
    return weights


def search_nn_architecture():
    x, y = read_data("data/zipcombo.dat")
    indices = np.arange(0, y.size)
    index_splits = [random_split_indices(indices, 0.8) for i in range(20)]
    num_hidden_layers = [1, 2, 3]
    batch_sizes = [16, 32, 64]
    momentum = [0.0, 0.9]
    factor = 0.75
    for n, b, m in itertools.product(num_hidden_layers, batch_sizes, momentum):
        layer_definition = [(16 * 16, 192)]
        for hidden_layer in range(n):
            layer_definition.append((layer_definition[-1][1], int(layer_definition[-1][1] * factor)))
        layer_definition.append((layer_definition[-1][1], 10))
        errors, losses = [], []
        for i in tqdm.trange(20):
            train, test = index_splits[i]
            weights, metrics = train_mlp(x[train], y[train], 100, 0.1, layer_definition, return_metrics=True,
                                         batching="Mini", batch_size=b, momentum=m, print_metrics=False,
                                         test_xs=x[test], test_ys=y[test])
            errors.append(metrics["test_err"].min())
            losses.append(metrics["test_loss"].min())
        print(f"num hidden layers: {n} - batch size: {b} - momentum: {m} - avg. error {np.average(errors)}"
              f" - avg. loss {np.average(losses)}")


if __name__ == '__main__':
    search_nn_architecture()
