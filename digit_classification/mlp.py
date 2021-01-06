import numpy as np
import numba
import copy


np.random.seed(42)


def reset_seed(seed=42):
    np.random.seed(seed)


@numba.njit(parallel=True)
def cross_entropy_loss(y, y_hat):
    # Implements the cross-entropy loss.
    loss = -y * np.log(y_hat + 1e-8)
    for i in numba.prange(loss.shape[1]):
        loss[np.isnan(loss[:, i]), i] = 0.0
        loss[np.isinf(loss[:, i]), i] = np.finfo(loss.dtype).max
    return np.sum(loss) / y.shape[0]


@numba.njit()
def softmax(x_):
    # Implements a numerically stable softmax function.
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
    # Implements the ReLU activation function. Not used.
    return np.maximum(x_, 0)


@numba.njit()
def relu_prime(x_):
    #  Implements the ReLU derivative. Not used.
    return np.minimum(np.maximum(x_, 0), 1)


@numba.njit()
def predict(x_, w_, b_):
    """
    Calculates the predictions y_hat = w_ @ x_ + b.
    """
    return x_ @ w_ + b_


def l1_prime(weight):
    # Implements L1 regularisation.
    return np.where(weight >= 0, 1.0, -1.0)


def l2_prime(weight):
    # Implements L2 regularisation.
    return weight


def forward_pass(xs, weights, return_prediction=False, activation=sigma):
    """
    Calculates one forward pass of the model. If return_predictions=True only the predictions of the model will be
    returned.
    """
    # First forward dict contains outputs, second forward dict contains outputs after feeding them through a logistic
    # function.
    forward_dicts = [{}, {}]
    prev_output = xs
    # Loop through layers, calculate outputs and feed them through activation functions.
    for i in range(len(weights)):
        weight, bias = weights[i]
        forward_dicts[0][i + 1] = predict(prev_output, weight, bias)
        if i == len(weights) - 1:
            prev_output = forward_dicts[1][i + 1] = np.array(
                [softmax(forward_dicts[0][i + 1][j]) for j in range(forward_dicts[0][i + 1].shape[0])])
        else:
            prev_output = forward_dicts[1][i + 1] = activation(forward_dicts[0][i + 1])
    if return_prediction:
        return prev_output
    return forward_dicts


def backward_pass(xs, ys, weights, forward_dicts, activation_derivative=sigma_prime, l1_reg=0.0, l2_reg=0.0):
    """
    Calculates one backward pass of the model.
    """
    num_layers = len(forward_dicts[0])
    gradients = []
    y_hat = forward_dicts[1][num_layers]
    ys = np.eye(y_hat.shape[1])[ys]
    # Delta for last layer.
    delta = (1.0 / ys.shape[0]) * (y_hat - ys)
    for i in range(num_layers - 1, -1, -1):
        # Loop through the layers and calculate the gradients for each weight/bias vector.
        prev_output = forward_dicts[1][i] if i >= 1 else xs
        weight_gradient = prev_output.T @ delta + l1_reg * l1_prime(weights[i][0]) + l2_reg * l2_prime(weights[i][0])
        bias_gradient = np.sum(delta, axis=0)
        gradients.append((weight_gradient, bias_gradient))
        if i != 0:
            delta = (delta @ weights[i][0].T) * activation_derivative(forward_dicts[0][i])
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


def analytical_gradients(xs, ys, weights, activation=sigma, activation_derivative=sigma_prime, l1_reg=0.0, l2_reg=0.0):
    """
    Calculate the analytical gradients of the model.
    """
    forward_dict = forward_pass(xs, weights, activation=activation)
    gradients = backward_pass(xs, ys, weights, forward_dict, activation_derivative=activation_derivative, l1_reg=l1_reg,
                              l2_reg=l2_reg)
    return gradients, forward_dict


def calculate_error_loss(xs, weights, true):
    """
    Calculate the error as well as the loss for a given dataset (xs, true) and weights.
    """
    predictions = forward_pass(xs, weights, return_prediction=True)
    ys = np.eye(predictions.shape[1])[true]
    loss = cross_entropy_loss(ys, predictions) / predictions.shape[0]
    error = (predictions.argmax(axis=1) != true).sum() / true.size * 100.0
    return loss, error


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
              momentum=0.0, l1_reg=0.0, l2_reg=0.0, return_metrics=False, return_best_weights=False, print_metrics=True,
              test_xs=None, test_ys=None):
    """
    Trains a multi-layer perceptron. The training is performed by gradient descent and backpropagation. The training
    supports different batch modes ('Full', 'Mini', and 'SGD'). The parameter batch_size specifies the size of a single
    batch for the batch mode 'Mini' (otherwise the parameter is ignored). The momentum parameter specifies the momentum
    coefficient. To disable momentum the parameter can be set to 0.0 (default). The parameters l1_reg and l2_reg control
    the regularisation coefficients. To disable regularisation set the parameters to 0.0 (default).
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
    :param l1_reg: L1 regularisation coefficient.
    :param l2_reg: L2 regularisation coefficient.
    :param return_metrics: Returns a dictionary containing all the training/validation accuracies and losses per epoch.
    :param return_best_weights: Return the best weights (defined by the highest validation accuracy).
    :param print_metrics: Print the metrics during training.
    :param test_xs: Test data.
    :param test_ys: Test labels.
    :return: weights [list], (metrics [dict])
    """
    weights = initialise_weights(layers)
    best_weights, best_error = weights, 100.0
    y_hat_alias, layer_count = f"h{len(layers)}_sigma", len(layers)
    metrics = {"train_loss": np.zeros(epochs), "train_err": np.zeros(epochs), "test_loss": np.zeros(epochs),
               "test_err": np.zeros(epochs)}
    prev_gradients = [(np.zeros_like(weight[0]), np.zeros_like(weight[1])) for weight in weights]
    # Reverse order of initial prev_gradients (based on weight shapes) as the gradients start from the last layer.
    prev_gradients.reverse()
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
            for i in range(len(x_batches)):
                # Train on mini-batches. prev_gradients is only used if momentum > 0.0.
                gradients, forward_dict = analytical_gradients(x_batches[i], y_batches[i], weights, l1_reg=l1_reg,
                                                               l2_reg=l2_reg)
                weights = optimizer(weights, gradients, learning_rate, layer_count, momentum, prev_gradients)
                prev_gradients = gradients
        else:
            # Train on full batch. prev_gradients is only used if momentum > 0.0.
            gradients, forward_dict = analytical_gradients(xs, ys, weights)
            weights = optimizer(weights, gradients, learning_rate, layer_count, momentum, prev_gradients)
            prev_gradients = gradients
        # Calculate error and loss on the training and test (if present) sets.
        train_loss, train_error = calculate_error_loss(xs, weights, ys)
        metrics["train_err"][epoch] = train_error
        metrics["train_loss"][epoch] = train_loss
        if train_error < best_error:
            # Save best weights.
            best_weights = copy.deepcopy(weights)
            best_error = train_error
        if test_xs is not None and test_ys is not None:
            test_loss, test_error = calculate_error_loss(test_xs, weights, test_ys)
            metrics["test_err"][epoch] = test_error
            metrics["test_loss"][epoch] = test_loss
            if print_metrics:
                print(
                    f"Epoch {epoch} - Training loss {train_loss} - Training error {train_error} - Test loss {test_loss}"
                    f" - Test error {test_error}")
    reset_seed()
    if return_best_weights:
        weights = best_weights
    if return_metrics:
        return weights, metrics
    return weights
