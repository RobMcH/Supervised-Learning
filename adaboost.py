import numpy as np


def train_adaboost_ensemble(x, y, num_classifiers, base_classifier):
    observation_weights = np.ones(len(x)) / len(x)
    classifiers, classifier_weights = [], []
    for i in range(0, num_classifiers):
        base_classifier().fit(x, y, sample_weights=observation_weights)
        classifiers.append(base_classifier)
        classifier_weights.append(np.copy(observation_weights))
        predictions = (base_classifier.predict(x) == y).astype(np.float64)
        err = np.dot(observation_weights, predictions) / np.sum(observation_weights)
        alpha_i = np.log((1.0 - err) / err) + np.log(num_classifiers - 1)
        observation_weights *= np.exp(alpha_i * predictions)
        observation_weights /= np.sum(observation_weights)


def predict_adaboost_ensemble(x, weak_learners, observation_weights):
    predictions = [weak_learners[i].predict(x) * observation_weights[i] for i in range(len(weak_learners))]
    return np.argmax(np.array(predictions), axis=1)
