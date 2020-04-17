from sklearn.neural_network import MLPRegressor
import numpy as np

import defaults
import utils
import logging

logger = logging.getLogger('translators')


class Translator:

    def __init__(self, project):
        self.project = project
        self.trainable = False

    def is_trainable(self, data):
        min_links = max(self.project.topics)
        if self.trainable:
            return True

        if len(data) >= self.project.omega * min_links:
            logger.info('Enough data to train translation matrix. Proceed with joined model.')
            self.trainable = True
            return True
        else:
            logger.info('Not enough data to train translation matrix. Proceed with changeset model only.')
            return False


class MLPR(Translator):

    def __init__(self, project):
        super().__init__(project)
        self.mlpr = None

    def fit(self, project, A, B):
        if self.mlpr is None:
            params = defaults.mlpr(len(A))
            if project.mlpr_neurons:
                neurons = int(max(project.topics) * project.mlpr_neurons)
                params['hidden_layer_sizes'] = (neurons, neurons)

            self.mlpr = MLPRegressor(**params).fit(B, A)
        else:
            self.mlpr.partial_fit(B, A)

    def predict(self, query):
        if self.mlpr is None:
            raise RuntimeError('Perceptron not trained!')

        return self.mlpr.predict(query.reshape(1, -1))[0]


class TMatrix(Translator):

    def __init__(self, project):
        super().__init__(project)
        self.T = None

    def fit(self, A, B):
        self.T, residue, rank, values = np.linalg.lstsq(B, A, rcond=None)

    def predict(self, query):
        if self.T is None:
            raise RuntimeError("T matrix not trained!")

        return utils.rescale(np.matmul(query, self.T))
