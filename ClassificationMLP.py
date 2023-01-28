from sklearn import datasets
import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class Network:
    # Create Directed Acyclic Network of given number layers.
    def __init__(self, hidden_layer_size, X, y, learning_rate, epochs):
        self.training_set = X
        self.training_outputs = y
        # Set outputs to be the unique values in the training set
        self.outputs = np.unique(y)
        # Create a list of layers, each layer is a list of nodes
        self.layers = [np.zeros((len(X[0]), 1)), np.zeros((
            hidden_layer_size, len(X[0]))), np.zeros((len(self.outputs), 1))]
        self.learning_rate = learning_rate
        self.epochs = epochs
        # initialize each weight with the values min_value=-0.5, max_value=0.5,
        self.weights = [np.zeros(
            (len(X[0]), hidden_layer_size)), np.zeros((hidden_layer_size, len(self.outputs)))]

    def init_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.uniform(-0.5,
                                                0.5, np.shape(self.weights[i]))

    def feedforward(self, example):
        # set the input layer to the example
        for i, xi in enumerate(example):
            self.layers[0][i] = xi

        # for each layer, calculate the value of each node wrt weights
        for l, layer in enumerate(self.layers):
            if l == 0:
                continue
            for j, node in enumerate(layer):
                inj = 0
                for i, w in enumerate(self.weights[l-1]):
                    inj += w[j] * self.layers[l-1][i][0]
                # inj = np.dot(self.weights[l-1][:, j], self.layers[l-1][:, 0])
                self.layers[l][j] = sigmoid(inj)

    def backpropogation(self, output):
        delta = [[]]
        # calculate the error for the output layer
        for i, node in enumerate(self.layers[-1]):
            delta[-1].append(
                (output[i] - node[0]) * sigmoid_derivative(node[0]))

        # calculate the error for the hidden layers
        for l in range(len(self.layers)-2, 0, -1):
            layer = self.layers[l]
            delta.insert(0, [])
            for j, node in enumerate(layer):
                inj = 0
                for i, w in enumerate(self.weights[l]):
                    inj += w[j] * delta[1][i]
                delta[0].append(sigmoid_derivative(node[0]) * inj)

        # update the weights
        for l, layer in enumerate(self.layers):
            if l == len(self.layers)-1:
                continue
            for i, node in enumerate(layer):
                for j, d in enumerate(delta[l]):
                    add = self.learning_rate * d * self.layers[l][i][0]
                    self.weights[l][i][j] += add

    def BackPropagationLearner(self):
        # initialize each weight with the values min_value=-0.5, max_value=0.5,
        self.init_weights()

        for epoch in range(self.epochs):
            for example_index, example in enumerate(self.training_set):
                self.feedforward(example)
                # create a vector of the correct output
                output = np.zeros((len(self.outputs), 1))
                output[self.training_outputs[example_index]] = 1
                self.backpropogation(output)

        return


def NeuralNetLearner(X, y, hidden_layer_sizes=None, learning_rate=0.01, epochs=100):
    """
    Layered feed-forward network.
    hidden_layer_sizes: List of number of hidden units per hidden layer if None set 3
    learning_rate: Learning rate of gradient descent
    epochs: Number of passes over the dataset
    activation:sigmoid
        """

    if hidden_layer_sizes is None:
        hidden_layer_sizes = 3

    # construct a raw network and call BackPropagationLearner
    net = Network(hidden_layer_sizes, X, y, learning_rate, epochs)
    net.BackPropagationLearner()

    def predict(example):
        net.feedforward(example)
        # return the index of the node with the highest value
        return np.argmax(net.layers[-1])

    return predict


iris = datasets.load_iris()
X = iris.data
y = iris.target


nNL = NeuralNetLearner(X, y)
# print([4.6, 3.1, 1.5, 0.2], "->", nNL([4.6, 3.1, 1.5, 0.2]))  # 0
# print([6.5, 3., 5.2, 2.], "->", nNL([6.5, 3., 5.2, 2.]))  # 2
