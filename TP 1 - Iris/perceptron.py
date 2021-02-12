import numpy as np

class Perceptron:
    def __init__(self, dimension, max_iter, learning_rate):
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.weights = np.random.randn(120, self.dimension)
        self.bias = np.random.randn(120, self.dimension)

    def sigmoid(self, z):
        s = 1/(1 + np.exp(-z))
        return s

    def cost(self, X, Y, A):
        m = X.shape[1]
        return (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    def forward(self, X, W, b):
        X = np.array(X)
        z = np.dot(W.T, X) + b
        return self.sigmoid(z)

    def propagate(self, W, b, X, Y):
        A = self.forward(X, W, b)
        grad, cost = self.backpropagation(X, Y, A)
        return grad, cost

    def backpropagation(self, X, Y, A):
        m = X.shape[1]
        dW = (1/m)*np.dot(X, (A-Y).T)
        db = (1/m)*np.sum(A-Y)

        cost = np.squeeze(self.cost(X,Y, A))

        grad = {"dW":dW, "db": db}

        return grad, cost

    def initialize_weights(self, dim):
        W = np.random.rand(dim,1).flatten()
        b = 0
        return W, b

    def fit(self, X, Y, verbose=False):
        W, b = self.initialize_weights(X.shape[0])

        costs = []


        for i in range(self.max_iter):
            grad, cost = self.propagate(W, b, X, Y)

            dW = grad["dW"]
            db = grad["db"]

            W = W - self.learning_rate * dW
            b = b - self.learning_rate * db

            if i%1000 == 0: 
                costs.append(cost)

            if verbose and i%1000 == 0:
                print("Cout apres iteration %i: %f" % (i, cost))

        return W, b, costs
