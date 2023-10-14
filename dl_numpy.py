import numpy as np
import copy


class Tensor:
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)


class Function(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def getParams(self):
        return []


class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.parameters:
            p.grad = 0.0


class Linear(Function):
    def __init__(self, in_nodes, out_nodes):
        self.weights = Tensor((in_nodes, out_nodes))
        self.bias = Tensor((1, out_nodes))
        self.type = "linear"

        # print('w shape', self.weights.data)
        # print('b shape', self.bias.data.shape)

    def forward(self, x):
        # print('x', x.shape)
        output = np.dot(x, self.weights.data) + self.bias.data
        self.input = x
        return output

    def backward(self, d_y):
        # print('w grad', self.weights.grad)
        self.weights.grad += np.dot(self.input.T, d_y)
        self.bias.grad += np.sum(d_y, axis=0, keepdims=True)

        # print(d_y.shape, self.weights.data.T.shape)

        grad_input = np.dot(d_y, self.weights.data.T)
        return grad_input

    def getParams(self):
        return [self.weights, self.bias]


probs = []


class SoftmaxWithLoss(Function):
    def __init__(self):
        self.type = "normalization"

    def forward(self, x, target):
        unnormalized_proba = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.proba = unnormalized_proba / np.sum(
            unnormalized_proba, axis=1, keepdims=True
        )
        self.target = target
        prob = copy.copy(self.proba)
        # print(self.proba)
        probs.append(prob)

        loss = -np.log(self.proba[range(len(self.target)), self.target])
        return loss.mean()

    def backward(self):
        gradient = self.proba
        # print('before', gradient.shape )
        # print('int arr index', gradient[range(len(self.target)),self.target].shape )
        gradient[range(len(self.target)), self.target] -= 1.0
        # print('grad', gradient.shape )
        gradient /= len(self.target)
        return gradient


class ReLU(Function):
    def __init__(self, inplace=True):
        self.type = "activation"
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x[x < 0] = 0.0
            self.activated = x
        else:
            self.activated = x * (x > 0)

        return self.activated

    def backward(self, d_y):
        return d_y * (self.activated > 0)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, weight_decay=0.0, momentum=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = []
        for p in parameters:
            self.velocity.append(np.zeros_like(p.grad))

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            v = self.momentum * v + p.grad + self.weight_decay * p.data
            p.data = p.data - self.lr * v


##############################Library Implementation **END##############################
