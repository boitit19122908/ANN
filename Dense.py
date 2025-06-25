import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
           
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights, self.biases = weights, biases

class Layer_Dense_Regularization:
    def __init__(self, n_inputs, n_neurons, 
                 weights_regularizer_l1=0, weights_regularizer_l2=0, 
                 biases_regularizer_l1=0, biases_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Regularization parameters
        self.weights_regularizer_l1 = weights_regularizer_l1
        self.weights_regularizer_l2 = weights_regularizer_l2
        self.biases_regularizer_l1 = biases_regularizer_l1
        self.biases_regularizer_l2 = biases_regularizer_l2

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Regularization gradients
        if self.weights_regularizer_l1 > 0:
            dL1 = np.where(self.weights < 0, -1, 1)
            self.dweights += self.weights_regularizer_l1 * dL1

        if self.weights_regularizer_l2 > 0:
            self.dweights += 2 * self.weights_regularizer_l2 * self.weights

        if self.biases_regularizer_l1 > 0:
            dL1 = np.where(self.biases < 0, -1, 1)
            self.dbiases += self.biases_regularizer_l1 * dL1

        if self.biases_regularizer_l2 > 0:
            self.dbiases += 2 * self.biases_regularizer_l2 * self.biases

        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights, self.biases = weights, biases

class Dense_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate  # invert the rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask