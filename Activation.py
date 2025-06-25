import numpy as np
import Loss as loss

class Activation_Linear:
    def predictions(self, outputs):
        return outputs
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

class Activation_ReLU:
    def predictions(self, outputs):
        return outputs
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_SoftMax:
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        self.exp_values = exp_values

    def backward(self, dvalues):
        self.dinputs = np.array([
            np.dot(
                np.diagflat(single_output) - np.dot(single_output.reshape(-1,1), single_output.reshape(-1,1).T),
                single_dvalues
            )
            for single_output, single_dvalues in zip(self.output, dvalues)
        ])

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = loss.Loss_CategoricalCrossentropy()

    def predictions(self, outputs):
        return outputs
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return (data_loss, self.regularization_loss()) if include_regularization else data_loss

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_true = np.argmax(y_true, axis=1) if len(y_true.shape) == 2 else y_true
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

    def regularization_loss(self):
        return sum(
            reg_type * np.sum(np.abs(param) if 'l1' in reg_name else param**2)
            for layer in self.trainable_layers
            for reg_name, reg_type in [
                ('weights_regularizer_l1', layer.weights_regularizer_l1),
                ('weights_regularizer_l2', layer.weights_regularizer_l2),
                ('biases_regularizer_l1', layer.biases_regularizer_l1),
                ('biases_regularizer_l2', layer.biases_regularizer_l2)
            ]
            if reg_type > 0
            for param in [layer.weights, layer.biases]
        )