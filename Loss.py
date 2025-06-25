import numpy as np

class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        data_loss = np.mean(self.forward(output, y))
        return (data_loss, self.regularization_loss()) if include_regularization else data_loss
    
    def regularization_loss(self):
        return sum(
            layer.weights_regularizer_l1 * np.sum(np.abs(layer.weights)) +
            layer.weights_regularizer_l2 * np.sum(layer.weights**2) +
            layer.biases_regularizer_l1 * np.sum(np.abs(layer.biases)) +
            layer.biases_regularizer_l2 * np.sum(layer.biases**2)
            for layer in self.trainable_layers 
            if any([
                layer.weights_regularizer_l1 > 0, 
                layer.weights_regularizer_l2 > 0, 
                layer.biases_regularizer_l1 > 0, 
                layer.biases_regularizer_l2 > 0
            ])
        )

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(len(y_pred)), y_true]
        else:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        
        return -np.log(correct_confidence)

    def backward(self, dvalues, y_true):
        samples, labels = len(dvalues), len(dvalues[0])
        y_true = np.eye(labels)[y_true] if len(y_true.shape) == 1 else y_true
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred)**2, axis=-1)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = -2 * (y_true - dvalues) / len(dvalues)
        self.dinputs /= samples