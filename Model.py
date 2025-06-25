import numpy as np
import pickle

class Input_Layer:
    def forward(self, inputs):
        self.output = inputs

class Model:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Input_Layer()
        layers_count = len(self.layers)

        for i in range(layers_count):
            if i == 0:
                self.layers[i].previous = self.input_layer
                self.layers[i].next = self.layers[i+1] if layers_count > 1 else self.loss
            elif i < layers_count - 1:
                self.layers[i].previous = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].previous = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.previous.output)
        return layer.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, validation_data=None, epochs=1, print_every=1):
        self.accuracy.init(y)
    
        for epoch in range(1, epochs+1):
            output = self.forward(X)
            data_loss = self.loss.calculate(output, y, include_regularization=False)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
        
            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {data_loss:.3f}, LR: {self.optimizer.current_learning_rate:.3f}')

        if validation_data is not None:
            output = self.forward(validation_data.P)
            loss = self.loss.calculate(output, validation_data.L, include_regularization=False)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, validation_data.L)
            print(f'validation, accuracy: {accuracy:.3f}, loss: {loss:.3f}')

    def get_parameters(self):
        return [layer.get_parameters() for layer in self.trainable_layers]
    
    def set_parameters(self, parameters):
        for parameters_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameters_set)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))