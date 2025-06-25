#%%
import numpy as np
import Activation as activation # type: ignore
import Dense as layer # type: ignore
import Optimization as optimization # type: ignore
import Points as point # type: ignore
import matplotlib.pyplot as plt
import Model as NNModel # type: ignore
import Loss as loss_function
import Accuracy as accuracy # type: ignore
#%%
n_points = 1000
Training_Data = point.Spiral(n_points,3,2)
Validation_Data = point.Spiral(n_points,3,2)

model = NNModel.Model()

model.add(layer.Dense(2,64))
model.add(activation.Activation_ReLU())
model.add(layer.Dense(64,64))
model.add(activation.Activation_ReLU())
model.add(layer.Dense(64,3))
model.add(activation.Activation_ReLU())
model.set(
    loss = activation.Activation_Softmax_Loss_CategoricalCrossEntropy(),
    optimizer = optimization.Optimizer_Adam(learning_rate = 0.03, decay = 1e-4),
    accuracy = accuracy.Accuracy_Categorical()
)
model.finalize()
model.train(Training_Data.P, Training_Data.L, Validation_Data, epochs = 1000, print_every = 100)

parameters = model.get_parameters()

output = model.forward(Training_Data.P)
predictions = np.argmax(output, axis = 1)
# Visualization of data (2D):
plt.figure(1)
plt.scatter(Training_Data.P[:, 0], Training_Data.P[:, 1], c=Training_Data.L, s=20, cmap=plt.cm.Spectral)
plt.title("Training data")

plt.figure(2)
plt.scatter(Training_Data.P[:, 0], Training_Data.P[:, 1], c=predictions, s=20, cmap=plt.cm.Spectral)
plt.title("Training Result")
plt.show()
# %%
