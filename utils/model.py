import numpy as np

class Perceptron:

  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4
    print(f"Initial weights before training : {self.weights}")
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z = W * X
    return np.where(z > 0, 1, 0)

  def fit(self, x, y):
    self.x = x
    self.y = y

    x_with_bias = np.c_[self.x,  -np.ones((len(self.x), 1))]
    print(f"X with bias: \n{x_with_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(x_with_bias, self.weights) #Forward Propogation
      print(f"Predicted Output after Forward Pass: \n{y_hat}")
      self.error = self.y - y_hat
      print(f"Error: \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error) #Backward Propogation
      print(f"Update weights after epoch:\n{epoch}/{self.epochs} :\n{self.weights}")
      print("####"*10)

  def predict(self, x):
    x_with_bias = np.c_[x,  -np.ones((len(x), 1))]
    return self.activationFunction(x_with_bias, self.weights)


  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total Loss: {total_loss}")
    return total_loss