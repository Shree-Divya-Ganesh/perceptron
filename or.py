from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

OR = {"x1":[0, 0, 1, 1], 
       "x2":[0, 1, 0, 1],
       "y":[0, 1, 1, 1]}

df_OR = pd.DataFrame(OR)
x_or, y_or = prepare_data(df_OR)

eta = 0.3
epochs = 10

model_or = Perceptron(eta, epochs)
model_or.fit(x_or, y_or)

_ =  model_or.total_loss()


save_model(model_or, "or.model")
save_plot(df_OR, "or.png", model_or)