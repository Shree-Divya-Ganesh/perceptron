from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotfilename):

    df = pd.DataFrame(data)
    df

    x, y = prepare_data(df)

    model = Perceptron(eta = eta, epochs = epochs)
    model.fit(x,y)

    _ =  model.total_loss()


    save_model(model,filename)
    save_plot(df, plotfilename, model)

if __name__ == "__main__":
    OR = {"x1":[0, 0, 1, 1], 
        "x2":[0, 1, 0, 1],
        "y":[0, 1, 1, 1]}

    ETA = 0.3
    EPOCHS = 10

    main(data =OR, eta = ETA, epochs=EPOCHS, filename ="or_1.model", plotfilename="or_1.png")