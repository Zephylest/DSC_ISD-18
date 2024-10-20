import numpy as np
import pandas as pd
from tensorflow.keras import layers, models # type: ignore

def initialize_cnn_model(path_to_csv, hfirst_layer_neuron, hsecond_layer_neuron, hthird_layer_neuron, save_path):
    # read clean csv data
    df = pd.read_csv(path_to_csv)
    print(df)

    # build CNN model
    model = models.Sequential()

    # input layer: input shape = number of columns
    model.add(layers.InputLayer(input_shape=(df.shape[1],)))

    # hidden layer 1: Dense + ReLU activation
    model.add(layers.Dense(hfirst_layer_neuron, activation='relu'))

    # hidden Layer 2: Dense + ReLU activation
    model.add(layers.Dense(hsecond_layer_neuron, activation='relu'))

    # hidden Layer 3: Dense + ReLU activation
    model.add(layers.Dense(hthird_layer_neuron, activation='relu'))

    # output Layer (1 neuron, sigmoid for binary)
    model.add(layers.Dense(1, activation='sigmoid'))

    model.save(save_path)

def main():
    initialize_cnn_model('data/processed_data_orang_jatim_without_result.csv', 64, 32, 16, 'model/cnn_model_1.keras')

if __name__ == "__main__":
    main()