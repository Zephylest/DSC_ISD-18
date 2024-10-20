import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

def predict_processed_data(model, data):
    prediction = model.predict(data)
    predicted_class = (prediction > 0.5)
    return predicted_class

def predict_raw_data(model, data):
    None
def predict_processed_data_csv(model, dataframe, save_path):
    predictions = model.predict(dataframe)
    predicted_class = (predictions > 0.5).astype(int)
    dataframe['Predicted_Layak'] = predicted_class
    dataframe.to_csv(save_path)

def predict_raw_data_csv(model, data, save_path):
    None

def main():
    # load model
    model = load_model('model/cnn_model_1.keras')
    model.load_weights('model/cnn_model_1.weights.h5')

    # data to predict
    # test dummy clean data
    data = np.array([[0.0,1.0,0.0,1.0,0.0,0.0,0.5,0.6577777777777778,1,1,1]])
    prediction = predict_processed_data(model, data)
    print(prediction)

    # test clean data csv
    df = pd.read_csv('data/clean_data_without_result.csv')
    predict_processed_data_csv(model,df,'data/prediction_result.csv')

if __name__ == "__main__":
    main()