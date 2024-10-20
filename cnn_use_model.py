import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def custom_scaling(value, min_val, max_val):
    # Linear scaling between min_val and max_val
    return (value - min_val) / (max_val - min_val)

def predict_processed_data(model, data):
    prediction = model.predict(data)
    predicted_class = (prediction > 0.5)
    return predicted_class

def predict_raw_data(model, data):
    df = pd.read_csv('data/data_orang_jatim_without_result.csv')
    
    # Append the new data to the DataFrame
    new_row = pd.DataFrame(data, columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)

    # Preprocess the data as you did before
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2, 5])], remainder='drop')
    clean_categorical = ct.fit_transform(df).toarray()
    categorical_df = pd.DataFrame(clean_categorical)

    # Drop original categorical columns
    df = df.drop(df.columns[[2, 5]], axis=1)

    # Encode dependent variables (0 and 1)
    df['Status Pernikahan'] = df['Status Pernikahan'].apply(lambda x: 1 if x == 'Menikah' else 0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Pria' else 0)
    df['Kewarganegaraan'] = df['Kewarganegaraan'].apply(lambda x: 1 if x == 'WNI' else 0)
    
    # Scale the numeric columns
    df['Penghasilan (Rp)'] = df['Penghasilan (Rp)'].apply(lambda x: custom_scaling(int(x), 0, 5000000))
    df['Umur (tahun)'] = df['Umur (tahun)'].apply(lambda x: custom_scaling(int(x), 18, 60))
    df['Jumlah Anak'] = df['Jumlah Anak'].apply(lambda x: custom_scaling(int(x), 0, 6))

    clean_df = pd.concat([categorical_df, df], axis=1)

    # Select the last row for prediction
    last_row = clean_df.iloc[-1].values.reshape(1, -1)  # Reshape to (1, number_of_features)

    print(clean_df)

    # Make the prediction
    prediction = model.predict(last_row)
    predicted_class = (prediction > 0.5)  # Assuming binary classification with sigmoid
    
    return predicted_class

def predict_processed_data_csv(model, dataframe, save_path):
    predictions = model.predict(dataframe)
    predicted_class = (predictions > 0.5).astype(int)
    dataframe['Predicted_Layak'] = predicted_class
    dataframe.to_csv(save_path, index=False)

def predict_raw_data_csv(model, data, save_path):
    # Read training data
    df_train = pd.read_csv('data/data_orang_jatim_without_result.csv')

    # Initialize the ColumnTransformer and fit on training data
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [2, 5])], remainder='drop')
    clean_categorical_train = ct.fit_transform(df_train).toarray()
    
    # Transform the new data using the fitted ColumnTransformer
    clean_categorical = ct.transform(data).toarray()
    categorical_df = pd.DataFrame(clean_categorical)

    raw_df = data

    # Drop original categorical columns from the new data
    data = data.drop(data.columns[[2, 5]], axis=1)

    # Encode dependent variables (0 and 1)
    data['Status Pernikahan'] = data['Status Pernikahan'].apply(lambda x: 1 if x == 'Menikah' else 0)
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Pria' else 0)
    data['Kewarganegaraan'] = data['Kewarganegaraan'].apply(lambda x: 1 if x == 'WNI' else 0)

    # Scale the numeric columns
    data['Penghasilan (Rp)'] = data['Penghasilan (Rp)'].apply(lambda x: custom_scaling(int(x), 0, 5000000))
    data['Umur (tahun)'] = data['Umur (tahun)'].apply(lambda x: custom_scaling(int(x), 18, 60))
    data['Jumlah Anak'] = data['Jumlah Anak'].apply(lambda x: custom_scaling(int(x), 0, 6))

    # Concatenate the transformed categorical data with the processed numeric data
    clean_df = pd.concat([categorical_df, data], axis=1)

    # Make predictions
    predictions = model.predict(clean_df)
    predicted_class = (predictions > 0.5).astype(int)
    raw_df['Predicted_Layak'] = predicted_class

    # Save to CSV
    raw_df.to_csv(save_path, index=False)

def main():
    # load model
    model = load_model('model/cnn_model_1.keras')
    model.load_weights('model/cnn_model_1.weights.h5')

    # data to predict
    # test dummy singular clean data
    data = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.7,1,1,0.5238095238095238,1,0.3333333333333333]])
    prediction = predict_processed_data(model, data)
    print(prediction)

    # test clean data csv
    df = pd.read_csv('data/processed_data_orang_jatim_before_prediction.csv')
    predict_processed_data_csv(model,df,'data/processed_data_orang_jatim_prediction.csv')

    # test dummy singular raw data
    raw_data = np.array([[2200000, 'Menikah', 'Jember', 'Wanita', 45, 'Sarjana', 'WNI', 1]])
    predicted_raw_data = predict_raw_data(model,raw_data)
    print(predicted_raw_data)

    # test dummy csv raw data
    df_raw = pd.read_csv('data/data_orang_jatim_without_result.csv')
    predict_raw_data_csv(model, df_raw, 'data/data_orang_jatim_raw_result.csv')

if __name__ == "__main__":
    main()