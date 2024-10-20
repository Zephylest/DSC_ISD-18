import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def custom_scaling(value, min_val, max_val):
    # Linear scaling between min_val and max_val
    return (value - min_val) / (max_val - min_val)

def preprocess_data(file_path, save_path):
    df = pd.read_csv(file_path)
    # Encoding independent variables (categorical, non boolean)
    # transformer 3rd parameter for index needed to transform into categorical data, remainder passthrough for passing through ignored data, otherwise deleted
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2, 5])], remainder='drop')
    clean_categorical = ct.fit_transform(df).toarray()
    categorical_df = pd.DataFrame(clean_categorical)

    # Drop original categorical columns (0 and 1)
    df = df.drop(df.columns[[2, 5]], axis=1)

    # Encoding dependent variables (boolean 0 1's)
    df['Status Pernikahan'] = df['Status Pernikahan'].apply(lambda x: 1 if x == 'Menikah' else 0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Pria' else 0)
    df['Kewarganegaraan'] = df['Kewarganegaraan'].apply(lambda x: 1 if x == 'WNI' else 0)
    df['Layak Bansos (Label)'] = df['Layak Bansos (Label)'].apply(lambda x: 1 if x == 'Layak' else 0)
    
    # Change column name and custom scaling values to liking, use loop if needed
    df['Penghasilan (Rp)'] = df['Penghasilan (Rp)'].apply(lambda x: custom_scaling(x, 0, 5000000))
    df['Umur (tahun)'] = df['Umur (tahun)'].apply(lambda x: custom_scaling(x, 18, 60))
    df['Jumlah Anak'] = df['Jumlah Anak'].apply(lambda x: custom_scaling(x, 0, 6))

    clean_df = pd.concat([categorical_df, df], axis=1)
    print(clean_df)
    clean_df.to_csv(save_path, index=False)


def main():
    preprocess_data('data/data_orang_jatim.csv', 'data/processed_data_orang_jatim.csv')

if __name__ == "__main__":
    main()