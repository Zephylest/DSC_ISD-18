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
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='drop')
    clean_categorical = ct.fit_transform(df)
    categorical_df = pd.DataFrame(clean_categorical)

    # Drop original categorical columns (0 and 1)
    df = df.drop(df.columns[[0, 1]], axis=1)

    # Encoding dependent variables (boolean 0 1's)
    df['Boolean'] = df['Boolean'].apply(lambda x: 1 if x == 'yes' else 0)
    df['Boolean2'] = df['Boolean2'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Change column name and custom scaling values to liking, use loop if needed
    df['Numeric'] = df['Numeric'].apply(lambda x: custom_scaling(x, 0, 10))
    df['Numeric2'] = df['Numeric2'].apply(lambda x: custom_scaling(x, 1000, 10000))

    clean_df = pd.concat([categorical_df, df], axis=1)
    print(clean_df)
    clean_df.to_csv(save_path)

def main():
    preprocess_data('data/data_sample.csv', 'data/clean_data.csv')

if __name__ == "__main__":
    main()