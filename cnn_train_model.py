import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import layers, models # type: ignore

def train_model(training_data_path, model_path, weights_path, test_size, epoch=10, batchsize=32):
    # read clean csv data
    df = pd.read_csv(training_data_path)

    # split df into input layers and target
    X = df.drop(columns=['Layak'])  # Replace 'target_column' with the actual target column name
    y = df['Layak']  # Binary target for classification

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # load cnn model
    model = load_model(model_path)

    # optional load model weights
    weights_file = weights_path
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    # recompile model with optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # training
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batchsize, validation_data=(X_test, y_test))

    model.save_weights(weights_file)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

def main():
    train_model('data/clean_data.csv', 'model/cnn_model_1.keras', 'model/cnn_model_1.weights.h5',0.1)

if __name__ == "__main__":
    main()