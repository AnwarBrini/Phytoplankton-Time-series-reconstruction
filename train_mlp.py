from time import time
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess_flat import preprocess_flat_data
from models.mlp_model import mlp_1000_500

def split_scale_data(data):
    """
    Parameters
    ----------
    data : pandas dataframe
        OCCCI or GC flattened data

    Returns
    -------
    X_train : numpy array
    X_val : numpy array
    y_train : numpy array
    y_val : numpy array
    scaler : sklearn scaler object

    """

    data_cols = list(data.columns)
    data_cols.pop(0)
    output_data = np.log(data["hl_"])
    input_data = data[data_cols]
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data.values)
    X_train, X_val, y_train, y_val = train_test_split(scaled_input,
                                                      output_data, test_size=0.2,
                                                      random_state=0)
    return X_train, X_val, y_train, y_val, scaler

if __name__ == "__main__":
    gc_data = pd.read_csv("./data/gc_flattened_data.csv")
    gc_data = preprocess_flat_data(gc_data)
    X_train, X_val, y_train, y_val, scaler = split_scale_data(gc_data)
    MODEL = mlp_1000_500()
    TB = TensorBoard(log_dir="tensorboardlogs/PIX{}".format(time()))
    MODEL_CHECKPOINT = ModelCheckpoint('./models/mlp_1000_500_PIX.h5', monitor='val_loss')
    EARLY_STOP = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    print(MODEL.summary())
    MODEL.fit(X_train, y_train, verbose=1, batch_size=256, epochs=2000,
          validation_data=[X_val, y_val],
          callbacks=[TB, MODEL_CHECKPOINT, EARLY_STOP])
