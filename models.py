import tensorflow as tf
import keras
import numpy as np
import pdb


def beh_LSTM():
    data = np.load("beh_vec_v100.npy",allow_pickle=True)
    labels = np.load("labels.npy",allow_pickle = True)
    data = keras.preprocessing.sequence.pad_sequences(data,maxlen=max([len(i) for i in data]),value=-100)
    # input_dim = 5651
    # data (6106,2,5651)
    input_dim = 8601
    # data (6106,2,8601)
    data = data.reshape((data.shape[0],2,input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-100))
    model.add(keras.layers.LSTM(256,dropout_W=0.2, dropout_U=0.2, input_shape=(data.shape[0],2,input_dim)))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy']) 
    model.fit(data, labels, epochs=50, batch_size=512,verbose=1)
    score = model.evaluate(data, labels,batch_size=256, verbose=1)
    print(score)

    # pdb.set_trace()


if __name__ == "__main__":
    beh_LSTM()