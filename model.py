import keras
from keras.layers import LSTM, Dense, Input
from keras.models import Model
import config as cfg
import os
from data import read_sample
from utils import list_files
import numpy as np


def build_model():
    hidden_size = 128
    input = Input(shape=(None, cfg.FRAME_SIZE), dtype='float32')
    x = LSTM(hidden_size, return_sequences=True)(input)
    x = Dense(128, activation='relu')(x)
    preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
    model = Model(input, preds)

    return model

def train():
    model = build_model()
    model.summary()

    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = list_files(val_path, 'pickle')
    
    def data_generator(filelist):
        while 1:
            for file in filelist:
                sample = read_sample(os.path.join(train_path, file))
                x = [np.array([sample['Frames']], dtype=np.float32)]
                y = [np.array([sample['Activation']], dtype=int)]
                yield (x, y)


    optimizer = keras.optimizers.RMSprop()
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 60
    history = model.fit_generator(data_generator(train_files), steps_per_epoch=len(train_files), epochs=num_epochs,
                        validation_data=data_generator(val_files), validation_steps=len(val_files),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                min_delta=0,
                                                                patience=3,
                                                                verbose=0,
                                                                mode='auto')])
    print(history)

if __name__ == "__main__":
    train()