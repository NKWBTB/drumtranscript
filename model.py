import keras
from keras.layers import LSTM, Dense, Input, Bidirectional
from keras.models import Model
import config as cfg
import os
from data import read_sample
from utils import list_files
import numpy as np


def build_model(architecture, input_shape=(None, cfg.FRAME_SIZE)):
    input = Input(shape=input_shape, dtype='float32')
    x = None
    if architecture == 'Bi-LSTM':
        x = Bidirectional(LSTM(128, return_sequences=True))(input)
        x = Dense(128, activation='relu')(x)
    elif architecture == 'LSTM':
        x = LSTM(128, return_sequences=True)(input)
        x = Dense(128, activation='relu')(x)
    
    preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
    model = Model(input, preds)
    return model

def data_generator(filelist):
    while 1:
        for file in filelist:
            sample = read_sample(file)
            x = [np.array([sample['Frames']], dtype=np.float32)]
            y = [np.array([sample['Activation']], dtype=int)]
            yield (x, y)

def last_epoch(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    epochs = [int(epoch.split('.')[0]) for epoch in checkpoints]
    epochs.append(0)
    last = max(epochs)
    return last

def train(architecture='Bi-LSTM'):
    num_epochs = 60
    # checkpoint settings
    checkpoint_path = "models/" + architecture + "/{epoch:d}.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1)

    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = list_files(val_path, 'pickle')
    
    model = None
    init_epoch = 0
    if not os.path.exists(checkpoint_dir) or last_epoch(checkpoint_dir) == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model = build_model(architecture)    
        optimizer = keras.optimizers.RMSprop()
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        init_epoch = last_epoch(checkpoint_dir)
        print('\nLoad existing model from epoch %i\n' % init_epoch)
        model = keras.models.load_model(os.path.join(checkpoint_dir, str(init_epoch)+'.h5'))
    
    model.summary()
    history = model.fit_generator(data_generator(train_files), epochs=num_epochs, initial_epoch=init_epoch, steps_per_epoch=len(train_files), 
                        validation_data=data_generator(val_files), validation_steps=len(val_files),
                        callbacks=[cp_callback, keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                min_delta=0,
                                                                patience=3,
                                                                verbose=0,
                                                                mode='auto')])
    print(history)

if __name__ == "__main__":
    train()