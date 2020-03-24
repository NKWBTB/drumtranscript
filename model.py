import keras
from keras.layers import LSTM, Dense, Input, Bidirectional
from keras.models import Model
import config as cfg
import os
from data import read_sample
from utils import list_files
import numpy as np

class BaseModel:
    def __init__(self):
        pass

    def train(self):
        ''' Train the model '''
        raise NotImplementedError

    def predict(self, input):
        ''' Predict the activation and onset using the model

        Returns:
            activation, onset(optional, default=None)
        '''
        raise NotImplementedError

    def save(self, path):
        ''' Save the model to the path'''
        raise NotImplementedError

    def load(self, path):
        ''' Load the model from the path'''
        raise NotImplementedError


class BiLSTM(BaseModel):
    def __init__(self):
        self.checkpoint_path = "models/Bi-LSTM/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None

    def build_model(self, input_shape=(None, cfg.FRAME_SIZE)):
        input = Input(shape=input_shape, dtype='float32')
        x = Bidirectional(LSTM(128, return_sequences=True))(input)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

    def data_generator(self, filelist):
        while 1:
            for file in filelist:
                sample = read_sample(file)
                x = [np.array([sample['Frames']], dtype=np.float32)]
                y = [np.array([sample['Activation']], dtype=int)]
                yield (x, y)
    
    def last_epoch(self):
        checkpoints = os.listdir(self.checkpoint_dir)
        epochs = [int(epoch.split('.')[0]) for epoch in checkpoints]
        epochs.append(0)
        last = max(epochs)
        return last

    def train(self, train_files, val_files):
        num_epochs = 60
        # checkpoint settings
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, 
            verbose=1)
        
        model = None
        init_epoch = 0
        if not os.path.exists(self.checkpoint_dir) or self.last_epoch() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.build_model()    
            optimizer = keras.optimizers.RMSprop()
            self.model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            init_epoch = self.last_epoch()
            print('\nLoad existing model from epoch %i\n' % init_epoch)
            self.load(os.path.join(self.checkpoint_dir, str(init_epoch)+'.h5'))
        
        self.model.summary()
        history = self.model.fit_generator(self.data_generator(train_files), 
                                        epochs=num_epochs, initial_epoch=init_epoch, steps_per_epoch=len(train_files), 
                                        validation_data=self.data_generator(val_files), validation_steps=len(val_files),
                                        callbacks=[cp_callback, keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                            min_delta=0,
                                                                                            patience=3,
                                                                                            verbose=0,
                                                                                            mode='auto')])
        return history

    def predict(self, frames, threshold=0.5):
        activation = model.predict(frames)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

class SimpleLSTM(BiLSTM):
    def __init__(self):
        self.checkpoint_path = "models/LSTM/{epoch:d}.h5"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.model = None
    
    def build_model(self, input_shape=(None, cfg.FRAME_SIZE)):
        input = Input(shape=input_shape, dtype='float32')
        x = LSTM(128, return_sequences=True)(input)
        x = Dense(128, activation='relu')(x)
        preds = Dense(cfg.PITCH_NUM, activation='sigmoid')(x)
        self.model = Model(input, preds)

if __name__ == "__main__":
    train_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'train')
    train_files = list_files(train_path, 'pickle')
    val_path = os.path.join(cfg.SEQ_SAMPLE_PATH, str(cfg.FRAME_TIME) + '_ms', 'validation')
    val_files = list_files(val_path, 'pickle')
    
    model = BiLSTM()
    
    #model.train(train_files, val_files)
    # train('Bi-LSTM')